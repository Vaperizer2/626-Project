"""The final, deployable classifier: frozen SigLIP So400m image tower + two linear heads, fused.

Why this design (see results/benchmark.csv): on data v2 the best *frozen* web-pretrained vision-language backbone
(SigLIP So400m) with a logistic-regression head scored higher than every fine-tuned network we trained. The
backbone's own zero-shot text classifier is fused in as a second head. That fusion was chosen from a Phase-1
analysis (+0.9 pt OOF) whose zero-shot scores used open_clip's slightly different preprocessing; with this single
deployable pipeline the gain is ~0 (exports/final/fit_report.json: 0.826 vs 0.825 OOF accuracy) and on the sealed
test the probe alone scored 0.844 vs 0.837 fused -- a tie within sampling error. It is kept because it was
pre-registered and costs nothing (a 20 x 1152 matrix). Both heads are linear on the same 1152-d image feature,
so the whole model is one network:

    image -> SigLIP So400m/14 @384 (frozen) -> f (1152)
          -> probe head:      log_softmax(W_p * standardise(f) + b_p)          (trained on dev rows)
          -> zero-shot head:  log_softmax(T_zs * cos(f, text embedding of "a photo of a <class>"...))
          -> fused = probe + w_zs * zero-shot;  logits = fused / temperature;  probs = softmax(logits)

Subcommands (from the repo root; feature extraction wants a CUDA GPU):
    uv run python -m compclass.final fit                 # OOF check of the exact exported pipeline, calibration,
                                                         # abstain threshold, refit head on all dev rows
    uv run python -m compclass.final export              # ONNX + labels/preprocess/metadata + parity checks
    uv run python -m compclass.final test --confirm      # THE one look at the sealed test split
    uv run python -m compclass.final test --manifest data/web_holdout_v1/manifest.json --confirm
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import ImageDataset, build_transforms, get_dev, get_fold, get_test, load_manifest
from .evaluate import compute_metrics
from .features import extract_features, fit_probe
from .models import build_model

KEY = "siglip_so400m_384"
T_ZS, W_ZS = 50.0, 0.25  # fixed a priori from the Phase-1 OOF analysis (results/phase1/fusion_ablation.csv)
OUT = Path("exports/final")


def text_weights(classes: list[str], device: torch.device) -> np.ndarray:
    """[C, D] L2-normalised, template-averaged SigLIP text embeddings (same prompts as zeroshot.py)."""
    import open_clip

    from .models.registry import REGISTRY
    from .zeroshot import text_classifier

    arch, tag = REGISTRY[KEY].zeroshot
    model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=tag, device=device)
    model.eval()
    with torch.no_grad():
        W = text_classifier(model, open_clip.get_tokenizer(arch), classes, device)
    return W.cpu().numpy().astype(np.float32)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def fused_scores(F: np.ndarray, head: dict) -> np.ndarray:
    """Un-tempered fused log-scores [N, C] from raw features (numpy mirror of FusedClassifier)."""
    z = (F - head["mu"]) / head["sd"]
    lp = log_softmax(z @ head["coef"].T + head["intercept"])
    fn = F / np.linalg.norm(F, axis=1, keepdims=True)
    return lp + head["w_zs"] * log_softmax(head["t_zs"] * fn @ head["W"].T)


def head_from_search(search, W: np.ndarray) -> dict:
    sc, lr = search.best_estimator_[0], search.best_estimator_[-1]
    return {"mu": sc.mean_.astype(np.float32), "sd": sc.scale_.astype(np.float32),
            "coef": lr.coef_.astype(np.float32), "intercept": lr.intercept_.astype(np.float32),
            "W": W, "t_zs": np.float32(T_ZS), "w_zs": np.float32(W_ZS), "C": float(search.best_params_[
                "logisticregression__C"])}


def fit_temperature(scores: np.ndarray, y: np.ndarray) -> float:
    """Temperature that minimises the negative log-likelihood of softmax(scores / T) (Guo et al., 2017)."""
    from scipy.optimize import minimize_scalar

    def nll(t: float) -> float:
        return -log_softmax(scores / t)[np.arange(len(y)), y].mean()

    return float(minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded").x)


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error: |accuracy - confidence| averaged over confidence bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def fit_head(F: np.ndarray, m, dev: list, W: np.ndarray) -> tuple[dict, dict]:
    """The deployed head on dev features ``F`` (one row per ``dev`` row) and text embeddings ``W``: out-of-fold scores
    of the exact exported pipeline, temperature and abstain threshold fitted on them, then the probe refitted on all
    dev rows. Returns (head arrays for head.npz, fit report)."""
    y = np.array([m.class_to_idx[r["label"]] for r in dev])
    pos = {r["id"]: i for i, r in enumerate(dev)}
    oof = np.zeros((len(dev), len(m.classes)), np.float32)
    oof_probe = np.zeros(len(dev), int)
    for k in range(5):  # out-of-fold scores of the exact exported pipeline
        tr, va = get_fold(m, k)
        itr, iva = [pos[r["id"]] for r in tr], [pos[r["id"]] for r in va]
        s = fit_probe(F[itr], y[itr], np.array([r["cluster"] for r in tr]))
        oof[iva] = fused_scores(F[iva], head_from_search(s, W))
        oof_probe[iva] = s.predict(F[iva])
    temp = fit_temperature(oof, y)
    probs = np.exp(log_softmax(oof / temp))
    pred, conf = probs.argmax(1), probs.max(1)
    ood = np.array(["ood_suspect" in r["flags"] for r in dev])
    # abstain threshold: smallest confidence at which accepted predictions are >= 90% correct (on OOF scores)
    grid = np.round(np.arange(0.0, 0.96, 0.05), 2)
    table = [{"threshold": float(t), "coverage": float((conf >= t).mean()),
              "accepted_acc": float((pred[conf >= t] == y[conf >= t]).mean()) if (conf >= t).any() else None}
             for t in grid]
    thr = next((r["threshold"] for r in table if r["accepted_acc"] and r["accepted_acc"] >= 0.90), 0.5)
    search = fit_probe(F, y, np.array([r["cluster"] for r in dev]))  # final head on ALL dev rows
    head = head_from_search(search, W)
    head["temperature"], head["abstain_threshold"] = np.float32(temp), np.float32(thr)
    met = compute_metrics(y, pred, m.classes)
    report = {"oof_accuracy": met["accuracy"], "oof_macro_f1": met["macro_f1"],
              "oof_inscope_accuracy": float((pred[~ood] == y[~ood]).mean()),
              "oof_probe_only_accuracy": float((oof_probe == y).mean()), "temperature": temp,
              "ece_before": ece(np.exp(log_softmax(oof)), y), "ece_after": ece(probs, y),
              "abstain_threshold": thr, "abstain_table": table, "final_C": head["C"],
              "t_zs": T_ZS, "w_zs": W_ZS, "manifest": m.fingerprint, "n_dev": len(dev)}
    return head, report


def cmd_fit(manifest: str) -> dict:
    m = load_manifest(manifest)
    dev = get_dev(m)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head, report = fit_head(extract_features(KEY, dev, m), m, dev, text_weights(m.classes, device))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "head.npz", **head)
    (OUT / "fit_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "abstain_table"}, indent=1))
    return report


class FusedClassifier(nn.Module):
    """Backbone + both heads + temperature, as one exportable network. Outputs (logits, probs)."""

    def __init__(self, trunk: nn.Module, head: dict):
        super().__init__()
        self.trunk = trunk
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
        for name in ("mu", "sd", "coef", "intercept", "W"):
            self.register_buffer(name, t(head[name]))
        self.t_zs, self.w_zs, self.temperature = float(head["t_zs"]), float(head["w_zs"]), float(head["temperature"])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.trunk(x)
        lp = torch.log_softmax(((f - self.mu) / self.sd) @ self.coef.T + self.intercept, dim=1)
        fn = f / f.norm(dim=1, keepdim=True)
        zs = torch.log_softmax(self.t_zs * fn @ self.W.T, dim=1)
        logits = (lp + self.w_zs * zs) / self.temperature
        return logits, torch.softmax(logits, dim=1)


def load_final(head_path: Path = OUT / "head.npz") -> tuple[FusedClassifier, object]:
    head = dict(np.load(head_path))
    trunk, spec = build_model(KEY, num_classes=0, pretrained=True)
    return FusedClassifier(trunk, head).eval(), spec


def cmd_export(manifest: str, n_check: int = 32, force: bool = False) -> dict:
    import onnxruntime as ort

    m = load_manifest(manifest)
    model, spec = load_final()
    size = spec.input_size
    onnx_path = OUT / "model.onnx"
    x0 = torch.randn(1, 3, size, size)
    dyn = ({0: torch.export.Dim("batch", min=1, max=64)},)
    external = (OUT / "model.onnx.data").exists()
    # The head (probe weights, temperature, threshold) is baked into the ONNX graph. Record which head.npz an
    # export was built from, and re-export whenever it changed, so model.onnx and preprocess.json never disagree.
    import hashlib

    head_sha = hashlib.sha256((OUT / "head.npz").read_bytes()).hexdigest()
    stamp = OUT / "model.onnx.head_sha256"
    stale = not stamp.exists() or stamp.read_text().strip() != head_sha
    if force or not onnx_path.exists() or stale:
        kw = dict(input_names=["input"], output_names=["logits", "probs"], opset_version=18, dynamo=True,
                  dynamic_shapes=dyn)
        try:
            torch.onnx.export(model, (x0,), onnx_path, external_data=False, **kw)
            external = False
        except Exception as e:  # > 2 GB protobuf limit -> weights go to model.onnx.data
            print("single-file export failed, using external data:", repr(e)[:200])
            torch.onnx.export(model, (x0,), onnx_path, external_data=True, **kw)
            external = True
        stamp.write_text(head_sha)
    else:
        print("reusing existing", onnx_path, "(built from the current head.npz)")
    (OUT / "labels.json").write_text(json.dumps(m.classes, indent=1))
    head = dict(np.load(OUT / "head.npz"))
    pre = {**spec.to_dict(), "color_order": "RGB", "layout": "NCHW", "scale": 1 / 255,
           "outputs": ["logits", "probs"], "abstain_threshold": float(head["abstain_threshold"]),
           "temperature": float(head["temperature"]), "model": KEY, "t_zs": T_ZS, "w_zs": W_ZS,
           "manifest_fingerprint": m.fingerprint}
    (OUT / "preprocess.json").write_text(json.dumps(pre, indent=1))
    # parity on REAL dev images (eval preprocessing), PyTorch vs onnxruntime vs OpenCV DNN
    rows = get_dev(m)[:n_check]
    ds = ImageDataset(rows, m.root, m.class_to_idx, build_transforms(spec, None, train=False))
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    with torch.no_grad():
        ref = torch.cat([model(X[i:i + 8])[1] for i in range(0, len(X), 8)]).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # the dynamo exporter may specialise the batch dimension to 1 -> feed one image at a time (what a demo does)
    got = np.concatenate([sess.run(["probs"], {"input": X[i:i + 1].numpy()})[0] for i in range(len(X))])
    t0 = time.perf_counter()
    for i in range(8):
        sess.run(["probs"], {"input": X[i:i + 1].numpy()})
    ort_ms = (time.perf_counter() - t0) / 8 * 1000
    rep = {"onnx": str(onnx_path), "external_data": external,
           "onnx_mb": round(sum(p.stat().st_size for p in OUT.glob("model.onnx*")) / 2**20, 1),
           "ort_max_abs_prob_diff": float(np.abs(ref - got).max()),
           "ort_argmax_agree": float((ref.argmax(1) == got.argmax(1)).mean()), "ort_cpu_ms_per_image": round(ort_ms, 1)}
    (OUT / "export_report.json").write_text(json.dumps(rep, indent=1))  # save before the riskier OpenCV check
    # OpenCV DNN runs in a child process: its ONNX importer is native code and can crash (segfault) on large
    # transformer graphs; a crash there must not take the export down with it.
    import subprocess
    import sys

    np.save(OUT / "_parity_x.npy", X.numpy())
    np.save(OUT / "_parity_ref.npy", ref)
    code = (
        "import json,sys,numpy as np,cv2;"
        f"X=np.load({str(OUT / '_parity_x.npy')!r});ref=np.load({str(OUT / '_parity_ref.npy')!r});"
        f"net=cv2.dnn.readNetFromONNX({str(onnx_path)!r});out=[]\n"
        "for i in range(len(X)):\n net.setInput(X[i:i+1]); out.append(net.forward('probs'))\n"
        "p=np.concatenate(out);print(json.dumps({'cv2_version':cv2.__version__,"
        "'cv2_max_abs_prob_diff':float(np.abs(ref-p).max()),'cv2_argmax_agree':float((ref.argmax(1)==p.argmax(1)).mean())}))"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=1800)
    if r.returncode == 0:
        rep.update(json.loads(r.stdout.strip().splitlines()[-1]))
    else:
        rep.update(cv2_error=f"exit {r.returncode}: {(r.stderr or '')[-300:]}")
    (OUT / "export_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return rep


def cmd_test(manifest: str, confirm: bool, tag: str | None = None) -> dict:
    m = load_manifest(manifest)
    rows = get_test(m, confirm=confirm)
    head = dict(np.load(OUT / "head.npz"))
    F = extract_features(KEY, rows, m)
    y = np.array([m.class_to_idx[r["label"]] for r in rows])
    probs = np.exp(log_softmax(fused_scores(F, head) / head["temperature"]))
    probe_only = (((F - head["mu"]) / head["sd"]) @ head["coef"].T + head["intercept"]).argmax(1)
    pred, conf = probs.argmax(1), probs.max(1)
    met = compute_metrics(y, pred, m.classes)
    ood = np.array(["ood_suspect" in r.get("flags", []) for r in rows])
    thr = float(head["abstain_threshold"])
    acc_ = conf >= thr
    rep = {"manifest": manifest, "fingerprint": m.fingerprint, "n": len(rows), **met,
           "probe_only_accuracy": float((probe_only == y).mean()),
           "inscope_n": int((~ood).sum()),
           "inscope_accuracy": float((pred[~ood] == y[~ood]).mean()) if (~ood).any() else None,
           "inscope_macro_f1": (compute_metrics(y[~ood], pred[~ood], m.classes)["macro_f1"] if (~ood).any() else None),
           "abstain_threshold": thr, "coverage": float(acc_.mean()),
           "accepted_accuracy": float((pred[acc_] == y[acc_]).mean()) if acc_.any() else None,
           "ece": ece(probs, y)}
    name = tag or Path(manifest).parent.name
    out = Path("results") / f"final_test_{name}.json"
    out.write_text(json.dumps(rep, indent=1, default=float))
    np.savez(Path("results") / f"final_test_{name}_preds.npz", ids=np.array([r["id"] for r in rows]), y=y, probs=probs)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("per_class", "confusion")}, indent=1, default=float))
    return rep


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit", "export", "test"])
    ap.add_argument("--manifest", default="data/v2/manifest.json")
    ap.add_argument("--confirm", action="store_true", help="test only: acknowledge spending the sealed-test look")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--force", action="store_true", help="export: re-export even if model.onnx matches head.npz")
    a = ap.parse_args(argv)
    {"fit": lambda: cmd_fit(a.manifest), "export": lambda: cmd_export(a.manifest, force=a.force),
     "test": lambda: cmd_test(a.manifest, a.confirm, a.tag)}[a.cmd]()


if __name__ == "__main__":
    main()
