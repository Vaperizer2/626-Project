"""Package a small-backbone candidate the way compclass.final packages SigLIP: fit, calibrate, export, check.

    uv run python tools/views/package_candidate.py --model mobileclip2_s3 --out exports/mobileclip2_s3

One command from a fresh checkout: whatever is missing is built first (the frozen features of the development rows
with compclass.features.extract_features, a GPU pass; the text embeddings and projection with tools/views/heads.py,
which downloads the weights from Hugging Face), and when ``--out`` is the model's folder under exports/ the app
manifest is refreshed at the end (tools/gen_manifests.py --only <model>), so its checksums match the graph just built.
The graph built on another machine is not byte-identical to the one in the release; `--no-manifest` skips the refresh.

Same pipeline as ``compclass.final fit`` + ``export`` (whose helpers this reuses), generalised to a backbone whose
timm features sit BEFORE its projection into the text space (tools/views/heads.py): out-of-fold scores of the exact
exported pipeline on the five development folds (probe + zero-shot head, T_zs 50, w_zs 0.25), temperature by NLL
on those scores, abstain threshold = smallest confidence at which accepted OOF predictions are >= 90 % correct,
probe refitted on all development rows, then one ONNX graph (tower in inference form + both heads + temperature;
outputs ``logits`` and ``probs``), fp32 ``model.onnx`` and an fp16 copy, with ``head.npz``, ``fit_report.json``,
``labels.json``, ``preprocess.json`` and ``export_report.json`` (onnx sha256 and graph IO, which tools/gen_manifests.py
verifies). Parity on 32 real development images: PyTorch vs onnxruntime CPU, max |prob| gap and argmax agreement.
Selection and calibration use development rows only; nothing here touches the sealed test or Wikimedia rows.
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

sys.path.insert(0, "src")
from compclass.data import ImageDataset, build_transforms, get_dev, get_fold, load_manifest
from compclass.evaluate import compute_metrics
from compclass.features import fit_probe
from compclass.final import T_ZS, W_ZS, ece, fit_temperature, head_from_search, log_softmax
from compclass.models import build_model

CACHE, HEADS = Path("results/cache/features"), Path("results/cache/zeroshot")
FP = "431315e87d2612cb-1fdfce4c916e"


def ensure_inputs(key, m, dev) -> None:
    """Build what a fresh checkout lacks: the cached development features and the two zero-shot matrices."""
    cache = CACHE / f"{key}-{FP}.npy"
    if not cache.exists():
        from compclass.features import extract_features

        print(f"feature cache {cache} is missing: extracting {len(dev)} development images (GPU recommended)", flush=True)
        extract_features(key, dev, m, num_workers=0 if os.name == "nt" else 4)  # Windows: no loader workers from a script
        if not cache.exists():
            raise SystemExit(f"extract_features did not write {cache}: the manifest differs from fingerprint {FP}")
    if not (HEADS / f"{key}_W.npy").exists() or not (HEADS / f"{key}_proj.npz").exists():
        print(f"zero-shot matrices for {key} are missing: building them (tools/views/heads.py)", flush=True)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import heads

        heads.write_heads(key)


def fused_scores(F, head):
    """Un-tempered fused log-scores from raw (pre-projection) features; numpy mirror of FusedClassifier."""
    lp = log_softmax(((F - head["mu"]) / head["sd"]) @ head["coef"].T + head["intercept"])
    e = F @ head["proj_w"].T + head["proj_b"]
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    return lp + head["w_zs"] * log_softmax(head["t_zs"] * e @ head["W"].T)


class FusedClassifier(nn.Module):
    def __init__(self, trunk, head):
        super().__init__()
        self.trunk = trunk
        for k in ("mu", "sd", "coef", "intercept", "W", "proj_w", "proj_b"):
            self.register_buffer(k, torch.as_tensor(np.asarray(head[k]), dtype=torch.float32))
        self.t_zs, self.w_zs, self.temperature = float(head["t_zs"]), float(head["w_zs"]), float(head["temperature"])

    def forward(self, x):
        f = self.trunk(x)
        lp = torch.log_softmax(((f - self.mu) / self.sd) @ self.coef.T + self.intercept, dim=1)
        e = f @ self.proj_w.T + self.proj_b
        zs = torch.log_softmax(self.t_zs * (e / e.norm(dim=1, keepdim=True)) @ self.W.T, dim=1)
        logits = (lp + self.w_zs * zs) / self.temperature
        return logits, torch.softmax(logits, dim=1)


def fit(key, m, dev):
    y = np.array([m.class_to_idx[r["label"]] for r in dev])
    pos = {r["id"]: i for i, r in enumerate(dev)}
    F = np.load(CACHE / f"{key}-{FP}.npy").astype(np.float32)
    W = np.load(HEADS / f"{key}_W.npy").astype(np.float32)
    pr = np.load(HEADS / f"{key}_proj.npz")
    extra = {"W": W, "proj_w": pr["weight"].astype(np.float32), "proj_b": pr["bias"].astype(np.float32)}
    oof = np.zeros((len(dev), len(m.classes)), np.float32); oof_probe = np.zeros(len(dev), int)
    for k in range(5):
        tr, va = get_fold(m, k)
        itr, iva = [pos[r["id"]] for r in tr], [pos[r["id"]] for r in va]
        s = fit_probe(F[itr], y[itr], np.array([r["cluster"] for r in tr]))
        oof[iva] = fused_scores(F[iva], {**head_from_search(s, W), **extra}); oof_probe[iva] = s.predict(F[iva])
    temp = fit_temperature(oof, y)
    probs = np.exp(log_softmax(oof / temp)); pred, conf = probs.argmax(1), probs.max(1)
    ood = np.array(["ood_suspect" in r["flags"] for r in dev])
    grid = np.round(np.arange(0.0, 0.96, 0.05), 2)
    table = [{"threshold": float(t), "coverage": float((conf >= t).mean()),
              "accepted_acc": float((pred[conf >= t] == y[conf >= t]).mean()) if (conf >= t).any() else None} for t in grid]
    thr = next((r["threshold"] for r in table if r["accepted_acc"] and r["accepted_acc"] >= 0.90), 0.5)
    search = fit_probe(F, y, np.array([r["cluster"] for r in dev]))
    head = {**head_from_search(search, W), **extra, "temperature": np.float32(temp), "abstain_threshold": np.float32(thr)}
    met = compute_metrics(y, pred, m.classes)
    report = {"model": key, "oof_accuracy": met["accuracy"], "oof_macro_f1": met["macro_f1"],
              "oof_inscope_accuracy": float((pred[~ood] == y[~ood]).mean()),
              "oof_probe_only_accuracy": float((oof_probe == y).mean()),
              "oof_probe_only_macro_f1": compute_metrics(y, oof_probe, m.classes)["macro_f1"], "temperature": temp,
              "ece_before": ece(np.exp(log_softmax(oof)), y), "ece_after": ece(probs, y), "abstain_threshold": thr,
              "abstain_table": table, "final_C": head["C"], "t_zs": T_ZS, "w_zs": W_ZS, "manifest": m.fingerprint,
              "n_dev": len(dev), "features": str(CACHE / f"{key}-{FP}.npy"), "text_embeddings": str(HEADS / f"{key}_W.npy"),
              "projection": str(HEADS / f"{key}_proj.npz")}
    return head, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n-check", type=int, default=32)
    ap.add_argument("--no-manifest", action="store_true", help="do not refresh exports/<model>/model.json afterwards")
    a = ap.parse_args()
    import onnx, onnxruntime as ort
    from onnxruntime.transformers.float16 import convert_float_to_float16
    from timm.utils.model import reparameterize_model

    m = load_manifest("data/v2/manifest.json"); dev = get_dev(m)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ensure_inputs(a.model, m, dev)
    head, report = fit(a.model, m, dev)
    np.savez(out / "head.npz", **head)
    (out / "fit_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "abstain_table"}, indent=1), flush=True)

    trunk, spec = build_model(a.model, 0, pretrained=True)
    model = reparameterize_model(FusedClassifier(trunk.eval(), head)).eval()
    S = spec.input_size
    p32 = out / "model.onnx"
    torch.onnx.export(model, (torch.randn(1, 3, S, S),), p32, input_names=["input"], output_names=["logits", "probs"],
                      opset_version=18, dynamo=True, external_data=False)
    onnx.save(convert_float_to_float16(onnx.load(p32), keep_io_types=True), out / "model_fp16.onnx")
    (out / "labels.json").write_text(json.dumps(m.classes, indent=1))
    pre = {**spec.to_dict(), "color_order": "RGB", "layout": "NCHW", "scale": 1 / 255, "outputs": ["logits", "probs"],
           "abstain_threshold": float(head["abstain_threshold"]), "temperature": float(head["temperature"]),
           "model": a.model, "t_zs": T_ZS, "w_zs": W_ZS, "manifest_fingerprint": m.fingerprint,
           "note": "pre-projection timm features; the checkpoint's projection into the text space is in the graph (head.npz proj_w/proj_b)"}
    (out / "preprocess.json").write_text(json.dumps(pre, indent=1))

    rows = dev[:a.n_check]
    ds = ImageDataset(rows, m.root, m.class_to_idx, build_transforms(spec, None, train=False))
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    with torch.no_grad():
        ref = torch.cat([model(X[i:i + 8])[1] for i in range(0, len(X), 8)]).numpy()
    rep = {}
    for name, p in (("fp32", p32), ("fp16", out / "model_fp16.onnx")):
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        got = np.concatenate([sess.run(["probs"], {"input": X[i:i + 1].numpy()})[0] for i in range(len(X))])
        t0 = time.perf_counter()
        for i in range(8):
            sess.run(["probs"], {"input": X[i:i + 1].numpy()})
        rep[name] = {"max_abs_prob_diff": float(np.abs(ref - got).max()), "argmax_agree": float((ref.argmax(1) == got.argmax(1)).mean()),
                     "ort_cpu_ms_per_image": round((time.perf_counter() - t0) / 8 * 1000, 1), "mb": round(p.stat().st_size / 2**20, 1)}
    g = onnx.load(p32, load_external_data=False)
    dims = lambda t: [d.dim_value if d.dim_value else d.dim_param for d in t.type.tensor_type.shape.dim]  # noqa: E731
    rep.update(onnx=str(p32), onnx_sha256=hashlib.sha256(p32.read_bytes()).hexdigest(), external_data=False,
               onnx_mb=rep["fp32"]["mb"], ort_max_abs_prob_diff=rep["fp32"]["max_abs_prob_diff"], ort_argmax_agree=rep["fp32"]["argmax_agree"],
               io={"inputs": [[i.name, dims(i)] for i in g.graph.input], "outputs": [[o.name, dims(o)] for o in g.graph.output]},
               inference_form="timm reparameterize_model", n_check=len(rows))
    (out / "export_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if not a.no_manifest and out.resolve() == (Path("exports") / a.model).resolve():
        # the app's manifest records the graph's size and sha256: refresh it for the graph just written
        r = subprocess.run([sys.executable, "tools/gen_manifests.py", "--only", a.model])
        if r.returncode != 0:
            print(f"manifest not refreshed: run `uv run python tools/gen_manifests.py --only {a.model}` once it is fixed")


if __name__ == "__main__":
    main()
