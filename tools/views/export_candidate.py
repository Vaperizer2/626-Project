"""Efficiency study, step 4: export a candidate as one ONNX graph (image tower + heads), fp32 and fp16.

    uv run python tools/views/export_candidate.py --model mobileclip2_s3 --out exports/candidates/mobileclip2_s3
    uv run python tools/views/export_candidate.py --model pe_core_l14_336 --out exports/candidates/pe_core_l14_336
    uv run python tools/views/export_candidate.py --checkpoint runs/win/<run>/fold0/best.pt --out exports/candidates/<name>

With ``--model`` the graph is the deployed recipe on that backbone: frozen pretrained tower -> f; probe head
log_softmax(coef . (f - mu) / sd + b), refitted on ALL development rows from the feature cache (compclass.features
.fit_probe, grouped CV for C); zero-shot head log_softmax(T_zs cos(proj(f), W)) from tools/views/heads.py
(``proj`` is the identity for SigLIP, whose W lives in exports/final/head.npz); outputs ``logits`` = probe + w_zs
zero-shot and ``probs`` = softmax(logits). No temperature or abstain threshold: this is the timing/size artefact, not
a release (compclass.final owns calibration). With ``--checkpoint`` the fine-tuned network is exported as trained
(its own 20-way head, outputs ``logits``/``probs``). Models are exported in their inference form (timm
reparameterize_model). Static batch 1, opset 18, torch dynamo exporter like compclass.export; ``model.onnx`` fp32 and ``model_fp16.onnx`` (onnxruntime's converter, float32 IO kept), with
labels.json and preprocess.json. Checks: ORT CPU vs PyTorch on random input (max |logit| gap, argmax) and, for
``--model``, the ONNX graph vs the numpy recipe on the cached features for 16 real development images (so the
exported head is the one that scored OOF).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, "src")
from compclass.data import load_manifest, get_dev, build_transforms
from compclass.features import fit_probe
from compclass.models import build_model

CACHE, HEADS = Path("results/cache/features"), Path("results/cache/zeroshot")
FP = "431315e87d2612cb-1fdfce4c916e"
T_ZS, W_ZS = 50.0, 0.25


class FusedHead(nn.Module):
    def __init__(self, mu, sd, coef, intercept, proj_w, proj_b, W):
        super().__init__()
        for k, v in dict(mu=mu, sd=sd, coef=coef, intercept=intercept, proj_w=proj_w, proj_b=proj_b).items():
            self.register_buffer(k, torch.as_tensor(np.asarray(v, np.float32)))
        self.register_buffer("Wn", F.normalize(torch.as_tensor(np.asarray(W, np.float32)), dim=1))

    def forward(self, f):
        lp = F.log_softmax((f - self.mu) / self.sd @ self.coef.T + self.intercept, dim=1)
        e = F.normalize(f @ self.proj_w.T + self.proj_b, dim=1)
        lz = F.log_softmax(T_ZS * (e @ self.Wn.T), dim=1)
        return lp + W_ZS * lz


class Candidate(nn.Module):
    def __init__(self, tower, head):
        super().__init__()
        self.tower, self.head = tower, head

    def forward(self, x):
        logits = self.head(self.tower(x))
        return logits, F.softmax(logits, dim=1)


def recipe_model(key, m, px=None):
    dev = get_dev(m)
    y = np.array([m.class_to_idx[r["label"]] for r in dev]); cl = np.array([r["cluster"] for r in dev])
    view = ("-crop100" if key == "siglip_so400m_384" else "") + (f"@{px}" if px else "")
    Fc = np.load(CACHE / f"{key}-{FP}{view}.npy").astype(np.float32)
    search = fit_probe(Fc, y, cl)
    sc, lr = search.best_estimator_.named_steps["standardscaler"], search.best_estimator_.named_steps["logisticregression"]
    print(f"probe on all {len(dev)} dev rows: C {search.best_params_['logisticregression__C']}, CV macro-F1 {search.best_score_:.4f}")
    if key == "siglip_so400m_384":
        W = np.load("exports/final/head.npz")["W"]; pw, pb = np.eye(Fc.shape[1], dtype=np.float32), np.zeros(Fc.shape[1], np.float32)
    else:
        W = np.load(HEADS / f"{key}_W.npy"); pr = np.load(HEADS / f"{key}_proj.npz"); pw, pb = pr["weight"], pr["bias"]
    head = FusedHead(sc.mean_, sc.scale_, lr.coef_, lr.intercept_, pw, pb, W)
    kw = ({"crop_pct": 1.0} if view else {}) | ({"input_size": px} if px else {})  # the view the cache was extracted with
    tower, spec = build_model(key, 0, pretrained=True, **kw)
    model = Candidate(tower.eval(), head).eval()
    # the exported head must reproduce the numpy recipe on cached features
    with torch.no_grad():
        ref = head(torch.from_numpy(Fc[:256])).numpy()
    lp = search.best_estimator_.predict_log_proba(Fc[:256])
    e = Fc[:256] @ pw.T + pb; cos = (e / np.linalg.norm(e, axis=1, keepdims=True)) @ (W / np.linalg.norm(W, axis=1, keepdims=True)).T
    z = T_ZS * cos; lz = z - z.max(1, keepdims=True); lz = lz - np.log(np.exp(lz).sum(1, keepdims=True))
    print(f"head vs numpy recipe on 256 cached rows: max |gap| {np.abs(ref - (lp + W_ZS * lz)).max():.2e}")
    return model, spec, dev, Fc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model"); g.add_argument("--checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--px", type=int, help="input size override (needs the matching view cache, e.g. crop100@308)")
    a = ap.parse_args()
    import onnxruntime as ort
    m = load_manifest("data/v2/manifest.json")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.model:
        model, spec, dev, Fc = recipe_model(a.model, m, a.px)
        key, extra = a.model, {"recipe": "probe (all dev rows) + zero-shot, T_zs 50, w_zs 0.25, uncalibrated"}
    else:
        from compclass.export import load_checkpoint_model
        net, ck = load_checkpoint_model(a.checkpoint)
        from compclass.models.registry import PreprocessSpec
        spec = PreprocessSpec(**{k: ck["preprocess"][k] for k in ("input_size", "mean", "std", "interpolation", "crop_pct")})
        model = Candidate(net, nn.Identity()).eval()
        key, extra = ck["config"]["model"]["key"], {"checkpoint": a.checkpoint, "config_hash": ck["config_hash"], "fold": ck["fold"],
                                                    "recipe": f"fine-tuned {ck['config']['name']}, own linear head"}
        dev = Fc = None
    from timm.utils.model import reparameterize_model
    model = reparameterize_model(model).eval()  # inference form (FastViT branches fold into single convs; no-op for ViTs)
    S = spec.input_size
    x = torch.randn(1, 3, S, S)
    p32 = out / "model.onnx"
    torch.onnx.export(model, (x,), p32, input_names=["input"], output_names=["logits", "probs"], opset_version=18,
                      dynamo=True, external_data=False)
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    m16 = convert_float_to_float16(onnx.load(p32), keep_io_types=True)
    p16 = out / "model_fp16.onnx"; onnx.save(m16, p16)
    (out / "labels.json").write_text(json.dumps(m.classes, indent=1))
    pre = {**spec.to_dict(), "color_order": "RGB", "layout": "NCHW", "scale": 1 / 255, "outputs": ["logits", "probs"],
           "model": key, "manifest_fingerprint": m.fingerprint, **extra}
    (out / "preprocess.json").write_text(json.dumps(pre, indent=1))
    rep = {"model": key, "px": S, "fp32_mb": p32.stat().st_size / 2**20, "fp16_mb": p16.stat().st_size / 2**20}
    with torch.no_grad():
        ref = model(x)[0].numpy()
    for name, p in (("fp32", p32), ("fp16", p16)):
        s = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        lg = s.run(["logits"], {"input": x.numpy()})[0]
        rep[f"{name}_max_gap_random"] = float(np.abs(lg - ref).max()); rep[f"{name}_argmax_agree_random"] = bool(lg.argmax() == ref.argmax())
    if dev is not None:  # 16 real images through the graph vs the numpy recipe on their cached features
        from PIL import Image
        tf = build_transforms(spec, None, train=False)
        s = ort.InferenceSession(str(p32), providers=["CPUExecutionProvider"])
        head = model.head
        agree, gaps = 0, []
        for i in range(0, 256, 16):
            with Image.open(m.root / dev[i]["path"]) as im:
                xi = tf(im.convert("RGB")).unsqueeze(0).numpy()
            lg = s.run(["logits"], {"input": xi})[0]
            with torch.no_grad():
                rf = head(torch.from_numpy(Fc[i:i + 1])).numpy()
            agree += int(lg.argmax() == rf.argmax()); gaps.append(float(np.abs(lg - rf).max()))
        rep["real16_argmax_agree"] = agree; rep["real16_max_gap"] = max(gaps)
    print(json.dumps(rep, indent=1))
    (out / "export_report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
