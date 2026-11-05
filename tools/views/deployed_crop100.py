"""The deployed model re-fitted and re-exported at crop fraction 1.0, next to (not instead of) exports/final.

    uv run python tools/views/deployed_crop100.py [--out exports/final_crop100]

One command from a fresh checkout: a missing crop-1.0 feature cache is extracted first (tools/views/extract_views.py,
a GPU pass), and when the output is exports/final_crop100 the app manifest is refreshed at the end
(tools/gen_manifests.py --only deployed_crop100; `--no-manifest` skips it).

Same SigLIP So400m tower, text embeddings (exports/final/head.npz W) and pipeline as compclass.final; the only change
is the eval preprocessing: resize the short side to 384 and crop nothing, instead of resize to 427 and centre-crop
384. The 2026-11-05 view control measured +1.48 ± 0.25 pt OOF macro-F1 for that change, 5/5 folds. Fit, calibration
and abstain rule are compclass.final.fit_head on the crop-1.0 feature cache (tools/views/extract_views.py
siglip_so400m_384 crop100); the export is one graph (tower + both heads + temperature, outputs logits/probs,
external data when the protobuf limit forces it), with head.npz, fit_report.json, labels.json, preprocess.json and
export_report.json (onnx sha256 + IO for tools/gen_manifests.py). exports/final, its recorded sealed-test number and
the course model C are untouched; the sealed test is not scored here.
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "src")
from compclass.data import ImageDataset, build_transforms, get_dev, load_manifest
from compclass.final import KEY, T_ZS, W_ZS, FusedClassifier, fit_head
from compclass.models import build_model

CACHE = Path("results/cache/features")
FP = "431315e87d2612cb-1fdfce4c916e"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="exports/final_crop100")
    ap.add_argument("--n-check", type=int, default=32)
    ap.add_argument("--no-manifest", action="store_true", help="do not refresh exports/final_crop100/model.json afterwards")
    a = ap.parse_args()
    import onnx, onnxruntime as ort

    m = load_manifest("data/v2/manifest.json"); dev = get_dev(m)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"{KEY}-{FP}-crop100.npy"
    if not cache.exists():
        print(f"feature cache {cache} is missing: extracting the development images at crop 1.0 (GPU)", flush=True)
        subprocess.run([sys.executable, "tools/views/extract_views.py", KEY, "crop100"], check=True)
    F = np.load(cache).astype(np.float32)
    W = np.load("exports/final/head.npz")["W"].astype(np.float32)
    head, report = fit_head(F, m, dev, W)
    report.update(view="crop_pct 1.0 (resize short side to 384, no crop)", features=str(CACHE / f"{KEY}-{FP}-crop100.npy"),
                  text_embeddings="exports/final/head.npz", baseline="exports/final/fit_report.json (crop_pct 0.9)")
    np.savez(out / "head.npz", **head)
    (out / "fit_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "abstain_table"}, indent=1), flush=True)

    trunk, spec = build_model(KEY, num_classes=0, pretrained=True, crop_pct=1.0)
    model = FusedClassifier(trunk, head).eval()
    S = spec.input_size
    p = out / "model.onnx"
    for f in (p, out / "model.onnx.data"):
        f.unlink(missing_ok=True)
    kw = dict(input_names=["input"], output_names=["logits", "probs"], opset_version=18, dynamo=True)
    try:
        torch.onnx.export(model, (torch.randn(1, 3, S, S),), p, external_data=False, **kw)
    except Exception as e:  # > 2 GB protobuf limit -> weights go to model.onnx.data
        print("single-file export failed, using external data:", repr(e)[:200], flush=True)
        torch.onnx.export(model, (torch.randn(1, 3, S, S),), p, external_data=True, **kw)
    external = (out / "model.onnx.data").exists()
    (out / "labels.json").write_text(json.dumps(m.classes, indent=1))
    pre = {**spec.to_dict(), "color_order": "RGB", "layout": "NCHW", "scale": 1 / 255, "outputs": ["logits", "probs"],
           "abstain_threshold": float(head["abstain_threshold"]), "temperature": float(head["temperature"]), "model": KEY,
           "t_zs": T_ZS, "w_zs": W_ZS, "manifest_fingerprint": m.fingerprint}
    (out / "preprocess.json").write_text(json.dumps(pre, indent=1))

    rows = dev[:a.n_check]
    ds = ImageDataset(rows, m.root, m.class_to_idx, build_transforms(spec, None, train=False))
    X = torch.stack([ds[i][0] for i in range(len(ds))])
    with torch.no_grad():
        ref = torch.cat([model(X[i:i + 4])[1] for i in range(0, len(X), 4)]).numpy()
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    got = np.concatenate([sess.run(["probs"], {"input": X[i:i + 1].numpy()})[0] for i in range(len(X))])
    t0 = time.perf_counter()
    for i in range(4):
        sess.run(["probs"], {"input": X[i:i + 1].numpy()})
    g = onnx.load(p, load_external_data=False)
    dims = lambda t: [d.dim_value if d.dim_value else d.dim_param for d in t.type.tensor_type.shape.dim]  # noqa: E731
    rep = {"onnx": str(p), "external_data": external, "onnx_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
           "onnx_mb": round(sum(f.stat().st_size for f in out.glob("model.onnx*")) / 2**20, 1),
           "ort_max_abs_prob_diff": float(np.abs(ref - got).max()), "ort_argmax_agree": float((ref.argmax(1) == got.argmax(1)).mean()),
           "ort_cpu_ms_per_image": round((time.perf_counter() - t0) / 4 * 1000, 1), "n_check": len(rows),
           "io": {"inputs": [[i.name, dims(i)] for i in g.graph.input], "outputs": [[o.name, dims(o)] for o in g.graph.output]}}
    (out / "export_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if not a.no_manifest and out.resolve() == Path("exports/final_crop100").resolve():
        r = subprocess.run([sys.executable, "tools/gen_manifests.py", "--only", "deployed_crop100"])
        if r.returncode != 0:
            print("manifest not refreshed: run `uv run python tools/gen_manifests.py --only deployed_crop100` once it is fixed")


if __name__ == "__main__":
    main()
