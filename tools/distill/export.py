"""Export a distilled student as a speed-harness candidate, and screen students on development-fold CV.

    uv run python tools/distill/export.py export --run RUN --dest CANDIDATE_DIR --commit SHA [--provider trt-fp16]
    uv run python tools/distill/export.py screen --run RUN [--run RUN ...] --manifest data/v2/manifest.json \
        --head exports/final/head.npz

``export`` writes model.onnx (input ``pixel_values`` [N,3,S,S] -> output ``embedding`` [N,1152], dynamic batch),
preprocess.json (zero_shot_fusion: the embedding lives in the SigLIP So400m pooled space), runtime.json, labels.json
and export_report.json (PyTorch vs onnxruntime parity on pool-val views).

``screen`` is UNOFFICIAL: PyTorch fp32 embeddings of the dev rows (the harness's eval preprocessing), then the
deployed head recipe fitted per development fold (standardise + logistic regression with grouped-CV C, fused with
the zero-shot text head), next to the teacher's own features run through the same code. The frozen harness's
``eval`` is the number that counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import STUDENTS, TEACHER_KEY, build_student, is_pool_val, load_pool, render_view, to_tensor  # noqa: E402


def load_student(run: Path):
    ck = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    model, pre = build_student(ck["student"], pretrained=False)
    model.load_state_dict(ck["model"])
    if STUDENTS[ck["student"]].reparameterize:
        from timm.utils.model import reparameterize_model

        model = reparameterize_model(model)
    return model.eval(), pre, ck


def cmd_export(a) -> None:
    import onnx
    import onnxruntime as ort
    from PIL import Image

    run, dest = Path(a.run), Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    model, pre, ck = load_student(run)
    S = pre["input_size"]
    torch.onnx.export(model, (torch.randn(2, 3, S, S),), str(dest / "model.onnx"), input_names=["pixel_values"],
                      output_names=["embedding"], opset_version=17, dynamo=False,
                      dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}})
    onnx.checker.check_model(str(dest / "model.onnx"))
    rows, pool = load_pool(a.pool)
    val = [r for r in rows if is_pool_val(r["id"])][:16]
    X = torch.stack([to_tensor(render_view(Image.open(Path(a.pool) / r["path"]).convert("RGB"), r["id"], 0, S),
                               pre["mean"], pre["std"]) for r in val])
    with torch.no_grad():
        ref = model(X).numpy()
    sess = ort.InferenceSession(str(dest / "model.onnx"), providers=["CPUExecutionProvider"])
    b1 = np.concatenate([sess.run(["embedding"], {"pixel_values": X[i:i + 1].numpy()})[0] for i in range(len(X))])
    b8 = np.concatenate([sess.run(["embedding"], {"pixel_values": X[i:i + 8].numpy()})[0] for i in range(0, len(X), 8)])
    cos = (ref * b1).sum(1) / np.linalg.norm(ref, axis=1) / np.linalg.norm(b1, axis=1)
    rep = {"n_images": len(X), "max_abs_diff_torch_vs_ort_b1": float(np.abs(ref - b1).max()),
           "max_abs_diff_ort_b1_vs_b8": float(np.abs(b1 - b8).max()), "min_cos_torch_vs_ort": float(cos.min()),
           "embedding_abs_max": float(np.abs(ref).max()),
           "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
           "pool_val_fidelity": ck["val"], "weights": ck["picked"]}
    assert rep["min_cos_torch_vs_ort"] > 0.9999 and rep["max_abs_diff_ort_b1_vs_b8"] < 1e-3, rep
    (dest / "preprocess.json").write_text(json.dumps({
        **pre, "resize_short_side": int(round(S / pre["crop_pct"])), "color_order": "RGB", "layout": "NCHW",
        "scale": 1 / 255, "zero_shot_fusion": True,
        "embedding_space": f"{TEACHER_KEY} pooled image embedding (1152-d), distilled",
        "built_from": f"wip/distill {a.commit}: student {ck['student']} ({STUDENTS[ck['student']].timm_name}) "
                      f"distilled from {TEACHER_KEY} on pool {pool['created_utc']} ({pool['n_kept']} images "
                      f"disjoint from data/v2 and web_holdout_v1)"}, indent=1))
    runtime = {"provider": "CUDAExecutionProvider", "options": {}}
    if a.provider == "trt-fp16":
        runtime = {"provider": "TensorrtExecutionProvider",
                   "options": {"trt_fp16_enable": True, "trt_engine_cache_enable": True,
                               "trt_engine_cache_path": "trt_cache", "trt_timing_cache_enable": True,
                               "trt_timing_cache_path": "trt_cache"}}
    (dest / "runtime.json").write_text(json.dumps(runtime, indent=1))
    (dest / "labels.json").write_text(Path(a.labels).read_text())
    (dest / "export_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def oof_scores(F: np.ndarray, m, rows, W: np.ndarray) -> dict:
    from compclass.features import fit_probe
    from compclass.final import fused_scores, head_from_search

    y = np.array([m.class_to_idx[r["label"]] for r in rows])
    fold = np.array([r["fold"] for r in rows])
    groups = np.array([r["cluster"] for r in rows])
    fused, probe, zs = np.zeros(len(rows), int), np.zeros(len(rows), int), np.zeros(len(rows), int)
    acc = []
    for k in range(5):
        tr, va = fold != k, fold == k
        s = fit_probe(F[tr], y[tr], groups[tr])
        fused[va] = fused_scores(F[va], head_from_search(s, W)).argmax(1)
        probe[va] = s.predict(F[va])
        acc.append(float((fused[va] == y[va]).mean()))
    Fn = F / np.linalg.norm(F, axis=1, keepdims=True)
    zs = (Fn @ W.T).argmax(1)
    return {"cv_acc_mean": float(np.mean(acc)), "cv_acc_sd": float(np.std(acc, ddof=1)), "fold_acc": acc,
            "oof_acc": float((fused == y).mean()), "probe_only_oof_acc": float((probe == y).mean()),
            "zeroshot_only_acc": float((zs == y).mean()), "_correct": fused == y}


def cmd_screen(a) -> None:
    from compclass.data import ImageDataset, build_transforms, get_dev, load_manifest
    from compclass.features import extract_features
    from types import SimpleNamespace

    m = load_manifest(a.manifest)
    rows = get_dev(m)
    W = np.load(a.head)["W"]
    T = extract_features(TEACHER_KEY, rows, m, cache_dir=a.cache)
    teacher = oof_scores(T, m, rows, W)
    t_correct = teacher.pop("_correct")
    out = {"teacher": teacher}
    print("teacher", json.dumps(teacher), flush=True)
    dev = torch.device("cuda")
    for run in map(Path, a.run):
        model, pre, ck = load_student(run)
        model.to(dev)
        spec = SimpleNamespace(**{k: pre[k] for k in ("input_size", "mean", "std", "interpolation", "crop_pct")})
        ds = ImageDataset(rows, m.root, m.class_to_idx, build_transforms(spec, None, train=False))
        dl = torch.utils.data.DataLoader(ds, batch_size=64, num_workers=8)
        with torch.no_grad():
            F = torch.cat([model(x.to(dev)).float().cpu() for x, _ in dl]).numpy()
        cos = (F * T).sum(1) / np.linalg.norm(F, axis=1) / np.linalg.norm(T, axis=1)
        res = oof_scores(F, m, rows, W)
        c = res.pop("_correct")
        res.update(dev_cos_to_teacher_mean=float(cos.mean()), dev_cos_to_teacher_p05=float(np.quantile(cos, 0.05)),
                   student_wrong_teacher_right=int((t_correct & ~c).sum()),
                   student_right_teacher_wrong=int((~t_correct & c).sum()), pool_val=ck["val"])
        out[run.name] = res
        print(run.name, json.dumps(res), flush=True)
        np.save(run / "dev_embeddings.npy", F)
        del model
        torch.cuda.empty_cache()
    Path(a.report).write_text(json.dumps(out, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--run", required=True)
    e.add_argument("--dest", required=True)
    e.add_argument("--pool", required=True)
    e.add_argument("--labels", required=True)
    e.add_argument("--provider", choices=["cuda", "trt-fp16"], default="cuda")
    e.add_argument("--commit", required=True, help="wip/distill commit the training and export code came from")
    s = sub.add_parser("screen")
    s.add_argument("--run", nargs="+", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--head", required=True, help="deployed head.npz; only its text embeddings W are read")
    s.add_argument("--cache", required=True)
    s.add_argument("--report", required=True)
    a = ap.parse_args(argv)
    cmd_export(a) if a.cmd == "export" else cmd_screen(a)


if __name__ == "__main__":
    main()
