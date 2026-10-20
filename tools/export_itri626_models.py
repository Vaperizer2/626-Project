"""Export the ITRI626 report's three models so the desktop app (apps/gpui-demo) can run each of them.

    uv run --frozen python tools/export_itri626_models.py all      # = ab + c (run on the GPU PC, CPU only)
    uv run --frozen python tools/export_itri626_models.py ab       # models A and B only
    uv run --frozen python tools/export_itri626_models.py c [--no-feature-check]

Nothing is retrained. Inputs (all on the PC):
    runs/itri626/scratch_cnn-4e4c6e8f6cf3/fold0/best.pt        (A)
    runs/itri626/convnext_tiny-bb9c72a49446/fold0/best.pt      (B)
    runs/itri626/siglip_so400m_linear/fold0/head.pt            (C: nn.Linear(1152, 20) + train-split mu / sd)
    exports/final/model.onnx (+ model.onnx.data)               (deployed graph; contains the SigLIP backbone)

Outputs:
    exports/itri626/{A_scratch_cnn,B_convnext_tiny}/{model.onnx, preprocess.json, labels.json, export_report.json}
        model.onnx: one file (no external data), input ``input`` [1,3,S,S] float32 NCHW RGB, outputs ``logits``
        and ``probs`` = softmax(logits) [1,20] -- the app's contract (apps/gpui-demo/src/model.rs).
    exports/final/model_multi.onnx + model_multi.json
        The deployed graph plus two outputs ``logits_c`` / ``probs_c`` = report model C, computed from the same pooled
        1152-d SigLIP feature (graph tensor ``select``). It references the *unchanged* exports/final/model.onnx.data;
        the standardisation (f - mu) / sd is folded into the weights:  logits_c = f @ (W / sd).T + (b - (W / sd) @ mu).
        Built by ONNX graph surgery (onnx + numpy only, external data never loaded or rewritten).
    exports/itri626/export_summary.json

Evaluation re-scores the sealed test split (418 rows) with models whose test numbers are already reported in
results/itri626/metrics.json; it spends no new test look (no model choice depends on it). Images are decoded and
preprocessed with compclass.demo.open_image + compclass.demo.preprocess -- the Python reference that the Rust app's
preprocess.rs ports bit-exactly -- driven by the exported preprocess.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MANIFEST = "data/v2/manifest.json"
FINAL = Path("exports/final")
OUT = Path("exports/itri626")
RUNS = Path("runs/itri626")
CNN_MODELS = {  # report name -> (run dir, export folder)
    "A_scratch_cnn": ("scratch_cnn-4e4c6e8f6cf3/fold0", "A_scratch_cnn"),
    "B_convnext_tiny": ("convnext_tiny-bb9c72a49446/fold0", "B_convnext_tiny"),
}
C_NAME, C_HEAD = "C_siglip_so400m_linear", "siglip_so400m_linear/fold0/head.pt"
SIGLIP = "siglip_so400m_384"
FEATURE_TENSOR = "select"  # pooled SigLIP feature in exports/final/model.onnx (input of the probe's Sub(mu))
DEPLOYED_DATA_SHA_PREFIX = "a6d11a2d"
DEPLOYED_GRAPH_SHA = "b3dce111bb9d19cd854d2f264429a12dca4a6843a229446c13faf8fc51059d6c"  # exports/final/model.onnx
ABSTAIN = 0.5  # same UNSURE rule as the deployed model; A/B/C probabilities are NOT temperature-calibrated
OPSET = 18


def sha256(p: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def jdump(obj, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def reported(name: str) -> dict:
    m = json.loads((ROOT / "results/itri626/metrics.json").read_text())["models"][name]["test"]
    return {"accuracy": float(m["accuracy"]), "n": int(m.get("n", 418))}


def acc_check(pred: np.ndarray, y: np.ndarray, name: str) -> dict:
    rep = reported(name)
    n_ok = int((pred == y).sum())
    n_rep = int(round(rep["accuracy"] * len(y)))
    return {"onnx_test_accuracy": n_ok / len(y), "onnx_test_correct": n_ok, "n_test": int(len(y)),
            "reported_accuracy": rep["accuracy"], "reported_correct": n_rep, "delta_images": n_ok - n_rep,
            "reproduces_within_1_image": abs(n_ok - n_rep) <= 1}


def ort_session(path: Path):
    import onnxruntime as ort

    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])  # default = ORT_ENABLE_ALL (= Level3)


def ort_run(sess, X: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """One image at a time (batch 1 static), like the app."""
    outs = [sess.run(names, {"input": X[i:i + 1]}) for i in range(len(X))]
    return {n: np.concatenate([o[k] for o in outs]) for k, n in enumerate(names)}


def ort_ms(sess, x: np.ndarray, n: int = 20) -> float:
    for _ in range(3):
        sess.run(["probs"], {"input": x})
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(["probs"], {"input": x})
        t.append(time.perf_counter() - t0)
    return round(float(np.median(t)) * 1000, 2)


def test_rows():
    from compclass.data import get_test, load_manifest

    m = load_manifest(MANIFEST)
    rows = get_test(m, confirm=True)  # re-scoring already-reported models; see module docstring
    y = np.array([m.class_to_idx[r["label"]] for r in rows])
    return m, rows, y


def app_tensors(m, rows, pre: dict) -> np.ndarray:
    """What the Rust app feeds the model: demo.open_image (EXIF transpose, alpha on white) + demo.preprocess."""
    from compclass import demo

    return np.concatenate([demo.preprocess(demo.open_image(m.root / r["path"]), pre) for r in rows])


def image_quirks(m, rows) -> dict:
    """Test images where the app's decoder path (EXIF orientation, alpha flatten) differs from ImageDataset."""
    from PIL import Image

    exif, alpha = [], []
    for r in rows:
        with Image.open(m.root / r["path"]) as im:
            o = im.getexif().get(0x0112)
            if o not in (None, 1):
                exif.append(r["id"])
            if im.mode in ("RGBA", "LA", "P"):
                alpha.append(r["id"])
    return {"n_exif_rotated": len(exif), "n_alpha_or_palette": len(alpha), "exif_ids": exif[:20],
            "alpha_ids": alpha[:20]}


# ---------------------------------------------------------------------------------------------------- A, B
def export_cnn(name: str, quirks: dict | None = None) -> dict:
    import onnx
    import torch
    from torch import nn

    from compclass.data import ImageDataset, build_transforms
    from compclass.export import load_checkpoint_model
    from compclass.models.registry import PreprocessSpec

    run_dir, folder = CNN_MODELS[name]
    ck_path = RUNS / run_dir / "best.pt"
    out = OUT / folder
    out.mkdir(parents=True, exist_ok=True)
    model, ck = load_checkpoint_model(ck_path)
    classes = list(ck["classes"])
    final_labels = json.loads((FINAL / "labels.json").read_text())
    m, rows, y = test_rows()
    assert classes == final_labels == list(m.classes), f"{name}: class order differs from exports/final/labels.json"
    p = dict(ck["preprocess"])
    spec = PreprocessSpec(p["input_size"], tuple(p["mean"]), tuple(p["std"]), p["interpolation"], p["crop_pct"])
    assert spec.to_dict()["resize_short_side"] == p["resize_short_side"]
    size = spec.input_size

    class WithProbs(nn.Module):
        def __init__(self, net: nn.Module):
            super().__init__()
            self.net = net

        def forward(self, x):
            z = self.net(x)
            return z, torch.softmax(z, dim=1)

    wrapped = WithProbs(model).eval()
    onnx_path = out / "model.onnx"
    torch.onnx.export(wrapped, (torch.randn(1, 3, size, size),), onnx_path, input_names=["input"],
                      output_names=["logits", "probs"], opset_version=OPSET, dynamo=True, external_data=False)
    g = onnx.load(str(onnx_path))
    onnx.checker.check_model(g)
    ext = [t.name for t in g.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    assert not ext and not (out / "model.onnx.data").exists(), f"{name}: export used external data"
    io = {"inputs": [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in g.graph.input],
          "outputs": [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in g.graph.output],
          "opset": OPSET, "ir_version": g.ir_version}
    assert io["inputs"] == [("input", [1, 3, size, size])], io
    assert [o[0] for o in io["outputs"]] == ["logits", "probs"], io

    (out / "labels.json").write_text(json.dumps(classes, indent=1))
    pre = {"input_size": size, "mean": list(spec.mean), "std": list(spec.std), "interpolation": spec.interpolation,
           "crop_pct": spec.crop_pct, "resize_short_side": p["resize_short_side"], "color_order": "RGB",
           "layout": "NCHW", "scale": 1 / 255, "outputs": ["logits", "probs"], "abstain_threshold": ABSTAIN,
           "temperature": 1.0, "model": ck["config"]["model"]["key"], "report_model": name,
           "config_hash": ck["config_hash"], "fold": ck["fold"], "manifest_fingerprint": ck["manifest_fingerprint"],
           "checkpoint": str(ck_path), "checkpoint_sha256": sha256(ck_path),
           "note": "probs = softmax(logits), not temperature-calibrated; abstain_threshold 0.5 copied from the "
                   "deployed model's rule, not tuned for this model"}
    (out / "preprocess.json").write_text(json.dumps(pre, indent=1))

    # ---- parity: PyTorch fp32 vs onnxruntime on the SAME tensors (training-side eval transform), all test rows
    ds = ImageDataset(rows, m.root, m.class_to_idx, build_transforms(spec, None, train=False))
    Xref = torch.stack([ds[i][0] for i in range(len(ds))]).numpy()
    with torch.no_grad():
        p_torch = np.concatenate([wrapped(torch.from_numpy(Xref[i:i + 32]))[1].numpy() for i in range(0, len(Xref), 32)])
    sess = ort_session(onnx_path)
    p_ort_ref = ort_run(sess, Xref, ["probs"])["probs"]
    # ---- what the app computes: demo pipeline from the exported preprocess.json
    Xapp = app_tensors(m, rows, pre)
    x_gap = float(np.abs(Xapp - Xref).max())
    p_app = p_ort_ref if x_gap == 0 else ort_run(sess, Xapp, ["probs"])["probs"]
    stored = np.load(ROOT / "results/itri626/preds" / f"{name}_test.npz")
    assert list(stored["ids"]) == [r["id"] for r in rows]
    rep = {"model": name, "onnx": str(onnx_path), "onnx_bytes": onnx_path.stat().st_size,
           "onnx_sha256": sha256(onnx_path), "external_data": False, "io": io,
           "labels_match_exports_final": True,
           "parity": {"n_images": len(rows), "inputs": "test images, compclass.data.build_transforms(train=False)",
                      "torch": f"fp32 CPU {torch.__version__}", "max_abs_prob_diff": float(np.abs(p_torch - p_ort_ref).max()),
                      "argmax_agreement": float((p_torch.argmax(1) == p_ort_ref.argmax(1)).mean())},
           "app_pipeline": {"preprocess": "compclass.demo.open_image + demo.preprocess(preprocess.json)",
                            "max_abs_input_diff_vs_build_transforms": x_gap,
                            "unsure_fraction_at_0.5": float((p_app.max(1) < ABSTAIN).mean())},
           "test": acc_check(p_app.argmax(1), y, name),
           "vs_reported_preds": {"note": "results/itri626/preds: PyTorch fp16 autocast on GPU at training time",
                                 "argmax_agreement": float((stored["probs"].argmax(1) == p_app.argmax(1)).mean()),
                                 "n_argmax_differ": int((stored["probs"].argmax(1) != p_app.argmax(1)).sum()),
                                 "max_abs_prob_diff": float(np.abs(stored["probs"] - p_app).max())},
           "ort_cpu_ms_per_image": ort_ms(sess, Xapp[:1]), "image_quirks": quirks}
    if spec.interpolation != "bicubic":  # what an app that only implements bicubic would get
        pre_b = {**pre, "interpolation": "bicubic"}
        p_b = ort_run(sess, app_tensors(m, rows, pre_b), ["probs"])["probs"]
        rep["if_resized_bicubic_instead"] = {**acc_check(p_b.argmax(1), y, name),
                                             "argmax_agreement_with_correct_pipeline": float((p_b.argmax(1) == p_app.argmax(1)).mean())}
    jdump(rep, out / "export_report.json")
    print(json.dumps({k: rep[k] for k in ("model", "onnx_bytes", "parity", "test", "vs_reported_preds")}, default=str),
          flush=True)
    return rep


# ---------------------------------------------------------------------------------------------------- C
def fold_head(W: np.ndarray, b: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """W ((f - mu) / sd) + b  ==  f @ Wf + bf  with Wf = (W / sd).T [D, C], bf = b - (W / sd) @ mu (float64 fold)."""
    Ws = W.astype(np.float64) / sd.astype(np.float64)[None, :]
    return Ws.T.astype(np.float32), (b.astype(np.float64) - Ws @ mu.astype(np.float64)).astype(np.float32)


def find_feature(g) -> str:
    """The pooled feature = first input of Sub(., mu) feeding Div(., sd); it must also feed the zero-shot ReduceL2."""
    init = {t.name: t for t in g.initializer}
    subs = [n for n in g.node if n.op_type == "Sub" and len(n.input) == 2 and n.input[1] == "mu"]
    assert len(subs) == 1 and list(init["mu"].dims) == [1152], "probe standardisation node not found"
    f = subs[0].input[0]
    users = {n.op_type for n in g.node if f in n.input}
    assert f == FEATURE_TENSOR and "ReduceL2" in users, (f, users)
    return f


def add_c_head(src: Path, dst: Path, W, b, mu, sd, extra_outputs: tuple[str, ...] = ()) -> dict:
    """Graph surgery: load ``src`` WITHOUT external data, append the folded C head, write the proto bytes to ``dst``.

    External initializers keep their (location, offset, length) entries untouched, so ``dst`` reads the same
    model.onnx.data as ``src`` when it sits in the same folder. Serialising the proto directly (not onnx.save) makes
    it impossible for the onnx library to (re)write any external data file.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    mdl = onnx.load(str(src), load_external_data=False)
    g = mdl.graph
    ext_before = [(t.name, [(e.key, e.value) for e in t.external_data]) for t in g.initializer
                  if t.data_location == TensorProto.EXTERNAL]
    f = find_feature(g)
    names = {t.name for t in g.initializer} | {o for n in g.node for o in n.output} | {i.name for i in g.input}
    new = ["c_weight", "c_bias", "c_matmul", "logits_c", "probs_c"]
    assert not names & set(new), names & set(new)
    Wf, bf = fold_head(W, b, mu, sd)
    g.initializer.extend([numpy_helper.from_array(Wf, "c_weight"), numpy_helper.from_array(bf, "c_bias")])
    g.node.extend([
        helper.make_node("MatMul", [f, "c_weight"], ["c_matmul"], name="c_head_matmul"),
        helper.make_node("Add", ["c_matmul", "c_bias"], ["logits_c"], name="c_head_add"),
        helper.make_node("Softmax", ["logits_c"], ["probs_c"], name="c_head_softmax", axis=1),
    ])
    C = Wf.shape[1]
    g.output.extend([helper.make_tensor_value_info("logits_c", TensorProto.FLOAT, [1, C]),
                     helper.make_tensor_value_info("probs_c", TensorProto.FLOAT, [1, C])])
    for e in extra_outputs:  # verification only (e.g. the raw feature)
        g.output.append(helper.make_tensor_value_info(e, TensorProto.FLOAT, [1, W.shape[1]]))
    mdl.metadata_props.append(onnx.StringStringEntryProto(
        key="model_multi", value="outputs logits/probs = deployed fused model (unchanged); logits_c/probs_c = ITRI626 "
                                 "model C (SigLIP So400m frozen + linear head, train split)"))
    ext_after = [(t.name, [(e.key, e.value) for e in t.external_data]) for t in g.initializer
                 if t.data_location == TensorProto.EXTERNAL]
    assert ext_before == ext_after
    cwd = os.getcwd()
    try:  # the checker resolves external-data locations against the cwd; it only checks that the file exists
        os.chdir(Path(src).resolve().parent)
        onnx.checker.check_model(mdl)
    finally:
        os.chdir(cwd)
    dst.write_bytes(mdl.SerializeToString())
    return {"feature_tensor": f, "n_external_initializers": len(ext_after),
            "external_locations": sorted({v for _, kv in ext_after for k, v in kv if k == "location"}),
            "added_initializers": {"c_weight": list(Wf.shape), "c_bias": list(bf.shape)},
            "added_nodes": ["MatMul(select, c_weight)", "Add(., c_bias) -> logits_c", "Softmax(axis=1) -> probs_c"]}


def deployed_initializer(name: str) -> np.ndarray:
    """Read one external initializer of exports/final/model.onnx (by offset/length; nothing else is loaded)."""
    import onnx
    from onnx import numpy_helper
    from onnx.external_data_helper import load_external_data_for_tensor

    g = onnx.load(str(FINAL / "model.onnx"), load_external_data=False).graph
    t = onnx.TensorProto()
    t.CopyFrom(next(x for x in g.initializer if x.name == name))
    load_external_data_for_tensor(t, str(FINAL))
    return numpy_helper.to_array(t)


def export_c(feature_check: bool = True) -> dict:
    import torch

    from compclass.data import get_dev, get_fold
    from compclass.linear_head import standardiser

    src, dst, data = FINAL / "model.onnx", FINAL / "model_multi.onnx", FINAL / "model.onnx.data"
    sha = {"model.onnx": sha256(src), "model.onnx.data": sha256(data), "head.npz": sha256(FINAL / "head.npz"),
           "preprocess.json": sha256(FINAL / "preprocess.json"), "labels.json": sha256(FINAL / "labels.json")}
    assert sha["model.onnx.data"].startswith(DEPLOYED_DATA_SHA_PREFIX), sha
    assert sha["model.onnx"] == DEPLOYED_GRAPH_SHA, sha
    head_path = RUNS / C_HEAD
    st = torch.load(head_path, map_location="cpu", weights_only=False)
    W, b, mu, sd = (st[k].float().numpy() for k in ("weight", "bias", "mu", "sd"))
    assert W.shape == (20, 1152) and b.shape == (20,) and mu.shape == sd.shape == (1152,)
    surgery = add_c_head(src, dst, W, b, mu, sd)

    m, rows, y = test_rows()
    assert json.loads((FINAL / "labels.json").read_text()) == list(m.classes)
    pre = json.loads((FINAL / "preprocess.json").read_text())
    run_pre = json.loads((ROOT / "results/itri626/runs" / f"{C_NAME}.json").read_text())["preprocess"]
    assert all(run_pre[k] == pre[k] for k in ("input_size", "resize_short_side", "interpolation", "mean", "std")), \
        "C was trained on features with a different eval transform than the deployed graph's"
    # ---- where mu / sd come from: head.pt; check they are the train-split (folds 1-4) statistics of the cache
    fp = m.fingerprint
    cache = ROOT / "results/cache/features"
    ids_hash = lambda rs: hashlib.sha256("".join(r["id"] for r in rs).encode()).hexdigest()[:12]  # noqa: E731
    dev = get_dev(m)
    Fdev_p, Fte_p = cache / f"{SIGLIP}-{fp}-{ids_hash(dev)}.npy", cache / f"{SIGLIP}-{fp}-{ids_hash(rows)}.npy"
    checks: dict = {}
    if Fdev_p.exists():
        Fdev = np.load(Fdev_p)
        pos = {r["id"]: i for i, r in enumerate(dev)}
        tr, _ = get_fold(m, 0)
        mu2, sd2 = standardiser(Fdev[[pos[r["id"]] for r in tr]])
        checks["mu_sd_equal_train_split_stats"] = bool(np.array_equal(mu2, mu) and np.array_equal(sd2, sd))
        checks["mu_sd_max_abs_diff"] = float(max(np.abs(mu2 - mu).max(), np.abs(sd2 - sd).max()))
    Fte = np.load(Fte_p) if Fte_p.exists() else None
    if Fte is not None:
        z_ref = ((Fte - mu) / sd) @ W.T + b
        checks["head_on_cached_features_accuracy"] = float((z_ref.argmax(1) == y).mean())
        Wf, bf = fold_head(W, b, mu, sd)
        z_fold = Fte @ Wf + bf
        checks["fold_vs_unfolded_max_abs_logit_diff_on_cache"] = float(np.abs(z_fold - z_ref).max())
    # the deployed graph's 'mu' initializer is the DEPLOYED probe's mean (all dev rows) -> confirms 'select' is the
    # raw feature space head.npz / compclass.features use
    mu_graph = deployed_initializer("mu")
    head_npz = np.load(FINAL / "head.npz")
    checks["graph_mu_equals_head_npz_mu"] = bool(np.array_equal(mu_graph, head_npz["mu"]))

    # ---- run model_multi vs model.onnx on all test images (app pipeline)
    X = app_tensors(m, rows, pre)
    t0 = time.perf_counter()
    s_multi = ort_session(dst)
    got = ort_run(s_multi, X, ["logits", "probs", "logits_c", "probs_c"])
    ms_multi = ort_ms(s_multi, X[:1], n=5)
    del s_multi
    s_ref = ort_session(src)
    ref = ort_run(s_ref, X, ["logits", "probs"])
    ms_ref = ort_ms(s_ref, X[:1], n=5)
    del s_ref
    run_s = time.perf_counter() - t0
    stored = np.load(ROOT / "results/itri626/preds" / f"{C_NAME}_test.npz")
    assert list(stored["ids"]) == [r["id"] for r in rows]
    ver = {"n_images": len(rows), "inputs": "test images via compclass.demo.open_image + demo.preprocess(exports/final/preprocess.json)",
           "fused_probs_bit_identical": bool(np.array_equal(got["probs"], ref["probs"])),
           "fused_logits_bit_identical": bool(np.array_equal(got["logits"], ref["logits"])),
           "fused_max_abs_prob_diff": float(np.abs(got["probs"] - ref["probs"]).max()),
           "probs_c_sum_max_dev": float(np.abs(got["probs_c"].sum(1) - 1).max()),
           "probs_c_equals_softmax_logits_c_max_diff": float(np.abs(softmax(got["logits_c"]) - got["probs_c"]).max()),
           "test_C": acc_check(got["probs_c"].argmax(1), y, C_NAME),
           "test_fused_accuracy": float((got["probs"].argmax(1) == y).mean()),
           "vs_reported_C_preds": {"note": "results/itri626/preds: linear head on fp16-autocast GPU features (PyTorch)",
                                   "argmax_agreement": float((stored["probs"].argmax(1) == got["probs_c"].argmax(1)).mean()),
                                   "n_argmax_differ": int((stored["probs"].argmax(1) != got["probs_c"].argmax(1)).sum()),
                                   "max_abs_prob_diff": float(np.abs(stored["probs"] - got["probs_c"]).max())},
           "ort_cpu_ms_per_image": {"model_multi": ms_multi, "model": ms_ref}, "wall_s": round(run_s, 1)}
    if Fte is not None:
        z_ref = ((Fte - mu) / sd) @ W.T + b
        ver["logits_c_vs_head_on_cached_features_max_abs_diff"] = float(np.abs(got["logits_c"] - z_ref).max())
    # ---- direct feature check: a throwaway variant exposing 'select', written next to model.onnx.data (the only
    # place onnxruntime / the onnx checker accept the external data; no links) and deleted right after use
    if feature_check:
        tmp = FINAL / "_verify_feature.onnx"
        try:
            add_c_head(src, tmp, W, b, mu, sd, extra_outputs=(FEATURE_TENSOR,))
            sf = ort_session(tmp)
            k = 16
            fo = ort_run(sf, X[:k], [FEATURE_TENSOR])[FEATURE_TENSOR]
            del sf
        finally:
            if tmp.exists():
                tmp.unlink()
        if Fte is not None:
            cos = (fo * Fte[:k]).sum(1) / (np.linalg.norm(fo, axis=1) * np.linalg.norm(Fte[:k], axis=1))
            ver["feature_check"] = {"n_images": k, "tensor": FEATURE_TENSOR,
                                    "vs": "results/cache/features (compclass.features.extract_features, fp16 autocast GPU)",
                                    "max_abs_diff": float(np.abs(fo - Fte[:k]).max()),
                                    "max_rel_diff": float((np.abs(fo - Fte[:k]).max(1) / np.abs(Fte[:k]).max(1)).max()),
                                    "min_cosine": float(cos.min()), "feature_abs_max": float(np.abs(Fte[:k]).max())}
    sha_after = {"model.onnx": sha256(src), "model.onnx.data": sha256(data)}
    ver["deployed_files_unchanged"] = sha_after == {k: sha[k] for k in sha_after}
    desc = {
        "file": "model_multi.onnx", "graph_bytes": dst.stat().st_size, "sha256": sha256(dst),
        "external_data": "model.onnx.data (shared with model.onnx, unchanged)", "built_from": sha,
        "input": {"name": "input", "shape": [1, 3, 384, 384], "dtype": "float32",
                  "preprocess": "exports/final/preprocess.json (identical for both heads)"},
        "labels": "exports/final/labels.json (index order of every output)",
        "outputs": [
            {"name": "logits", "model": "deployed final model (R2_deployed_fused)", "shape": [1, 20],
             "meaning": "temperature-scaled fused log-scores (probe + 0.25 x zero-shot) / T; identical to model.onnx"},
            {"name": "probs", "model": "deployed final model (R2_deployed_fused)", "shape": [1, 20],
             "meaning": "calibrated probabilities = softmax(logits); bit-identical to model.onnx 'probs'; use "
                        "abstain_threshold from preprocess.json"},
            {"name": "logits_c", "model": "ITRI626 model C (SigLIP So400m/14@384 frozen + nn.Linear(1152,20))",
             "shape": [1, 20], "meaning": "W (f - mu)/sd + b with f = pooled feature 'select'; standardisation folded"},
            {"name": "probs_c", "model": "ITRI626 model C", "shape": [1, 20],
             "meaning": "softmax(logits_c), not temperature-calibrated"}],
        "c_head": {"source": str(head_path), "sha256": sha256(head_path), "train_rows": "dev folds 1-4 (2010)",
                   "mu_sd": "head.pt keys 'mu','sd' = per-feature mean / population std of the train-split cached features",
                   "settings": st.get("settings"), "best_epoch": st.get("best_epoch"),
                   "fold": "c_weight = (W / sd).T, c_bias = b - (W / sd) @ mu (float64, stored float32)"},
        "surgery": surgery, "checks": checks, "verification": ver,
        "run_both_heads_cost": "one backbone pass; the C head adds one 1152x20 MatMul",
    }
    jdump(desc, FINAL / "model_multi.json")
    print(json.dumps({"checks": checks, "verification": ver}, default=str), flush=True)
    return desc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["all", "ab", "c"])
    ap.add_argument("--no-feature-check", action="store_true", help="c: skip the direct check of tensor 'select' "
                    "against the cached features (writes and deletes exports/final/_verify_feature.onnx)")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    import torch

    torch.set_num_threads(a.threads)
    import onnx
    import onnxruntime

    summary = {"versions": {"python": platform.python_version(), "torch": torch.__version__, "onnx": onnx.__version__,
                            "onnxruntime": onnxruntime.__version__}, "opset": OPSET}
    if a.cmd in ("all", "ab"):
        m, rows, _ = test_rows()
        q = image_quirks(m, rows)
        summary["image_quirks_test"] = q
        for name in CNN_MODELS:
            r = export_cnn(name, q)
            summary[name] = {"parity": r["parity"], "test": r["test"], "onnx_bytes": r["onnx_bytes"]}
    if a.cmd in ("all", "c"):
        d = export_c(not a.no_feature_check)
        summary[C_NAME] = {"model_multi": d["verification"], "checks": d["checks"]}
    jdump(summary, OUT / f"export_summary_{a.cmd}.json")


if __name__ == "__main__":
    main()
