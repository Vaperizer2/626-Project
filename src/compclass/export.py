"""Export a trained checkpoint to ONNX with the metadata a deployment needs.

    uv run python -m compclass.export --checkpoint runs/<run>/best.pt --out exports/<name>

Writes ``model.onnx`` (input ``input`` float32 NCHW RGB, output ``logits``), ``labels.json`` (class index ->
name) and ``preprocess.json`` (resize short side, centre-crop size, mean/std, interpolation). An OpenCV or
onnxruntime client reproduces training-time eval preprocessing from preprocess.json alone:

    rgb = resize_short_side(img, resize_short_side) -> centre_crop(input_size) -> /255 -> (x - mean) / std

Finally it runs the same random batch through PyTorch and onnxruntime and reports the max |logit| gap and
whether the argmax agrees (parity check).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .models import build_model


def load_checkpoint_model(checkpoint: str | Path) -> tuple[torch.nn.Module, dict]:
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mc = ck["config"]["model"]
    model, _ = build_model(mc["key"], len(ck["classes"]), pretrained=False, **mc.get("kwargs", {}))
    model.load_state_dict(ck["model"])
    return model.eval(), ck


def export_onnx(checkpoint: str | Path, out_dir: str | Path, dynamic_batch: bool = False,
                opset: int = 18) -> dict:
    """Export + sidecars + parity check. Static batch 1 (default) is the most OpenCV-DNN-friendly graph."""
    model, ck = load_checkpoint_model(checkpoint)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    size = ck["preprocess"]["input_size"]
    x = torch.randn(1, 3, size, size)
    dyn = ({0: torch.export.Dim("batch", min=1, max=256)},) if dynamic_batch else None
    onnx_path = out / "model.onnx"
    torch.onnx.export(model, (x,), onnx_path, input_names=["input"], output_names=["logits"], opset_version=opset,
                      dynamo=True, dynamic_shapes=dyn, external_data=False)
    (out / "labels.json").write_text(json.dumps(ck["classes"], indent=1))
    pre = {**ck["preprocess"], "color_order": "RGB", "layout": "NCHW", "scale": 1 / 255, "output": "logits",
           "model": ck["config"]["model"]["key"], "config_hash": ck["config_hash"], "fold": ck["fold"],
           "manifest_fingerprint": ck["manifest_fingerprint"]}
    (out / "preprocess.json").write_text(json.dumps(pre, indent=1))
    report = {"onnx": str(onnx_path), **parity_check(model, onnx_path, size, batch=4 if dynamic_batch else 1)}
    print(json.dumps(report))
    return report


def parity_check(model: torch.nn.Module, onnx_path: Path, size: int, batch: int = 1, trials: int = 3) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    gaps, agree = [], []
    for seed in range(trials):
        x = torch.randn(batch, 3, size, size, generator=torch.Generator().manual_seed(seed))
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
        gaps.append(float(np.abs(ref - got).max()))
        agree.append(bool((ref.argmax(1) == got.argmax(1)).all()))
    return {"max_abs_logit_diff": max(gaps), "argmax_agree": all(agree), "parity_ok": all(agree) and max(gaps) < 1e-3}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dynamic-batch", action="store_true")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args(argv)
    export_onnx(args.checkpoint, args.out, args.dynamic_batch, args.opset)


if __name__ == "__main__":
    main()
