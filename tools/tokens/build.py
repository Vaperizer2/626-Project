"""Build token-count candidates of the deployed SigLIP So400m trunk for the frozen speed harness (tools/speed).

    tools/tokens/build.py <candidate_id> [...]   # build into 626-artifacts/tokens/candidates/<id>
    tools/tokens/build.py --list

Every candidate keeps the frozen So400m weights and changes only how many patch tokens the 27 blocks see.
Run with the harness venv (torch CPU + onnx + onnxruntime):
    /home/vaper/Work/Prototypes/626-artifacts/harness/venv/bin/python tools/tokens/build.py <id>

Resolution recipes edit the baseline ONNX graph in place of re-exporting it: the position-embedding initializer is
resampled to the new k x k grid and every Reshape target holding the token count 729 becomes k*k, so the op
sequence TensorRT sees is the deployed one. ``check`` runs the result and the baseline on CPU on dev images.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from onnx.external_data_helper import load_external_data_for_tensor

HARNESS = Path("/home/vaper/Work/Prototypes/626-wt/harness/tools/speed")
BASELINE = Path("/home/vaper/Work/Prototypes/626-artifacts/harness/candidates/baseline_fp32")
OUT = Path("/home/vaper/Work/Prototypes/626-artifacts/tokens/candidates")
PATCH, GRID, DEPLOYED_SHORT_SIDE = 14, 27, 427
TRT_FP16 = {"provider": "TensorrtExecutionProvider",
            "options": {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": "trt_cache",
                        "trt_timing_cache_enable": True, "trt_timing_cache_path": "trt_cache",
                        "trt_max_workspace_size": 2147483648}}


@dataclass(frozen=True)
class Recipe:
    """size: input px. The stride-14 patch conv floors, so the grid is size // 14 (the deployed 384 px gives 27 and
    ignores the last 6 pixel rows and columns). pos: how the 27x27 position embedding reaches that grid,
    'interp' (bicubic, antialiased, as timm.resample_abs_pos_embed; same field of view, coarser pixels) or
    'center' (the central grid x grid slice at the deployed pixel scale; a tighter crop, no resampling)."""
    hypothesis: str
    size: int
    pos: str = "interp"

    @property
    def grid(self) -> int:
        return self.size // PATCH

    @property
    def crop_pct(self) -> float:
        return 0.9 if self.pos == "interp" else self.size / DEPLOYED_SHORT_SIDE


RECIPES = {
    "tokens_res384_trtfp16": Recipe("control: surgery at the deployed grid is the identity", 384),
    "tokens_res336_trtfp16": Recipe("H1 lower resolution, 576 tokens", 336),
    "tokens_res308_trtfp16": Recipe("H1 lower resolution, 484 tokens", 308),
    "tokens_res280_trtfp16": Recipe("H1 lower resolution, 400 tokens", 280),
    "tokens_res252_trtfp16": Recipe("H1 lower resolution, 324 tokens", 252),
    "tokens_res224_trtfp16": Recipe("H1 lower resolution, 256 tokens", 224),
}


def resample_pos(pos: np.ndarray, r: Recipe) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    c = pos.shape[-1]
    grid = torch.from_numpy(pos.copy()).reshape(1, GRID, GRID, c).permute(0, 3, 1, 2)
    if r.pos == "interp":
        out = F.interpolate(grid, size=(r.grid, r.grid), mode="bicubic", antialias=True, align_corners=False)
    else:
        o = (GRID - r.grid) // 2
        out = grid[:, :, o:o + r.grid, o:o + r.grid]
    return out.permute(0, 2, 3, 1).reshape(1, r.grid * r.grid, c).contiguous().numpy()


def resize_graph(m: onnx.ModelProto, r: Recipe) -> None:
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    pos = inits["trunk.pos_embed"]
    load_external_data_for_tensor(pos, str(BASELINE))
    new = numpy_helper.from_array(resample_pos(numpy_helper.to_array(pos), r), pos.name)
    pos.CopyFrom(new)
    tokens = r.grid * r.grid
    for name in {n.input[1] for n in g.node if n.op_type == "Reshape"}:
        shape = numpy_helper.to_array(inits[name]).copy()
        assert GRID not in shape, f"{name} {shape}: a grid-shaped Reshape needs its own rewrite"
        if (shape == GRID * GRID).any():
            shape[shape == GRID * GRID] = tokens
            inits[name].CopyFrom(numpy_helper.from_array(shape, name))
    dims = g.input[0].type.tensor_type.shape.dim
    dims[2].dim_value = dims[3].dim_value = r.size


def build(cid: str) -> Path:
    r = RECIPES[cid]
    dst = OUT / cid
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    m = onnx.load(BASELINE / "model.onnx", load_external_data=False)
    resize_graph(m, r)
    onnx.save(m, dst / "model.onnx")
    subprocess.run(["cp", "--reflink=auto", str(BASELINE / "model.onnx.data"), str(dst / "model.onnx.data")],
                   check=True)
    onnx.checker.check_model(str(dst / "model.onnx"))
    sha = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    pre = json.loads((BASELINE / "preprocess.json").read_text())
    pre |= {"input_size": r.size, "crop_pct": r.crop_pct, "resize_short_side": round(r.size / r.crop_pct),
            "zero_shot_fusion": True,
            "built_from": f"baseline_fp32 trunk, {r.grid}x{r.grid} tokens, pos_embed {r.pos} "
                          f"(tools/tokens/build.py {cid} @ {sha})"}
    (dst / "preprocess.json").write_text(json.dumps(pre, indent=1))
    (dst / "runtime.json").write_text(json.dumps(TRT_FP16, indent=1))
    shutil.copy(BASELINE / "labels.json", dst / "labels.json")
    print(f"{cid}: {r.size}px, short side {pre['resize_short_side']}, {r.grid * r.grid} tokens -> {dst}")
    return dst


def check(cid: str, n: int = 6) -> dict:
    """CPU fp32 embeddings of n dev images: candidate vs baseline. At grid 27 they must agree exactly."""
    import onnxruntime as ort

    sys.path.insert(0, str(HARNESS))
    from evaluate import dev_rows, preprocessed
    from harness import load_candidate

    m, rows = dev_rows()
    step = len(rows) // n
    rows = rows[::step][:n]
    emb = {}
    for d in (BASELINE, OUT / cid):
        c = load_candidate(d)
        X = preprocessed(c, m, rows)
        s = ort.InferenceSession(str(d / "model.onnx"), providers=["CPUExecutionProvider"])
        emb[d.name] = np.concatenate([s.run(["embedding"], {"pixel_values": X[i:i + 1]})[0] for i in range(n)])
        del s
    a, b = emb[BASELINE.name], emb[cid]
    cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    rep = {"candidate": cid, "n_images": n, "cosine_to_baseline": [round(float(x), 5) for x in cos],
           "max_abs_diff": float(np.abs(a - b).max()), "finite": bool(np.isfinite(b).all())}
    (OUT / cid / "check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true", help="CPU embedding check against the baseline after building")
    a = ap.parse_args()
    if a.list or not a.ids:
        for k, r in RECIPES.items():
            print(f"{k:28s} {r.size}px {r.grid * r.grid:4d} tokens pos={r.pos:6s} {r.hypothesis}")
        return
    for cid in a.ids:
        build(cid)
        if a.check:
            check(cid)


if __name__ == "__main__":
    main()
