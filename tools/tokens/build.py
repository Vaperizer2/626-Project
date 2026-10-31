"""Build token-count candidates of the deployed SigLIP So400m trunk for the frozen speed harness (tools/speed).

    build.py --list
    build.py --inputs                  # legion: 6 preprocessed dev images per recipe size, for --check anywhere
    build.py <candidate_id> [--check]  # build into $TOKENS_OUT/<id> (default 626-artifacts/tokens/candidates)

Every candidate keeps the frozen So400m weights and changes only how many tokens the 27 blocks see. Run with the
harness venv on legion, or the project venv on carlpc2 with TOKENS_BASELINE / TOKENS_OUT pointing at its copies.

Recipes without merging edit the baseline ONNX graph instead of re-exporting it: the position-embedding initializer
is resampled to the new grid and every Reshape target holding the token count 729 becomes the new count, so
TensorRT sees the deployed op sequence. Recipes with merging export ``siglip.Trunk`` from torch, then point every
weight whose bytes equal a baseline tensor at that tensor's offset in the baseline ``model.onnx.data``. The graph
file is then a few MB and the weights are a reflink of the baseline file, so it ships between machines cheaply.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
BASELINE = Path(os.environ.get("TOKENS_BASELINE",
                               "/home/vaper/Work/Prototypes/626-artifacts/harness/candidates/baseline_fp32"))
OUT = Path(os.environ.get("TOKENS_OUT", "/home/vaper/Work/Prototypes/626-artifacts/tokens/candidates"))
INPUTS = OUT.parent / "check_inputs"
PATCH, GRID, DEPTH, DEPLOYED_SHORT_SIDE = 14, 27, 27, 427
DATA = "model.onnx.data"
TRT_FP16 = {"provider": "TensorrtExecutionProvider",
            "options": {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": "trt_cache",
                        "trt_timing_cache_enable": True, "trt_timing_cache_path": "trt_cache",
                        "trt_max_workspace_size": 2147483648}}


def const(r: int, first: int = 0) -> tuple[int, ...]:
    """ToMe's constant schedule: merge r tokens after the attention of every block from ``first`` on."""
    return tuple(r if i >= first else 0 for i in range(DEPTH))


@dataclass(frozen=True)
class Recipe:
    """size: input px. The stride-14 patch conv floors, so the grid is size // 14 (the deployed 384 px gives 27 and
    ignores the last 6 pixel rows and columns). pos: how the 27x27 position embedding reaches that grid,
    'interp' (bicubic, antialiased, as timm.resample_abs_pos_embed; same field of view, coarser pixels) or
    'center' (the central grid x grid slice at the deployed pixel scale; a tighter crop, no resampling).
    merge: tokens merged after each block's attention (siglip.bipartite_merge); None keeps the deployed graph,
    an all-zero schedule exports the torch rebuild unchanged (the control for the merge recipes)."""
    hypothesis: str
    size: int
    pos: str = "interp"
    merge: tuple[int, ...] | None = None

    @property
    def grid(self) -> int:
        return self.size // PATCH

    @property
    def crop_pct(self) -> float:
        return 0.9 if self.pos == "interp" else self.size / DEPLOYED_SHORT_SIDE

    def tokens(self) -> list[int]:
        """Token count each block's MLP sees."""
        t, out = self.grid * self.grid, []
        for r in self.merge or (0,) * DEPTH:
            t -= r
            out.append(t)
        return out


RECIPES = {
    "tokens_res384_trtfp16": Recipe("control: surgery at the deployed grid is the identity", 384),
    "tokens_res336_trtfp16": Recipe("H1 lower resolution, 576 tokens", 336),
    "tokens_res308_trtfp16": Recipe("H1 lower resolution, 484 tokens", 308),
    "tokens_res280_trtfp16": Recipe("H1 lower resolution, 400 tokens", 280),
    "tokens_res252_trtfp16": Recipe("H1 lower resolution, 324 tokens", 252),
    "tokens_res224_trtfp16": Recipe("H1 lower resolution, 256 tokens", 224),
    "tokens_torch384_trtfp16": Recipe("control: torch rebuild, no merging", 384, merge=const(0)),
    "tokens_tome384_r16_trtfp16": Recipe("H2 ToMe r=16 every block at 384", 384, merge=const(16)),
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


def resized_graph(r: Recipe) -> onnx.ModelProto:
    m = onnx.load(BASELINE / "model.onnx", load_external_data=False)
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    pos = inits["trunk.pos_embed"]
    load_external_data_for_tensor(pos, str(BASELINE))
    pos.CopyFrom(numpy_helper.from_array(resample_pos(numpy_helper.to_array(pos), r), pos.name))
    tokens = r.grid * r.grid
    for name in {n.input[1] for n in g.node if n.op_type == "Reshape"}:
        shape = numpy_helper.to_array(inits[name]).copy()
        assert GRID not in shape, f"{name} {shape}: a grid-shaped Reshape needs its own rewrite"
        if (shape == GRID * GRID).any():
            shape[shape == GRID * GRID] = tokens
            inits[name].CopyFrom(numpy_helper.from_array(shape, name))
    dims = g.input[0].type.tensor_type.shape.dim
    dims[2].dim_value = dims[3].dim_value = r.size
    return m


def exported_graph(r: Recipe) -> onnx.ModelProto:
    import torch

    import siglip

    w = siglip.onnx_weights(BASELINE / "model.onnx")
    if r.pos != "interp":
        w["pos_embed"] = resample_pos(w["pos_embed"], r)
    model = siglip.Trunk(w, r.size, r.merge).eval()
    del w
    with torch.no_grad():
        m = siglip.export(model, r.size)
    return relink(m)


def relink(m: onnx.ModelProto, min_bytes: int = 4096) -> onnx.ModelProto:
    """Point each initializer whose raw bytes equal a baseline weight at that weight's slice of the baseline data
    file. Content-addressed, so it is exact whatever the exporter named or ordered the tensors."""
    base = onnx.load(BASELINE / "model.onnx", load_external_data=False)
    where = {}
    with open(BASELINE / DATA, "rb") as f:
        for t in base.graph.initializer:
            if t.data_location == onnx.TensorProto.EXTERNAL:
                e = {x.key: x.value for x in t.external_data}
                f.seek(int(e.get("offset", 0)))
                where[hashlib.sha256(f.read(int(e["length"]))).digest()] = (e.get("offset", "0"), e["length"])
    internal = 0
    for t in m.graph.initializer:
        if len(t.raw_data) < min_bytes:
            continue
        hit = where.get(hashlib.sha256(t.raw_data).digest())
        if hit is None:
            internal += len(t.raw_data)
            continue
        t.ClearField("raw_data")
        t.data_location = onnx.TensorProto.EXTERNAL
        del t.external_data[:]
        for k, v in (("location", DATA), ("offset", hit[0]), ("length", hit[1])):
            t.external_data.add(key=k, value=str(v))
    print(f"relinked to the baseline weights; {internal / 2**20:.1f} MiB stay inside the graph")
    assert internal < 64 * 2**20, "most weights should match the baseline; a transposed or folded weight slipped in"
    return m


def build(cid: str) -> Path:
    r = RECIPES[cid]
    dst = OUT / cid
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    m = resized_graph(r) if r.merge is None else exported_graph(r)
    onnx.save(m, dst / "model.onnx")
    subprocess.run(["cp", "--reflink=auto", str(BASELINE / DATA), str(dst / DATA)], check=True)
    onnx.checker.check_model(str(dst / "model.onnx"))
    sha = os.environ.get("TOKENS_SHA") or subprocess.run(
        ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    pre = json.loads((BASELINE / "preprocess.json").read_text())
    how = "baseline graph edited" if r.merge is None else f"torch rebuild, merge {list(r.merge)}"
    pre |= {"input_size": r.size, "crop_pct": r.crop_pct, "resize_short_side": round(r.size / r.crop_pct),
            "zero_shot_fusion": True, "tokens_per_block": r.tokens(),
            "built_from": f"baseline_fp32 weights, {r.grid}x{r.grid} patches, pos_embed {r.pos}, {how} "
                          f"(tools/tokens/build.py {cid} @ {sha})"}
    (dst / "preprocess.json").write_text(json.dumps(pre, indent=1))
    (dst / "runtime.json").write_text(json.dumps(TRT_FP16, indent=1))
    shutil.copy(BASELINE / "labels.json", dst / "labels.json")
    print(f"{cid}: {r.size}px, short side {pre['resize_short_side']}, tokens {r.tokens()[0]}..{r.tokens()[-1]} "
          f"-> {dst}")
    return dst


def write_inputs(n: int = 6) -> None:
    """Preprocess the same n dev images at every recipe's (size, crop) with the harness transform (legion only)."""
    sys.path.insert(0, str(HARNESS))
    from evaluate import dev_rows, preprocessed
    from harness import Candidate

    m, rows = dev_rows()
    rows = rows[::len(rows) // n][:n]
    INPUTS.mkdir(parents=True, exist_ok=True)
    for r in {(r.size, r.crop_pct) for r in RECIPES.values()}:
        pre = {"input_size": r[0], "crop_pct": r[1], "interpolation": "bicubic", "mean": [0.5] * 3, "std": [0.5] * 3}
        c = Candidate(Path("."), "inputs", "", pre, {}, 0, None)
        np.save(INPUTS / f"{r[0]}_{r[1]:.4f}.npy", preprocessed(c, m, rows))


def embed(model_dir: Path, X: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    s = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(["embedding"], {"pixel_values": X[i:i + 1]})[0] for i in range(len(X))])


def check(cid: str) -> dict:
    """CPU fp32 embeddings of the check images, candidate vs baseline (cosine) and, for torch recipes, the ONNX
    graph vs the torch module it came from (export fidelity of the merge ops)."""
    r = RECIPES[cid]
    X = np.load(INPUTS / f"{r.size}_{r.crop_pct:.4f}.npy")
    a = embed(BASELINE, np.load(INPUTS / f"384_{0.9:.4f}.npy"))
    b = embed(OUT / cid, X)
    cos = (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    rep = {"candidate": cid, "n_images": len(X), "cosine_to_baseline": [round(float(x), 5) for x in cos],
           "max_abs_diff_to_baseline": float(np.abs(a - b).max()), "finite": bool(np.isfinite(b).all())}
    if r.merge is not None:
        import torch

        import siglip

        w = siglip.onnx_weights(BASELINE / "model.onnx")
        if r.pos != "interp":
            w["pos_embed"] = resample_pos(w["pos_embed"], r)
        with torch.no_grad():
            t = siglip.Trunk(w, r.size, r.merge).eval()(torch.from_numpy(X)).numpy()
        rep["max_abs_diff_onnx_vs_torch"] = float(np.abs(t - b).max())
    (OUT / cid / "check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--inputs", action="store_true", help="write the check images (legion)")
    ap.add_argument("--check", action="store_true", help="CPU embedding check against the baseline after building")
    a = ap.parse_args()
    if a.inputs:
        write_inputs()
    if a.list or not (a.ids or a.inputs):
        for k, r in RECIPES.items():
            t = r.tokens()
            print(f"{k:30s} {r.size}px pos={r.pos:6s} tokens {t[0]}->{t[-1]} mean {np.mean(t):.0f}  {r.hypothesis}")
    for cid in a.ids:
        build(cid)
        if a.check:
            check(cid)


if __name__ == "__main__":
    main()
