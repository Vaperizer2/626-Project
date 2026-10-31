"""INT8 explicit-quantisation (QDQ) candidates of the So400m trunk for the TensorRT execution provider.

    qdq.py smooth <src_dir> <stats.npz> <dst_dir> --alpha A
        SmoothQuant, folded exactly into the fp32 graph: for the qkv and fc1 GEMMs the per-input-channel factor s
        divides that block's LayerNorm gamma/beta and multiplies the weight rows; for attn.proj it divides the V
        columns of qkv (weight and bias) and multiplies the proj weight rows. Still a float32 graph.
    qdq.py qdq <src_dir> <stats.npz> <dst_dir> [--calib max|p99.99|...] [--calib-override PAT=CALIB,...]
               [--exclude PAT,...] [--runtime JSON]
        Weights: int8 initializers, symmetric per output channel (DequantizeLinear axis=1).
        Activations: per-tensor symmetric scale amax/127 from the pool statistics of <src_dir>'s graph, with
        --calib for every GEMM except roles matching an override pattern (first match wins).
        GEMMs whose role matches an --exclude substring stay in floating point (TensorRT runs them in fp16).

GEMM roles are named from the bias they feed, e.g. ``blocks.3.mlp.fc2`` or ``attn_pool.kv``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from calib_stats import gemms
from candidate import TRT_FP16, git_sha

D = 1152
DEFAULT_EXCLUDE = "blocks.0.,blocks.26.,attn_pool."


def roles(m: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    """GEMM role (``blocks.<i>.attn.qkv`` ...) -> MatMul node, from the bias initializer its Add consumes."""
    cons = {}
    for n in m.graph.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    out = {}
    for n in gemms(m):
        add = next(c for c in cons[n.output[0]] if c.op_type == "Add")
        bias = next(i for i in add.input if i.startswith("trunk."))
        out[re.sub(r"^trunk\.|\.bias$", "", bias)] = n
    return out


def load(src: Path) -> tuple[onnx.ModelProto, dict[str, np.ndarray]]:
    m = onnx.load(src / "model.onnx")
    return m, {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}


def set_init(m: onnx.ModelProto, name: str, arr: np.ndarray) -> None:
    t = next(t for t in m.graph.initializer if t.name == name)
    t.CopyFrom(numpy_helper.from_array(arr, name))


def save(m: onnx.ModelProto, src: Path, dst: Path, note: str, runtime: dict, recipe: dict) -> None:
    dst.mkdir(parents=True, exist_ok=False)
    onnx.save(m, dst / "model.onnx", save_as_external_data=True, all_tensors_to_one_file=True,
              location="model.onnx.data", size_threshold=1024)
    pre = json.loads((src / "preprocess.json").read_text())
    pre["built_from"] = f"{src} ({note}; tools/quant/qdq.py @ {git_sha()})"
    (dst / "preprocess.json").write_text(json.dumps(pre, indent=1))
    (dst / "runtime.json").write_text(json.dumps(runtime, indent=1))
    (dst / "quant_recipe.json").write_text(json.dumps(recipe, indent=1))
    shutil.copy(src / "labels.json", dst / "labels.json")


def cmax(stats, act: str) -> np.ndarray:
    return stats[f"cmax/{act}"].max(0)


def smooth(src: Path, stats_path: Path, dst: Path, alpha: float) -> None:
    m, w = load(src)
    stats = np.load(stats_path)
    r = roles(m)
    ln = {n.output[0]: n for n in m.graph.node if n.op_type == "LayerNormalization"}
    factors = {}

    def s_for(role: str) -> np.ndarray:
        n = r[role]
        a = np.maximum(cmax(stats, n.input[0]), 1e-5)
        wr = np.maximum(np.abs(w[n.input[1]]).max(1), 1e-5)
        s = (a ** alpha / wr ** (1 - alpha)).astype(np.float32)
        factors[role] = {"min": float(s.min()), "max": float(s.max()), "act_max": float(a.max()),
                         "act_max_after": float((a / s).max())}
        return s

    divided = {}
    blocks = sorted({int(k.split(".")[1]) for k in r if k.startswith("blocks.")})
    for b in blocks:
        for gemm in ("attn.qkv", "mlp.fc1"):
            role = f"blocks.{b}.{gemm}"
            n = r[role]
            norm = ln[n.input[0]]
            s = s_for(role)
            divided[n.input[0]] = s
            set_init(m, norm.input[1], w[norm.input[1]] / s)
            set_init(m, norm.input[2], w[norm.input[2]] / s)
            w[n.input[1]] = w[n.input[1]] * s[:, None]
            set_init(m, n.input[1], w[n.input[1]])
        role = f"blocks.{b}.attn.proj"
        s = s_for(role)
        divided[r[role].input[0]] = s
        qkv = r[f"blocks.{b}.attn.qkv"]
        wq = w[qkv.input[1]].copy()
        wq[:, 2 * D:] /= s
        set_init(m, qkv.input[1], wq)
        bias = f"trunk.blocks.{b}.attn.qkv.bias"
        bq = w[bias].copy()
        bq[2 * D:] /= s
        set_init(m, bias, bq)
        set_init(m, r[role].input[1], w[r[role].input[1]] * s[:, None])
    rt = json.loads((src / "runtime.json").read_text()) if (src / "runtime.json").exists() else {}
    save(m, src, dst, f"SmoothQuant alpha={alpha} folded into LayerNorm and V, float32", rt,
         {"smooth_alpha": alpha, "stats": str(stats_path), "factors": factors})
    np.savez(dst / "stats_max.npz", **smoothed_stats(stats, divided))


def smoothed_stats(stats, divided: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The smoothed graph's activation statistics, derived exactly from the source graph's: every smoothed
    activation is the source one divided per channel by s. Only ``top[:, 0]`` (the per-image max) survives the
    division, so the result supports the ``max`` calibrator alone; deeper ranks are NaN and percentiles refuse."""
    out = {"acts": stats["acts"]}
    for x in stats["acts"]:
        c, t = stats[f"cmax/{x}"], stats[f"top/{x}"]
        if x in divided:
            c = c / divided[x]
            t = np.full_like(t, np.nan)
            t[:, 0] = c.max(1)
        out[f"cmax/{x}"], out[f"top/{x}"] = c, t
    return out


def tokens(role: str) -> int:
    """Rows of the GEMM's activation: 729 patch tokens, except the attention pool's single-query tail."""
    return 1 if role in ("attn_pool.proj", "attn_pool.mlp.fc1", "attn_pool.mlp.fc2") else 729


def amax(stats, act: str, calib: str, n_tokens: int = 729) -> float:
    top = stats[f"top/{act}"]
    if calib == "max":
        return float(top[:, 0].max())
    if np.isnan(top).any():
        raise SystemExit(f"{act}: only per-image maxima known (smoothed statistics); use --calib max")
    q = float(calib.removeprefix("p")) / 100
    n_img, numel = len(top), stats[f"cmax/{act}"].shape[1] * n_tokens
    rank = int(round((1 - q) * n_img * numel))
    pooled = np.sort(top.ravel())[::-1]
    if rank >= len(pooled):
        raise SystemExit(f"{calib} needs rank {rank}, only top-{top.shape[1]} per image kept")
    return float(pooled[rank])


def qdq(src: Path, stats_path: Path, dst: Path, calib: str, exclude: list[str], runtime: dict,
        overrides: dict[str, str] | None = None) -> None:
    m, w = load(src)
    stats = np.load(stats_path)
    g = m.graph
    r = roles(m)
    chosen = {k: n for k, n in r.items() if not any(p in k for p in exclude)}
    new_nodes = {}
    scales = {}
    for role, n in chosen.items():
        x, wname = n.input[0], n.input[1]
        a = amax(stats, x, next((c for p, c in (overrides or {}).items() if p in role), calib), tokens(role))
        scales[role] = a
        sa = f"q.{role}.act_scale"
        g.initializer.extend([numpy_helper.from_array(np.array(a / 127, np.float32), sa),
                              numpy_helper.from_array(np.array(0, np.int8), f"q.{role}.act_zp")])
        W = w[wname]
        sw = np.maximum(np.abs(W).max(0), 1e-8) / 127
        Wq = np.clip(np.round(W / sw), -127, 127).astype(np.int8)
        g.initializer.extend([numpy_helper.from_array(Wq, f"q.{role}.w_int8"),
                              numpy_helper.from_array(sw.astype(np.float32), f"q.{role}.w_scale"),
                              numpy_helper.from_array(np.zeros(len(sw), np.int8), f"q.{role}.w_zp")])
        new_nodes[id(n)] = [
            helper.make_node("QuantizeLinear", [x, sa, f"q.{role}.act_zp"], [f"q.{role}.xq"]),
            helper.make_node("DequantizeLinear", [f"q.{role}.xq", sa, f"q.{role}.act_zp"], [f"q.{role}.xdq"]),
            helper.make_node("DequantizeLinear", [f"q.{role}.w_int8", f"q.{role}.w_scale", f"q.{role}.w_zp"],
                             [f"q.{role}.wdq"], axis=1)]
        n.input[0], n.input[1] = f"q.{role}.xdq", f"q.{role}.wdq"
    nodes = []
    for n in g.node:
        nodes.extend(new_nodes.get(id(n), []))
        nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    used = {i for n in g.node for i in n.input}
    keep = [t for t in g.initializer if t.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    onnx.checker.check_model(m, full_check=False)
    rt = runtime or {"provider": "TensorrtExecutionProvider", "options": {**TRT_FP16, "trt_int8_enable": True}}
    note = f"act {calib}" + "".join(f", {p} {c}" for p, c in (overrides or {}).items())
    save(m, src, dst, f"INT8 QDQ on {len(chosen)}/{len(r)} GEMMs, {note}, exclude {','.join(exclude)}", rt,
         {"calib": calib, "calib_overrides": overrides or {}, "exclude": exclude, "stats": str(stats_path),
          "quantized": sorted(chosen), "fp": sorted(set(r) - set(chosen)), "act_amax": scales})
    print(f"{dst}: {len(chosen)} INT8 GEMMs, {len(r) - len(chosen)} kept floating point")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("smooth", "qdq"):
        p = sub.add_parser(name)
        p.add_argument("src", type=Path)
        p.add_argument("stats", type=Path)
        p.add_argument("dst", type=Path)
    sub.choices["smooth"].add_argument("--alpha", type=float, required=True)
    q = sub.choices["qdq"]
    q.add_argument("--calib", default="max")
    q.add_argument("--calib-override", default="", help="PAT=CALIB,... e.g. mlp.fc2=p99.999")
    q.add_argument("--exclude", default=DEFAULT_EXCLUDE)
    q.add_argument("--runtime", default="")
    a = ap.parse_args()
    if a.cmd == "smooth":
        smooth(a.src, a.stats, a.dst, a.alpha)
    else:
        qdq(a.src, a.stats, a.dst, a.calib, [p for p in a.exclude.split(",") if p],
            json.loads(a.runtime) if a.runtime else {},
            dict(kv.split("=", 1) for kv in a.calib_override.split(",") if kv))


if __name__ == "__main__":
    main()
