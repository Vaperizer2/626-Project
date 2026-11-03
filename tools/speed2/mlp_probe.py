"""Diagnostic: does TensorRT overlap independent MLP neuron groups? Synthetic INT8 QDQ MLP blocks, random weights.

    tools/quant/py tools/speed2/mlp_probe.py <out_dir> [--tokens 441] [--layers 4] [--groups 4304 3840,464 ...]

Each layer is x + fc2(GELU(fc1(LN(x)))) with fc1/fc2 INT8 (per-channel weights, per-tensor activations), as the
trunk's MLP after the quant recipe. A groups spec like 3840,464 splits the hidden neurons into branches whose fc2
outputs are summed, which is the same function; a spec starting with ``k:`` (k:2176,2128) keeps fc1 whole and
splits only fc2 along its reduction dim (Slice of the GELU output, one fc2 per slice, summed). Runs tools/quant/trtprobe.py --fp16 --int8 per spec. Not a result.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

QUANT = Path(__file__).resolve().parents[1] / "quant"
D = 1152


def graph(tokens: int, layers: int, groups: list[int], k_split: bool = False) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    n, inits, nodes = helper.make_node, [], []

    def init(name: str, a: np.ndarray) -> str:
        inits.append(numpy_helper.from_array(a, name))
        return name

    def qdq_act(x: str, tag: str, amax: float) -> str:
        s, z = init(f"{tag}.s", np.array(amax / 127, np.float32)), init(f"{tag}.z", np.array(0, np.int8))
        nodes.extend([n("QuantizeLinear", [x, s, z], [f"{tag}.q"]), n("DequantizeLinear", [f"{tag}.q", s, z], [f"{tag}.dq"])])
        return f"{tag}.dq"

    def linear(x: str, tag: str, k: int, m: int) -> str:
        w = rng.normal(0, k ** -0.5, (k, m)).astype(np.float32)
        sw = np.abs(w).max(0) / 127
        wq = init(f"{tag}.wq", np.clip(np.round(w / sw), -127, 127).astype(np.int8))
        nodes.append(n("DequantizeLinear", [wq, init(f"{tag}.ws", sw.astype(np.float32)),
                                            init(f"{tag}.wz", np.zeros(m, np.int8))], [f"{tag}.w"], axis=1))
        nodes.extend([n("MatMul", [qdq_act(x, f"{tag}.x", 4.0), f"{tag}.w"], [f"{tag}.mm"]),
                      n("Add", [f"{tag}.mm", init(f"{tag}.b", np.zeros(m, np.float32))], [f"{tag}.y"])])
        return f"{tag}.y"

    x = "x"
    for i in range(layers):
        nodes.append(n("LayerNormalization", [x, init(f"l{i}.g", np.ones(D, np.float32)),
                                              init(f"l{i}.be", np.zeros(D, np.float32))], [f"l{i}.ln"], axis=-1))
        outs = []
        for j, h in enumerate([sum(groups)] if k_split else groups):
            a = linear(f"l{i}.ln", f"l{i}.g{j}.fc1", D, h)
            sq2 = init(f"l{i}.g{j}.r2", np.array(2 ** -0.5, np.float32))
            nodes.extend([n("Mul", [a, sq2], [f"l{i}.g{j}.u"]), n("Erf", [f"l{i}.g{j}.u"], [f"l{i}.g{j}.e"]),
                          n("Add", [f"l{i}.g{j}.e", init(f"l{i}.g{j}.one", np.array(1, np.float32))], [f"l{i}.g{j}.e1"]),
                          n("Mul", [a, f"l{i}.g{j}.e1"], [f"l{i}.g{j}.ae"]),
                          n("Mul", [f"l{i}.g{j}.ae", init(f"l{i}.g{j}.half", np.array(0.5, np.float32))], [f"l{i}.g{j}.act"])])
            if not k_split:
                outs.append(linear(f"l{i}.g{j}.act", f"l{i}.g{j}.fc2", h, D))
                continue
            lo = 0
            for c, kc in enumerate(groups):
                st, en = init(f"l{i}.k{c}.st", np.array([lo], np.int64)), init(f"l{i}.k{c}.en", np.array([lo + kc], np.int64))
                nodes.append(n("Slice", [f"l{i}.g{j}.act", st, en, init(f"l{i}.k{c}.ax", np.array([2], np.int64))],
                               [f"l{i}.k{c}.a"]))
                outs.append(linear(f"l{i}.k{c}.a", f"l{i}.k{c}.fc2", kc, D))
                lo += kc
        acc = x
        for j, o in enumerate(outs):
            nodes.append(n("Add", [acc, o], [f"l{i}.r{j}"]))
            acc = f"l{i}.r{j}"
        x = acc
    g = helper.make_graph(nodes, "mlp", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, tokens, D])],
                          [helper.make_tensor_value_info(x, TensorProto.FLOAT, [1, tokens, D])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 10
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--tokens", type=int, default=441)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--groups", nargs="+", default=["4304", "3840,464", "3840"])
    a = ap.parse_args()
    res = {}
    for spec in a.groups:
        d = a.out / f"t{a.tokens}_l{a.layers}_g{spec.replace(',', '-').replace(':', '')}"
        d.mkdir(parents=True, exist_ok=True)
        k_split = spec.startswith("k:")
        onnx.save(graph(a.tokens, a.layers, [int(h) for h in spec.removeprefix("k:").split(",")], k_split),
                  d / "mlp.onnx")
        r = subprocess.run([sys.executable, str(QUANT / "trtprobe.py"), str(d / "mlp.onnx"), str(d / "probe"), "--fp16",
                            "--int8", "--iters", "200"], capture_output=True, text=True, cwd=QUANT)
        if r.returncode:
            raise SystemExit(r.stderr[-2000:])
        layers = json.loads((d / "probe" / "layers.json").read_text())["Layers"]
        prof = json.loads((d / "probe" / "profile.json").read_text())
        res[spec] = {"execute": next(ln for ln in r.stdout.splitlines() if "execute_v2" in ln),
                     "streams": sorted({x.get("StreamId", 0) for x in layers}),
                     "top": sorted(((round(v, 4), k[:50], next((x.get("TacticName", "")[:60] for x in layers
                                                                  if x["Name"] == k), "")) for k, v in prof.items()),
                                   reverse=True)[:8]}
        print(spec, json.dumps(res[spec]), flush=True)
    (a.out / f"mlp_t{a.tokens}_l{a.layers}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
