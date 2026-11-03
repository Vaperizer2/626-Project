"""Bisection aid for bug 2: undo onnxruntime's MatMulAddFusion for chosen GEMMs of a level-1 optimised graph.

    revert_gemm.py <basic.onnx> <out.onnx> --roles PAT,...   (substring of the bias name, e.g. blocks__3__,attn_pool)

MatMulAddFusion rewrites ``MatMul(x, W) + b`` on a static 3-D x as ``Reshape(x, [T, C]) -> Gemm(., W, b) ->
Reshape(., [1, T, N])``, and QDQPropagation then repeats x's Q/DQ pair after the first Reshape. For each Gemm whose
bias name contains a pattern this puts back ``MatMul(x, W) -> Add(., b)`` with x the tensor the first Reshape read,
feeding the trailing Reshape (kept: ReshapeFusion may have merged it with the next Reshape, e.g. the packed qkv
split), and drops the nodes nothing reads any more. Same math, so CUDA EP gives the same output.
"""

from __future__ import annotations

import argparse

import onnx
from onnx import helper


def revert(m: onnx.ModelProto, pats: list[str]) -> int:
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    cons: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    swap, k = {}, 0
    for n in g.node:
        if n.op_type != "Gemm" or not any(p in n.input[2] for p in pats):
            continue
        t = n.input[0]
        while prod[t].op_type != "Reshape":
            t = prod[t].input[0]
        x = prod[t].input[0]
        post = cons[n.output[0]][0]
        assert post.op_type == "Reshape" and len(cons[n.output[0]]) == 1
        mm = helper.make_node("MatMul", [x, n.input[1]], [f"revert.{k}.mm"])
        add = helper.make_node("Add", [f"revert.{k}.mm", n.input[2]], [f"revert.{k}.add"])
        post.input[0] = f"revert.{k}.add"
        swap[id(n)] = [mm, add]
        k += 1
    nodes = [x for n in g.node for x in swap.get(id(n), [n])]
    live, needed = [], {o.name for o in g.output}
    for n in reversed(nodes):
        if any(o in needed for o in n.output):
            live.append(n)
            needed.update(n.input)
    del g.node[:]
    g.node.extend(reversed(live))
    del g.value_info[:]
    return k


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--roles", required=True)
    a = ap.parse_args()
    m = onnx.load(a.src, load_external_data=False)
    k = revert(m, a.roles.split(","))
    onnx.save(m, a.dst)
    print(f"{a.dst}: {k} Gemm reverted to MatMul + Add")


if __name__ == "__main__":
    main()
