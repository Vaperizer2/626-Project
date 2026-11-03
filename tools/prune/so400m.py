"""The deployed So400m trunk graph as named weights, and candidate directories built from edited weights.

Every pruning variant edits initializers of the baseline graph in place and keeps its nodes, so a candidate differs
from `baseline_fp32` only in weight values (and, for structured pruning, in the MLP hidden width). Latency
differences then come from the weights alone, not from a different export.

ONNX stores each Linear as MatMul(x, W) with W of shape [in, out] (the transpose of torch's [out, in]).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ART = Path("/home/vaper/Work/Prototypes/626-artifacts")
BASE = Path(os.environ.get("PRUNE_BASE", ART / "harness/candidates/baseline_fp32"))
DEPTH = 27
BLOCK_LINEARS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
TRT_FP16 = {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": "trt_cache",
            "trt_timing_cache_enable": True, "trt_timing_cache_path": "trt_cache",
            "trt_max_workspace_size": 2147483648}


def load(base: Path = BASE) -> onnx.ModelProto:
    return onnx.load(base / "model.onnx", load_external_data=True)


def linears(g: onnx.GraphProto) -> dict[str, tuple[str, str]]:
    """'blocks.3.mlp.fc1', 'attn_pool.kv', ... -> (MatMul weight initializer [in, out], bias initializer)."""
    inits = {t.name for t in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    out = {}
    for n in g.node:
        bias = [i for i in n.input if i in inits and i.startswith("trunk.") and i.endswith(".bias")]
        if n.op_type != "Add" or not bias:
            continue
        mm = producer.get(next(i for i in n.input if i != bias[0]))
        if mm is not None and mm.op_type == "MatMul" and mm.input[1] in inits:
            out[bias[0][len("trunk."):-len(".bias")]] = (mm.input[1], bias[0])
    return out


def block_linear(i: int, name: str) -> str:
    return f"blocks.{i}.{name}"


class Weights:
    """Numpy views of the graph's initializers by name; `set` writes back (shape may change)."""

    def __init__(self, model: onnx.ModelProto):
        self.model = model
        self.init = {t.name: t for t in model.graph.initializer}
        self.lin = linears(model.graph)
        assert len(self.lin) == DEPTH * len(BLOCK_LINEARS) + 5, f"found {len(self.lin)} linears"

    def get(self, name: str) -> np.ndarray:
        return numpy_helper.to_array(self.init[name])

    def set(self, name: str, a: np.ndarray) -> None:
        t = self.init[name]
        t.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(a, np.float32), name))

    def w(self, lin: str) -> np.ndarray:
        return self.get(self.lin[lin][0])

    def b(self, lin: str) -> np.ndarray:
        return self.get(self.lin[lin][1])

    def set_w(self, lin: str, a: np.ndarray) -> None:
        self.set(self.lin[lin][0], a)

    def set_b(self, lin: str, a: np.ndarray) -> None:
        self.set(self.lin[lin][1], a)

    def torch_state(self) -> dict[str, np.ndarray]:
        """timm `vit_so400m_patch14_siglip_384` (num_classes=0) state dict, Linear weights as [out, in]."""
        sd = {k[len("trunk."):]: self.get(k) for k in self.init if k.startswith("trunk.")}
        for lin, (w, _) in self.lin.items():
            sd[f"{lin}.weight"] = self.get(w).T
        sd["attn_pool.latent"] = self.get("expand")
        return sd

    def load_torch_state(self, sd: dict[str, np.ndarray], prefixes: tuple[str, ...]) -> list[str]:
        """Write the Linear weights/biases of `sd` whose name starts with one of `prefixes` back into the graph."""
        done = []
        for lin, (w, b) in self.lin.items():
            if lin.startswith(prefixes):
                self.set(w, sd[f"{lin}.weight"].T)
                self.set(b, sd[f"{lin}.bias"])
                done.append(lin)
        return done


def save_candidate(model: onnx.ModelProto, out: Path, note: str, runtime: dict, base: Path = BASE,
                   fusion: bool = True) -> Path:
    """Write `out` as a harness candidate: model.onnx + model.onnx.data, preprocess.json, runtime.json."""
    out.mkdir(parents=True, exist_ok=False)
    del model.graph.value_info[:]  # intermediate shapes change under structured pruning; let ORT re-infer them
    model.ir_version = 10
    onnx.save(model, out / "model.onnx", save_as_external_data=True, all_tensors_to_one_file=True,
              location="model.onnx.data", size_threshold=1024)
    pre = json.loads((base / "preprocess.json").read_text())
    pre["zero_shot_fusion"] = fusion
    pre["built_from"] = f"{base} ({note}; tools/prune)"
    (out / "preprocess.json").write_text(json.dumps(pre, indent=1))
    (out / "runtime.json").write_text(json.dumps(runtime, indent=1))
    shutil.copy(base / "labels.json", out / "labels.json")
    return out


def trt_runtime(sparse: bool = False) -> dict:
    opts = dict(TRT_FP16)
    if sparse:
        opts["trt_sparsity_enable"] = True
    return {"provider": "TensorrtExecutionProvider", "options": opts}
