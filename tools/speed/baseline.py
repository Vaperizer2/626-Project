"""export-baseline: the deployed ``exports/final/model.onnx`` with its fused head cut off, as a candidate.

The deployed graph (torch.onnx dynamo export, opset 18) is static at batch 1 because torch.export specialises
size-1 example dimensions. The trunk is cut at the head's input (the tensor the ``Sub(f, mu)`` node reads) and
every Reshape target whose leading 1 is the batch becomes -1, so the same weights and op sequence accept
batches of 1..N. A parity check on real dev images proves the cut and the batch rewrite change nothing.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from harness import BASELINE, DEPLOYED, INPUT, MAX_BATCH, OUTPUT, gpu_lock, load_candidate


def cut_trunk(m: onnx.ModelProto) -> onnx.ModelProto:
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    head_in = next(n.input[0] for n in g.node if n.op_type == "Sub" and n.input[1] == "mu")
    producer = {o: n for n in g.node for o in n.output}
    need, stack = set(), [head_in]
    while stack:
        t = stack.pop()
        if t in producer and id(producer[t]) not in need:
            need.add(id(producer[t]))
            stack.extend(producer[t].input)
    nodes = [n for n in g.node if id(n) in need]
    used = {i for n in nodes for i in n.input}
    rename = {g.input[0].name: INPUT, head_in: OUTPUT}
    for n in nodes:
        n.input[:] = [rename.get(i, i) for i in n.input]
        n.output[:] = [rename.get(o, o) for o in n.output]
    reshape_shapes = {n.input[1] for n in nodes if n.op_type == "Reshape"}
    assert not any(s in reshape_shapes for n in nodes if n.op_type != "Reshape" for s in n.input), \
        "a Reshape target is shared with another op"
    for name in reshape_shapes:
        shape = numpy_helper.to_array(inits[name]).copy()
        if shape[0] == 1 and -1 not in shape:
            shape[0] = -1
            inits[name].CopyFrom(numpy_helper.from_array(shape, name))
    del g.node[:]
    g.node.extend(nodes)
    keep = [t for t in g.initializer if t.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    del g.value_info[:]
    x = g.input[0]
    x.name = INPUT
    x.type.tensor_type.shape.dim[0].dim_param = "batch"
    out = onnx.helper.make_tensor_value_info(OUTPUT, onnx.TensorProto.FLOAT, ["batch", 1152])
    del g.output[:]
    g.output.append(out)
    return m


def parity(dest: Path, n: int = 16) -> dict:
    """Deployed graph (batch 1) vs candidate + head.npz in numpy (batch 1 and batch 8), on real dev images."""
    import onnxruntime as ort

    from compclass.final import fused_scores, log_softmax
    from evaluate import dev_rows, preprocessed

    c = load_candidate(dest)
    m, rows = dev_rows()
    X = preprocessed(c, m, rows[:n])
    head = dict(np.load(DEPLOYED / "head.npz"))
    ort.preload_dlls()
    cuda = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    with gpu_lock():
        s = ort.InferenceSession(str(DEPLOYED / "model.onnx"), providers=cuda)
        ref = np.concatenate([s.run(["probs"], {"input": X[i:i + 1]})[0] for i in range(n)])
        del s
        s = ort.InferenceSession(str(dest / "model.onnx"), providers=cuda)
        f1 = np.concatenate([s.run([OUTPUT], {INPUT: X[i:i + 1]})[0] for i in range(n)])
        f8 = np.concatenate([s.run([OUTPUT], {INPUT: X[i:i + MAX_BATCH]})[0] for i in range(0, n, MAX_BATCH)])
        del s
    probs = np.exp(log_softmax(fused_scores(f8, head) / head["temperature"]))
    return {"n_images": n, "provider": "CUDAExecutionProvider",
            "max_abs_prob_diff_vs_deployed": float(np.abs(probs - ref).max()),
            "argmax_agree_vs_deployed": float((probs.argmax(1) == ref.argmax(1)).mean()),
            "max_abs_embedding_diff_batch1_vs_batch8": float(np.abs(f1 - f8).max()),
            "embedding_abs_max": float(np.abs(f8).max())}


def export_baseline(dest: Path = BASELINE) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    m = cut_trunk(onnx.load(DEPLOYED / "model.onnx", load_external_data=False))
    onnx.save(m, dest / "model.onnx")
    if not (dest / "model.onnx.data").exists():
        subprocess.run(["cp", "--reflink=auto", str(DEPLOYED / "model.onnx.data"), str(dest / "model.onnx.data")],
                       check=True)
    onnx.checker.check_model(str(dest / "model.onnx"))
    pre = json.loads((DEPLOYED / "preprocess.json").read_text())
    keep = ("input_size", "mean", "std", "interpolation", "crop_pct", "resize_short_side")
    (dest / "preprocess.json").write_text(json.dumps(
        {**{k: pre[k] for k in keep}, "zero_shot_fusion": True,
         "built_from": f"{DEPLOYED}/model.onnx trunk (tools/speed export-baseline)"}, indent=1))
    (dest / "runtime.json").write_text(json.dumps({"provider": "CUDAExecutionProvider", "options": {}}, indent=1))
    shutil.copy(DEPLOYED / "labels.json", dest / "labels.json")
    rep = parity(dest)
    (dest / "export_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if rep["argmax_agree_vs_deployed"] < 1 or rep["max_abs_prob_diff_vs_deployed"] > 1e-3:
        raise SystemExit("baseline does not reproduce the deployed graph")
    return rep
