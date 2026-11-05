"""tools/gen_manifests.py on a small synthetic export (no onnx module needed: the export records its IO)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "gen_manifests.py"
spec = importlib.util.spec_from_file_location("gen_manifests", TOOL)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

DISABLE = "optimization.disable_specified_optimizers"


@pytest.fixture(autouse=True)
def no_onnx(monkeypatch):
    """The fake graph is not ONNX: its IO comes from the export report, as under a python without onnx."""
    monkeypatch.setattr(gen, "onnx", None)


def export(tmp_path: Path, session_config: dict) -> Path:
    d = tmp_path / "m"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"not a real graph")
    (d / "labels.json").write_text(json.dumps(["a", "b"]))
    (d / "preprocess.json").write_text(json.dumps({
        "input_size": 8, "resize_short_side": 9, "interpolation": "bicubic", "scale": 1 / 255,
        "mean": [0.5] * 3, "std": [0.5] * 3, "abstain_threshold": 0.5, "temperature": 0.9}))
    (d / "export_report.json").write_text(json.dumps({"io": {"inputs": [["pixel_values", [1, 3, 8, 8]]],
                                                             "outputs": [["probs", [1, 2]]]}}))
    (d / "runtime.json").write_text(json.dumps({
        "provider": "TensorrtExecutionProvider", "options": {"trt_fp16_enable": True, "trt_engine_cache_path": "c"},
        "graph_optimization_level": "ORT_ENABLE_ALL", "session_config": session_config,
        "provider_order": ["TensorrtExecutionProvider", "CPUExecutionProvider"], "static_batch": 1}))
    return tmp_path


SPEC = dict(dir="m", id="m", graph="model.onnx", external=[], precision="int8_qdq",
            heads=[gen.classify("m", "M", "M", "probs", True)])


def test_session_config_reaches_the_manifest(tmp_path):
    m = gen.build(export(tmp_path, {DISABLE: "MatMulAddFusion;TransposeOptimizer"}), SPEC, False)
    assert m["runtime"]["session_config"] == {DISABLE: "MatMulAddFusion;TransposeOptimizer"}
    assert [p["ep"] for p in m["runtime"]["providers"]] == ["tensorrt", "cpu"]


def test_no_session_config_leaves_the_manifest_as_it_was(tmp_path):
    assert "session_config" not in gen.build(export(tmp_path, {}), SPEC, False)["runtime"]


def test_a_comma_separated_optimizer_list_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="separated by ';'"):
        gen.build(export(tmp_path, {DISABLE: "MatMulAddFusion,TransposeOptimizer"}), SPEC, False)


def test_training_notebook_builds_app_folder_without_optional_graphs(tmp_path, monkeypatch):
    import shutil

    exports = tmp_path / "exports"
    for model in gen.MODELS[:4]:
        d = exports / model["dir"]
        d.mkdir(parents=True)
        for name in (model["graph"], *model["external"]):
            (d / name).write_bytes(b"synthetic graph")
        (d / "labels.json").write_text(json.dumps(["a", "b"]))
        (d / "preprocess.json").write_text(json.dumps({
            "input_size": 8, "resize_short_side": 8, "interpolation": "bicubic", "scale": 1 / 255,
            "mean": [0.5] * 3, "std": [0.5] * 3, "abstain_threshold": 0.5, "temperature": 1.0}))
        (d / "export_report.json").write_text(json.dumps({"io": {
            "inputs": [["input", [1, 3, 8, 8]]],
            "outputs": [[h["output"], [1, 2]] for h in model["heads"]]}}))
        if model["id"] == "fast":
            (d / "runtime.json").write_text(json.dumps({"provider": "CPUExecutionProvider", "options": {}}))

    def tool(path, *args):
        monkeypatch.setattr(sys, "argv", [str(path), *map(str, args)])
        gen.main()

    root = TOOL.parents[1]
    nb = json.loads((root / "notebooks/ITRI626_training.ipynb").read_text())
    cell = next("".join(c["source"]) for c in nb["cells"]
                if c["cell_type"] == "code" and "if BUILD[\"App folder\"]:" in "".join(c["source"]))
    exec(compile(cell, "ITRI626_training:App folder", "exec"), {
        "BUILD": {"App folder": True}, "EXPORTS": exports, "ROOT": root,
        "in_scratch": lambda p: p, "shutil": shutil, "tool": tool})
    index = json.loads((exports / "models.json").read_text())
    assert index["default"] == ["fast", "deployed"]
    assert all((exports / p).is_file() for p in index["manifests"])
    for name in ("mobileclip2_s3", "final_crop100"):
        assert (exports / name / "labels.json").is_file()
        assert not (exports / name / "model.onnx").exists()
