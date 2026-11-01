"""tools/gen_manifests.py on a small synthetic export (no onnx module needed: the export records its IO)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "gen_manifests.py"
spec = importlib.util.spec_from_file_location("gen_manifests", TOOL)
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

DISABLE = "optimization.disable_specified_optimizers"


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
