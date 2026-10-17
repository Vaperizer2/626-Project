"""Fast CPU smoke tests on a synthetic 3-class manifest (no downloads, a few seconds)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from compclass import classical, export, features, train
from compclass.config import config_hash, load_config
from compclass.data import (ImageDataset, build_transforms, compute_fingerprint, get_fold, get_test, load_manifest,
                            make_synthetic_manifest)
from compclass.models import REGISTRY, build_model

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPECTED = {"scratch_cnn", "resnet50", "efficientnet_v2_s", "mobilenetv3_large", "convnext_tiny", "vit_b16", "swin_t",
            "dinov2_vits14", "dinov2_vitb14", "clip_vitb16", "clip_vitl14", "siglip_b16", "siglip2_b16"}
torch.set_num_threads(4)


@pytest.fixture(scope="module")
def manifest_path(tmp_path_factory) -> Path:
    return make_synthetic_manifest(tmp_path_factory.mktemp("synth"), n_classes=3, per_class=15)


def test_manifest_validation(manifest_path, tmp_path):
    m = load_manifest(manifest_path)
    assert m.classes == ["class_a", "class_b", "class_c"] and m.fingerprint == compute_fingerprint(m.rows)
    raw = json.loads(manifest_path.read_text())
    raw["rows"][0]["label"] = "class_b" if raw["rows"][0]["label"] != "class_b" else "class_a"  # silent edit
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="fingerprint"):
        load_manifest(bad)
    raw = json.loads(manifest_path.read_text())
    dev = [r for r in raw["rows"] if r["split"] == "dev"]
    a, b = next((x, y) for x in dev for y in dev if x["fold"] != y["fold"])
    b["cluster"] = a["cluster"]  # a cluster split across two folds = leakage
    bad.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="cluster"):
        load_manifest(bad, verify=False)


def test_folds_and_sealed_test(manifest_path):
    m = load_manifest(manifest_path)
    seen = set()
    for k in range(m.n_folds):
        tr, va = get_fold(m, k)
        assert not {r["id"] for r in tr} & {r["id"] for r in va}
        assert all(r["split"] == "dev" for r in tr + va)
        seen |= {r["id"] for r in va}
    assert seen == {r["id"] for r in m.rows if r["split"] == "dev"}
    with pytest.raises(PermissionError):
        get_test(m)
    assert len(get_test(m, confirm=True)) == 9


def test_config_inheritance():
    cfg = load_config(CONFIGS / "siglip2_b16.yaml", ["train.lr=1e-3", "aug.mixup=0.2"])
    assert cfg["name"] == "siglip2_b16" and cfg["model"]["key"] == "siglip2_b16"
    assert cfg["train"]["layer_decay"] == 0.65 and cfg["train"]["lr"] == 1e-3 and cfg["aug"]["mixup"] == 0.2
    assert config_hash(cfg) != config_hash(load_config(CONFIGS / "siglip2_b16.yaml"))
    for path in CONFIGS.glob("*.yaml"):  # every config resolves to a registered model
        assert load_config(path)["model"]["key"] in REGISTRY


def test_registry_is_complete():
    import timm

    assert EXPECTED <= set(REGISTRY)
    pretrained = set(timm.list_pretrained())
    for key, e in REGISTRY.items():
        assert len(e.doc) > 100, f"{key} needs a family explainer docstring"
        assert e.timm_name is None or e.timm_name in pretrained, e.timm_name


@pytest.mark.parametrize("key", ["scratch_cnn", "mobilenetv3_large"])
def test_build_forward(key):
    model, spec = build_model(key, num_classes=3, pretrained=False)
    x = torch.randn(2, 3, spec.input_size, spec.input_size)
    assert model.eval()(x).shape == (2, 3)
    feat, _ = build_model(key, num_classes=0, pretrained=False)
    assert feat.eval()(x).ndim == 2


def test_transforms_and_dataset(manifest_path):
    m = load_manifest(manifest_path)
    _, spec = build_model("scratch_cnn", 3, pretrained=False)
    aug = load_config(CONFIGS / "base.yaml")["aug"]
    for is_train in (True, False):
        ds = ImageDataset(get_fold(m, 0)[0], m.root, m.class_to_idx, build_transforms(spec, aug, is_train))
        x, y = ds[0]
        assert x.shape == (3, spec.input_size, spec.input_size) and 0 <= y < 3


def test_train_evaluate_export(manifest_path, tmp_path):
    results = tmp_path / "runs.jsonl"
    sets = [f"data.manifest={manifest_path}", "data.num_workers=0", "train.epochs=3", "train.batch_size=8",
            "train.warmup_epochs=1", "train.freeze_epochs=1", "aug.mixup=0.2", f"output.runs_dir={tmp_path / 'runs'}",
            f"output.results={results}", "model.kwargs.width=8", "train.lr=3e-3"]
    cfg = load_config(CONFIGS / "scratch_cnn.yaml", sets)
    rec = train.run(cfg, fold=0)
    assert rec["epochs_run"] == 3 and 0 <= rec["macro_f1"] <= 1
    line = json.loads(results.read_text().splitlines()[-1])
    assert line["fold"] == 0 and line["config_hash"] == config_hash(cfg) and line["params"] > 0
    ckpt = Path(rec["run_dir"]) / "best.pt"
    report = export.export_onnx(ckpt, tmp_path / "export")
    assert report["parity_ok"], report
    pre = json.loads((tmp_path / "export" / "preprocess.json").read_text())
    assert pre["input_size"] == 160 and json.loads((tmp_path / "export" / "labels.json").read_text())[0] == "class_a"


def test_classical_and_probe(manifest_path, tmp_path):
    results = str(tmp_path / "runs.jsonl")
    rec = classical.run_classical(str(manifest_path), [0], results=results)[0]
    assert rec["accuracy"] > 0.6  # coloured squares are easy for a colour histogram
    rec = features.run_probe(str(manifest_path), "scratch_cnn", [0], results=results, pretrained=False,
                             cache_dir=tmp_path / "cache", num_workers=0)[0]
    assert rec["feature_dim"] == 256 and 0 <= rec["macro_f1"] <= 1
