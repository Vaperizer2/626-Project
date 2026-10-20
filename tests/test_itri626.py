"""Tests for the ITRI626 additions: learning-curve history, --test outputs, the linear head, the split export."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from compclass import train
from compclass.config import load_config
from compclass.data import make_synthetic_manifest
from compclass.linear_head import select_weight_decay, train_head

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
torch.set_num_threads(4)
HIST_KEYS = {"epoch", "train_loss", "train_ce", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr"}


@pytest.fixture(scope="module")
def manifest_path(tmp_path_factory) -> Path:
    return make_synthetic_manifest(tmp_path_factory.mktemp("synth"), n_classes=3, per_class=15)


def test_history_and_test_outputs(manifest_path, tmp_path):
    sets = [f"data.manifest={manifest_path}", "data.num_workers=0", "train.epochs=2", "train.batch_size=8",
            "train.warmup_epochs=1", f"output.runs_dir={tmp_path / 'runs'}", f"output.results={tmp_path / 'r.jsonl'}",
            "model.kwargs.width=8", "train.lr=3e-3"]
    rec = train.run(load_config(CONFIGS / "scratch_cnn.yaml", sets), fold=0, test=True)
    h = rec["history"]
    assert len(h) == 2 and all(HIST_KEYS <= set(e) for e in h)
    for e in h:
        assert 0 <= e["train_acc"] <= 1 and 0 <= e["val_acc"] <= 1 and e["val_loss"] > 0 and e["train_ce"] > 0
    run_dir = Path(rec["run_dir"])
    met = json.loads((run_dir / "test_metrics.json").read_text())
    p = np.load(run_dir / "test_preds.npz")
    assert met["split"] == "test" and met["n"] == 9 == len(p["ids"]) and p["probs"].shape == (9, 3)
    assert np.allclose(p["probs"].sum(1), 1, atol=1e-4)
    assert {"balanced_accuracy", "macro_precision", "weighted_f1", "per_class", "confusion"} <= set(met)
    assert rec["test"]["accuracy"] == met["accuracy"]


def test_history_with_mixup_has_no_plain_train_ce(manifest_path, tmp_path):
    sets = [f"data.manifest={manifest_path}", "data.num_workers=0", "train.epochs=1", "train.batch_size=8",
            "aug.mixup=0.2", f"output.runs_dir={tmp_path / 'runs'}", f"output.results={tmp_path / 'r.jsonl'}",
            "model.kwargs.width=8"]
    h = train.run(load_config(CONFIGS / "scratch_cnn.yaml", sets), fold=0)["history"]
    assert h[0]["train_ce"] is None and 0 <= h[0]["train_acc"] <= 1


def test_linear_head_learns_and_selects_on_val():
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 3, (4, 16))
    def make(n):
        y = rng.integers(0, 4, n)
        return (centres[y] + rng.normal(0, 1, (n, 16))).astype(np.float32), y
    (Xtr, ytr), (Xva, yva) = make(200), make(80)
    classes = list("abcd")
    r = train_head(Xtr, ytr, Xva, yva, classes, epochs=30, patience=5, batch_size=64)
    assert r.best_val_macro_f1 > 0.9 and r.history[r.best_epoch]["val_macro_f1"] == r.best_val_macro_f1
    assert {"train_loss", "train_ce", "train_acc", "val_loss", "val_acc", "val_macro_f1"} <= set(r.history[0])
    assert r.predict_proba(Xva).shape == (80, 4)
    best, table = select_weight_decay(Xtr, ytr, Xva, yva, classes, grid=(1e-4, 1e-1), epochs=10, patience=3)
    assert sum(t["selected"] for t in table) == 1 and len(table) == 2


def test_export_splits_matches_manifest():
    if not (ROOT / "data" / "v2" / "manifest.json").exists():
        pytest.skip("data/v2 manifest not present")
    spec = importlib.util.spec_from_file_location("export_splits", ROOT / "tools" / "export_splits.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    frames, summary, info = mod.build()
    assert info["sizes"] == {"train": 2010, "val": 502, "test": 418}
    assert (frames["val"]["fold"] == 0).all() and frames["test"]["fold"].isna().all()
    tot = summary[summary["class"] == "TOTAL"].iloc[0]
    assert tot["final_rows"] == 2930 and tot["train"] + tot["val"] + tot["test"] == 2930
