"""Import an existing ``compclass.train --test`` run into the ITRI626 course artefacts without retraining.

    uv run python tools/itri626_import_run.py --name A_scratch_cnn \
        --run-dir runs/win/scratch_cnn-b0afa2aa4d60/fold0 --results results/runs_win.jsonl \
        --sets train.epochs=200 train.patience=40

Writes exactly what ``scripts/itri626_experiments.py --steps train_A`` would have written for that run:
``results/itri626/runs/<name>.json`` (run record, config, history, preprocess, cli, plus the machine it ran on)
and ``results/itri626/preds/<name>_{val,test}.npz``; appends the run record to ``results/itri626/runs.jsonl``.
Run ``scripts/itri626_experiments.py --steps collect`` afterwards.

Why this exists: the sealed test split may be scored once per final model. The 200-epoch scratch-CNN recipe took
its one look in ``runs/win/.../fold0`` (EXPERIMENT_LOG.md, 2026-11-04); re-running ``train_A`` would train a new
model and spend a second look. The run's own ``test_preds.npz`` is copied instead.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from compclass.data import get_fold, get_test, load_manifest  # noqa: E402

OUT = ROOT / "results" / "itri626"
KEEP = ("name", "model", "config_hash", "fold", "params", "trainable_params", "input_size", "epochs_run",
        "best_epoch", "train_time_s", "train_img_per_s", "peak_vram_mb", "effective_batch", "amp", "run_dir")


def machine() -> dict:
    import importlib.metadata as md

    def ver(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return None

    env = {"python": platform.python_version(), "platform": platform.platform(),
           "packages": {p: ver(p) for p in ("torch", "torchvision", "timm", "numpy", "scikit-learn")},
           "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
           "cpu": platform.processor(), "cpu_logical_cores": os.cpu_count()}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env["gpu"] = {"name": p.name, "memory_gib": round(p.total_memory / 2**30, 2),
                      "compute_capability": f"{p.major}.{p.minor}"}
        try:
            env["nvidia_driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                                  capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    return env


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="course model key, e.g. A_scratch_cnn")
    ap.add_argument("--run-dir", required=True, help="fold directory holding best.pt, val_preds.npz, test_preds.npz")
    ap.add_argument("--results", required=True, help="runs .jsonl that holds this run's record (with history)")
    ap.add_argument("--config", default="configs/scratch_cnn.yaml", help="config file named in the recorded cli")
    ap.add_argument("--sets", nargs="*", default=[], help="--set overrides that reproduce the run")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    recs = [json.loads(ln) for ln in Path(args.results).read_text().splitlines() if ln.strip()]
    recs = [r for r in recs if r["config_hash"] == ck["config_hash"] and r["fold"] == ck["fold"]
            and Path(r["run_dir"]).as_posix() == run_dir.as_posix()]
    if not recs:
        raise SystemExit("no record in --results matches the checkpoint's config hash, fold and run_dir")
    rec = recs[-1]  # the latest training in this directory is the one whose checkpoint and predictions are on disk
    if rec["best_epoch"] != ck["epoch"]:
        raise SystemExit(f"record best_epoch {rec['best_epoch']} != checkpoint epoch {ck['epoch']}")
    cfg = rec["config"]
    m = load_manifest(cfg["data"]["manifest"])
    if m.fingerprint != ck["manifest_fingerprint"]:
        raise SystemExit("checkpoint manifest fingerprint differs from the current manifest")

    v, t = np.load(run_dir / "val_preds.npz"), np.load(run_dir / "test_preds.npz")
    val_ids = [r["id"] for r in get_fold(m, ck["fold"])[1]]
    test_ids = [r["id"] for r in get_test(m, confirm=True)]  # ids only; the predictions were made already
    if list(v["ids"]) != val_ids or list(t["ids"]) != test_ids:
        raise SystemExit("prediction ids do not match the manifest's validation / test rows in order")
    assert abs(float((v["probs"].argmax(1) == v["y"]).mean()) - rec["accuracy"]) < 1e-9, "val accuracy != record"
    preds = OUT / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    for split, z in (("val", v), ("test", t)):
        np.savez_compressed(preds / f"{args.name}_{split}.npz", ids=np.asarray(z["ids"]), y=np.asarray(z["y"]),
                            probs=np.asarray(z["probs"], dtype=np.float32), classes=np.asarray(m.classes))

    keep = {k: rec[k] for k in KEEP}
    keep["preprocess"] = ck["preprocess"]
    keep.update(cli=f"uv run python -m compclass.train --config {args.config} --fold {ck['fold']} --test --set "
                    + " ".join(args.sets + [f"output.runs_dir={run_dir.parent.parent.as_posix()}",
                                            f"output.results={Path(args.results).as_posix()}"]),
                config=cfg, history=rec["history"], imported_from=args.results, trained_utc=rec["time_utc"],
                machine=machine())
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "runs" / f"{args.name}.json").write_text(json.dumps(keep, indent=1))
    with open(OUT / "runs.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    tm = json.loads((run_dir / "test_metrics.json").read_text())
    print(f"imported {args.name}: val macro-F1 {rec['macro_f1']:.4f}, test acc {tm['accuracy']:.4f} "
          f"macro-F1 {tm['macro_f1']:.4f}, best epoch {rec['best_epoch']} of {rec['epochs_run']}")


if __name__ == "__main__":
    main()
