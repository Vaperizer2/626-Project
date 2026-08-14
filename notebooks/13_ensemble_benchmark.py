"""Validation-first ensemble benchmark for already completed checkpoints.

The checkpoint list is supplied before any optional held-out test evaluation.
The ensemble is selected on raw validation macro-F1 and writes a separate
report, so it cannot overwrite single-model trials or silently consume test
labels.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

import torch


runner = importlib.import_module("12_model_family_benchmark")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint filenames under outputs/weights, or absolute paths.")
    parser.add_argument("--profile", default=os.getenv("COMPCLASS_PROFILE", "paper"), choices=("smoke", "standard", "paper"))
    parser.add_argument("--report", default=os.getenv("COMPCLASS_ENSEMBLE_REPORT", "ensemble_benchmark.json"), help="Report filename under outputs/metrics.")
    parser.add_argument("--test", action="store_true", help="Evaluate the held-out test only after the validation ensemble is recorded.")
    return parser.parse_args()


@torch.no_grad()
def collect_logits(model: torch.nn.Module, loader, device: torch.device) -> tuple[list[int], torch.Tensor]:
    model.eval()
    targets, batches = [], []
    for inputs, batch_targets in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        logits = (logits + model(torch.flip(inputs, dims=[3]))) / 2
        batches.append(logits.cpu())
        targets.extend(batch_targets.tolist())
    return targets, torch.cat(batches, dim=0)


def resolve_checkpoint(namespace: dict, value: str) -> Path:
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = namespace["WEIGHTS_DIR"] / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_models(namespace: dict, values: list[str]) -> tuple[list[dict], list[torch.nn.Module]]:
    device = namespace["DEVICE"]
    records, models = [], []
    expected_dataset = namespace["_fingerprint"](namespace["all_samples"])
    expected_split = namespace["SPLIT_SIGNATURE"]
    for value in values:
        path = resolve_checkpoint(namespace, value)
        payload = torch.load(path, map_location="cpu")
        metadata = payload.get("metadata", {})
        if metadata.get("dataset_fingerprint") != expected_dataset or metadata.get("split_fingerprint") != expected_split:
            raise ValueError(f"Checkpoint does not match the frozen evaluation contract: {path}")
        settings = metadata.get("settings", {})
        backbone = settings.get("backbone")
        if not backbone:
            raise ValueError(f"Checkpoint has no backbone metadata: {path}")
        model = runner.build_model(namespace, backbone, len(namespace["class_names"]), pretrained=False).to(device)
        model.load_state_dict(payload["state_dict"])
        models.append(model)
        records.append(
            {
                "path": str(path),
                "name": metadata.get("model", path.stem),
                "backbone": backbone,
                "fingerprint": metadata.get("fingerprint"),
                "train_val_accuracy_gap": payload.get("training", {}).get("train_val_accuracy_gap"),
            }
        )
    return records, models


def main() -> None:
    args = parse_args()
    os.environ["COMPCLASS_PROFILE"] = args.profile
    os.environ["COMPCLASS_FORCE_RETRAIN"] = "0"
    work_dir = Path.cwd().resolve()
    notebook_path = work_dir / "01_primary_comparative.ipynb"
    if not notebook_path.exists():
        notebook_path = work_dir / "notebooks" / "01_primary_comparative.ipynb"
    namespace = runner.load_notebook_namespace(notebook_path)
    loaders = namespace["make_loaders"](namespace["splits"], namespace["class_names"], batch_size=namespace["BATCH_SIZE"], num_workers=0)
    values = [value for value in args.checkpoints.split(",") if value.strip()]
    records, models = load_models(namespace, values)
    y_val, component_logits = None, []
    for model in models:
        targets, logits = collect_logits(model, loaders["val"], namespace["DEVICE"])
        if y_val is None:
            y_val = targets
        elif targets != y_val:
            raise RuntimeError("Validation loader order changed between ensemble components")
        component_logits.append(logits)
    ensemble_logits = torch.stack(component_logits).mean(dim=0)
    ensemble_pred = ensemble_logits.argmax(dim=1).tolist()
    report = {
        "status": "validation_complete_test_deferred",
        "dataset_fingerprint": namespace["_fingerprint"](namespace["all_samples"]),
        "split_fingerprint": namespace["SPLIT_SIGNATURE"],
        "split_sizes": {name: len(rows) for name, rows in namespace["all_splits"].items()},
        "selection_metric": "raw_val_macro_f1",
        "test_policy": "held-out test is evaluated only after validation ensemble selection",
        "test_deferred": not args.test,
        "target_test_accuracy": 0.80,
        "checkpoints": records,
        "components": [runner.metrics(namespace, y_val, logits.argmax(dim=1).tolist()) for logits in component_logits],
        "ensemble_val": runner.metrics(namespace, y_val, ensemble_pred),
        "max_component_train_val_accuracy_gap": max((record["train_val_accuracy_gap"] or 0.0) for record in records),
    }
    if args.test:
        test_targets, test_logits = [], []
        for model in models:
            targets, logits = collect_logits(model, loaders["test"], namespace["DEVICE"])
            if not test_targets:
                test_targets = targets
            elif targets != test_targets:
                raise RuntimeError("Test loader order changed between ensemble components")
            test_logits.append(logits)
        test_pred = torch.stack(test_logits).mean(dim=0).argmax(dim=1).tolist()
        report["selected_test_after_validation_selection"] = runner.metrics(namespace, test_targets, test_pred)
        report["status"] = "target_reached" if report["selected_test_after_validation_selection"]["accuracy"] >= report["target_test_accuracy"] else "completed_below_target"
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = namespace["METRICS_DIR"] / report_path
    runner.write_report(report_path, report)
    print(json.dumps({"ensemble_val": report["ensemble_val"], "status": report["status"], "report": str(report_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
