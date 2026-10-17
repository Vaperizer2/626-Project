"""Evaluate the untouched baseline on the original split and reviewed raw test.

The curation manifest was inspected after the original run, so this report
does not call the filtered score an unbiased benchmark. It is a diagnostic:
the split stays fixed and only reviewed raw-test rows are removed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import nbformat
import torch
from sklearn.metrics import accuracy_score, f1_score


os.environ.pop("COMPCLASS_CURATION_MANIFEST", None)
os.environ["COMPCLASS_FORCE_RETRAIN"] = "1"
os.environ.setdefault("COMPCLASS_PROFILE", "paper")


def load_notebook_namespace(notebook_path: Path) -> dict:
    notebook = nbformat.read(notebook_path, as_version=4)
    namespace: dict = {"__name__": "__main__"}
    for index in (4, 5, 7):
        source = "".join(notebook.cells[index].source)
        exec(compile(source, f"{notebook_path}:cell-{index}", "exec"), namespace)
    return namespace


def metrics(namespace: dict, y_true, y_pred) -> dict:
    labels = list(range(len(namespace["class_names"])))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }


def evaluate(namespace: dict, model, loader, device) -> dict:
    y_true, y_pred = namespace["predict_loader"](model, loader, device)
    return metrics(namespace, y_true, y_pred)


def main() -> None:
    work_dir = Path.cwd().resolve()
    notebook_path = work_dir / "01_primary_comparative.ipynb"
    if not notebook_path.exists():
        notebook_path = work_dir / "notebooks" / "01_primary_comparative.ipynb"
    namespace = load_notebook_namespace(notebook_path)
    manifest_path = namespace["DATA_DIR"] / "curation" / "reviewed_exclusions_v1.json"
    payload = json.loads(manifest_path.read_text())
    excluded_paths = set()
    for row in payload.get("exclude", []):
        value = row["relative_path"] if isinstance(row, dict) else row
        candidate = Path(value)
        if candidate.is_absolute():
            excluded_paths.add(str(candidate.resolve()))
        else:
            excluded_paths.add(str((namespace["dataset_root"] / candidate).resolve()))
            excluded_paths.add(str((namespace["WORK_DIR"] / candidate).resolve()))

    raw_test = namespace["splits"]["test"]
    retained_test = [sample for sample in raw_test if str(Path(sample.path).resolve()) not in excluded_paths]
    removed_test = [sample for sample in raw_test if sample not in retained_test]
    filtered_splits = dict(namespace["splits"])
    filtered_splits["test"] = retained_test

    namespace["build_transforms"] = namespace["build_transforms"]
    full_loaders = namespace["make_loaders"](
        namespace["splits"], namespace["class_names"], batch_size=namespace["BATCH_SIZE"], num_workers=0
    )
    filtered_loaders = namespace["make_loaders"](
        filtered_splits, namespace["class_names"], batch_size=namespace["BATCH_SIZE"], num_workers=0
    )
    checkpoint = namespace["WEIGHTS_DIR"] / "primary_resnet18.pt"
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    model = namespace["build_backbone"]("resnet18", len(namespace["class_names"]))
    model.load_state_dict(checkpoint_payload["state_dict"])
    model = model.to(namespace["DEVICE"])

    output = {
        "checkpoint": str(checkpoint),
        "checkpoint_split_fingerprint": checkpoint_payload.get("metadata", {}).get("split_fingerprint"),
        "evaluated_split_fingerprint": namespace["SPLIT_SIGNATURE"],
        "curation_manifest": str(manifest_path),
        "raw_counts": {
            "train": len(namespace["splits"]["train"]),
            "val": len(namespace["splits"]["val"]),
            "test": len(raw_test),
        },
        "reviewed_exclusions_in_raw_test": len(removed_test),
        "raw_test": evaluate(namespace, model, full_loaders["test"], namespace["DEVICE"]),
        "raw_test_after_reviewed_exclusions": evaluate(
            namespace, model, filtered_loaders["test"], namespace["DEVICE"]
        ),
    }
    output_path = namespace["METRICS_DIR"] / "resnet18_raw_test_curation_eval.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {output_path}", flush=True)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
