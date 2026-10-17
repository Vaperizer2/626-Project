"""Retrain ResNet18 on the opt-in curated split using the baseline recipe.

This is a single controlled comparison against the existing raw-data
checkpoint. It uses the same optimizer, augmentation, freeze schedule, and
epoch budget as the primary baseline, while selecting the checkpoint only on
the curated validation split. The raw baseline checkpoint is never changed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import nbformat
import torch
from sklearn.metrics import accuracy_score, f1_score


os.environ.setdefault("COMPCLASS_PROFILE", "paper")
os.environ.setdefault("COMPCLASS_FORCE_RETRAIN", "1")


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
    if not os.getenv("COMPCLASS_CURATION_MANIFEST", "").strip():
        raise RuntimeError("Set COMPCLASS_CURATION_MANIFEST to an explicit reviewed manifest.")

    work_dir = Path.cwd().resolve()
    notebook_path = work_dir / "01_primary_comparative.ipynb"
    if not notebook_path.exists():
        notebook_path = work_dir / "notebooks" / "01_primary_comparative.ipynb"
    namespace = load_notebook_namespace(notebook_path)
    standard_build_transforms = namespace["build_transforms"]
    namespace["build_transforms"] = standard_build_transforms

    loaders = namespace["make_loaders"](
        namespace["splits"],
        namespace["class_names"],
        batch_size=namespace["BATCH_SIZE"],
        num_workers=0,
    )
    device = namespace["DEVICE"]
    num_classes = len(namespace["class_names"])

    namespace["set_seed"]()
    model = namespace["build_backbone"]("resnet18", num_classes)
    training = namespace["train_torch_model"](
        model,
        loaders,
        device=device,
        num_classes=num_classes,
        epochs=25,
        lr=1e-3,
        weight_decay=1e-4,
        patience=5,
        freeze_backbone_epochs=1,
    )
    curated_val = evaluate(namespace, training["model"], loaders["val"], device)
    curated_test = evaluate(namespace, training["model"], loaders["test"], device)

    baseline_checkpoint = namespace["WEIGHTS_DIR"] / "primary_resnet18.pt"
    baseline_payload = torch.load(baseline_checkpoint, map_location="cpu")
    baseline_model = namespace["build_backbone"]("resnet18", num_classes)
    baseline_model.load_state_dict(baseline_payload["state_dict"])
    baseline_model = baseline_model.to(device)
    baseline_val = evaluate(namespace, baseline_model, loaders["val"], device)
    baseline_test = evaluate(namespace, baseline_model, loaders["test"], device)

    output = {
        "dataset_fingerprint": namespace["_fingerprint"](namespace["all_samples"]),
        "split_fingerprint": namespace["SPLIT_SIGNATURE"],
        "curation_manifest": os.environ["COMPCLASS_CURATION_MANIFEST"],
        "selection_metric": "curated_val.macro_f1",
        "recipe": {
            "backbone": "resnet18",
            "pretrained": True,
            "epochs": 25,
            "patience": 5,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "freeze_backbone_epochs": 1,
            "augmentation": "primary_standard",
        },
        "baseline_raw_checkpoint": {
            "checkpoint": str(baseline_checkpoint),
            "curated_val": baseline_val,
            "curated_test": baseline_test,
        },
        "curated_retrain": {
            "training": {key: value for key, value in training.items() if key != "model"},
            "curated_val": curated_val,
            "curated_test_after_validation_selection": curated_test,
        },
    }
    metrics_path = namespace["METRICS_DIR"] / "resnet18_curated_matched_retrain.json"
    metrics_path.write_text(json.dumps(output, indent=2))

    checkpoint_path = namespace["WEIGHTS_DIR"] / "resnet18_curated_matched.pt"
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in training["model"].state_dict().items()},
            "metadata": {
                "model": "resnet18",
                "variant": "curated_matched_baseline_recipe",
                "class_names": namespace["class_names"],
                "split_fingerprint": namespace["SPLIT_SIGNATURE"],
                "dataset_fingerprint": output["dataset_fingerprint"],
                "curation_manifest": os.environ["COMPCLASS_CURATION_MANIFEST"],
                "selection_metric": "curated_val.macro_f1",
                "recipe": output["recipe"],
            },
            "training": output["curated_retrain"]["training"],
        },
        checkpoint_path,
    )
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {checkpoint_path}", flush=True)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
