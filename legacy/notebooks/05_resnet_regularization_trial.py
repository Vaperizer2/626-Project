"""Small, validation-only ResNet18 generalisation experiment.

This intentionally reuses the notebook's current helpers so the split,
preprocessing, model, and metric definitions stay comparable to the baseline.
It does not modify the raw dataset or the primary baseline checkpoint.
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
    class_names = namespace["class_names"]
    labels = list(range(len(class_names)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }


def main() -> None:
    work_dir = Path.cwd().resolve()
    notebook_path = work_dir / "01_primary_comparative.ipynb"
    if not notebook_path.exists():
        notebook_path = work_dir / "notebooks" / "01_primary_comparative.ipynb"
    namespace = load_notebook_namespace(notebook_path)
    standard_build_transforms = namespace["build_transforms"]
    transforms = namespace["transforms"]
    image_size = namespace["IMAGE_SIZE"]
    imagenet_mean = namespace["IMAGENET_MEAN"]
    imagenet_std = namespace["IMAGENET_STD"]

    def stronger_build_transforms(train: bool, size=None):
        size = image_size if size is None else size
        if not train:
            return standard_build_transforms(False, size=size)
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size, scale=(0.72, 1.0), ratio=(0.88, 1.14)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(18),
                transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08)),
                transforms.ColorJitter(0.20, 0.20, 0.12),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.15, scale=(0.02, 0.10), ratio=(0.4, 2.5), value="random"),
                transforms.Normalize(imagenet_mean, imagenet_std),
            ]
        )

    split_fingerprint = namespace["SPLIT_SIGNATURE"]
    dataset_fingerprint = namespace["_fingerprint"](namespace["all_samples"])
    device = namespace["DEVICE"]
    trials = [
        {
            "name": "regularized_standard_aug",
            "transform": standard_build_transforms,
            "epochs": 16,
            "patience": 5,
            "lr": 7e-4,
            "weight_decay": 1e-3,
            "freeze_backbone_epochs": 3,
        },
        {
            "name": "regularized_stronger_aug",
            "transform": stronger_build_transforms,
            "epochs": 16,
            "patience": 5,
            "lr": 7e-4,
            "weight_decay": 1e-3,
            "freeze_backbone_epochs": 3,
        },
    ]

    results = []
    best = None
    for config in trials:
        print(f"\n=== {config['name']} ===", flush=True)
        namespace["set_seed"]()
        namespace["build_transforms"] = config["transform"]
        loaders = namespace["make_loaders"](
            namespace["splits"],
            namespace["class_names"],
            batch_size=namespace["BATCH_SIZE"],
            num_workers=0,
        )
        model = namespace["build_backbone"]("resnet18", len(namespace["class_names"]))
        training = namespace["train_torch_model"](
            model,
            loaders,
            device=device,
            num_classes=len(namespace["class_names"]),
            epochs=config["epochs"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            patience=config["patience"],
            freeze_backbone_epochs=config["freeze_backbone_epochs"],
        )
        val_true, val_pred = namespace["predict_loader"](training["model"], loaders["val"], device)
        val_metrics = metrics(namespace, val_true, val_pred)
        print(f"{config['name']} val={val_metrics}", flush=True)
        record = {
            "name": config["name"],
            "settings": {key: value for key, value in config.items() if key != "transform"},
            "training": {key: value for key, value in training.items() if key != "model"},
            "val": val_metrics,
        }
        results.append(record)
        if best is None or val_metrics["macro_f1"] > best["record"]["val"]["macro_f1"]:
            best = {"record": record, "model": training["model"], "loaders": loaders}
        del model, training, loaders
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert best is not None
    best_true, best_pred = namespace["predict_loader"](best["model"], best["loaders"]["test"], device)
    best_test = metrics(namespace, best_true, best_pred)
    best["record"]["test_after_selection"] = best_test

    baseline_checkpoint = namespace["WEIGHTS_DIR"] / "primary_resnet18.pt"
    baseline_payload = torch.load(baseline_checkpoint, map_location="cpu")
    baseline_model = namespace["build_backbone"]("resnet18", len(namespace["class_names"]))
    baseline_model.load_state_dict(baseline_payload["state_dict"])
    baseline_model = baseline_model.to(device)
    namespace["build_transforms"] = standard_build_transforms
    baseline_loaders = namespace["make_loaders"](
        namespace["splits"],
        namespace["class_names"],
        batch_size=namespace["BATCH_SIZE"],
        num_workers=0,
    )
    baseline_true, baseline_pred = namespace["predict_loader"](baseline_model, baseline_loaders["test"], device)
    baseline_test = metrics(namespace, baseline_true, baseline_pred)

    output = {
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": split_fingerprint,
        "class_names": namespace["class_names"],
        "selection_metric": "val.macro_f1",
        "baseline": {"checkpoint": str(baseline_checkpoint), "test": baseline_test},
        "trials": results,
        "selected_trial": best["record"]["name"],
        "selected_test_after_validation_selection": best_test,
    }
    metrics_path = namespace["METRICS_DIR"] / "resnet18_regularization_trials.json"
    metrics_path.write_text(json.dumps(output, indent=2))

    best_path = namespace["WEIGHTS_DIR"] / "resnet18_regularized_best.pt"
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in best["model"].state_dict().items()},
            "metadata": {
                "model": "resnet18",
                "variant": best["record"]["name"],
                "class_names": namespace["class_names"],
                "split_fingerprint": split_fingerprint,
                "dataset_fingerprint": dataset_fingerprint,
                "selection_metric": "val.macro_f1",
                "settings": best["record"]["settings"],
            },
            "training": best["record"]["training"],
        },
        best_path,
    )
    print(f"\nWrote {metrics_path}", flush=True)
    print(f"Wrote {best_path}", flush=True)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
