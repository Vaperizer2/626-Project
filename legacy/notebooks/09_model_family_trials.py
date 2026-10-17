"""Compare two higher-capacity pretrained backbones on the fixed raw split.

Each model uses the primary ResNet18 recipe and is selected on raw validation
macro-F1. The raw test set is evaluated only after both validation decisions.
No existing checkpoint is overwritten.
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
    loaders = namespace["make_loaders"](
        namespace["splits"], namespace["class_names"], batch_size=namespace["BATCH_SIZE"], num_workers=0
    )
    device = namespace["DEVICE"]
    num_classes = len(namespace["class_names"])
    trial_names = ("resnet50", "efficientnet_b0")
    results = []
    selected = None

    for name in trial_names:
        print(f"\n=== {name} ===", flush=True)
        namespace["set_seed"]()
        model = namespace["build_backbone"](name, num_classes)
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
        val_metrics = evaluate(namespace, training["model"], loaders["val"], device)
        print(f"{name} val={val_metrics}", flush=True)
        record = {
            "name": name,
            "settings": {
                "backbone": name,
                "pretrained": True,
                "epochs": 25,
                "patience": 5,
                "lr": 1e-3,
                "weight_decay": 1e-4,
                "freeze_backbone_epochs": 1,
            },
            "training": {key: value for key, value in training.items() if key != "model"},
            "val": val_metrics,
        }
        results.append(record)
        checkpoint_path = namespace["WEIGHTS_DIR"] / f"{name}_full.pt"
        torch.save(
            {
                "state_dict": {key: value.detach().cpu() for key, value in training["model"].state_dict().items()},
                "metadata": {
                    "model": name,
                    "variant": "full_raw_split_baseline_recipe",
                    "class_names": namespace["class_names"],
                    "split_fingerprint": namespace["SPLIT_SIGNATURE"],
                    "dataset_fingerprint": namespace["_fingerprint"](namespace["all_samples"]),
                    "selection_metric": "raw_val.macro_f1",
                    "settings": record["settings"],
                },
                "training": record["training"],
            },
            checkpoint_path,
        )
        record["checkpoint"] = str(checkpoint_path)
        if selected is None or record["val"]["macro_f1"] > selected["val"]["macro_f1"]:
            selected = record
        del model, training
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert selected is not None
    selected_model = namespace["build_backbone"](selected["name"], num_classes)
    selected_payload = torch.load(selected["checkpoint"], map_location="cpu")
    selected_model.load_state_dict(selected_payload["state_dict"])
    selected_model = selected_model.to(device)
    selected_test = evaluate(namespace, selected_model, loaders["test"], device)
    selected["test_after_validation_selection"] = selected_test

    output = {
        "dataset_fingerprint": namespace["_fingerprint"](namespace["all_samples"]),
        "split_fingerprint": namespace["SPLIT_SIGNATURE"],
        "selection_metric": "raw_val.macro_f1",
        "trials": results,
        "selected_model": selected["name"],
        "selected_test_after_validation_selection": selected_test,
    }
    output_path = namespace["METRICS_DIR"] / "model_family_trials.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {output_path}", flush=True)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
