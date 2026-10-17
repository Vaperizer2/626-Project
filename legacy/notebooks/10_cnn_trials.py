"""Run a small, validation-selected scratch-CNN comparison on the raw split.

The trials isolate capacity and regularisation changes in the existing CNN
path. The raw test set is evaluated only for the validation-selected variant.
The width-64 no-hidden-layer variant was intentionally skipped at the
2026-08-10 checkpoint and is not part of this runnable set.
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


class ScratchCNN(torch.nn.Module):
    def __init__(self, num_classes: int, *, width: int, hidden: int, dropout: float, head_dropout: float) -> None:
        super().__init__()
        channels = [width, width * 2, width * 4, width * 6]
        blocks, in_channels = [], 3
        for out_channels in channels:
            blocks.extend(
                [
                    torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    torch.nn.BatchNorm2d(out_channels),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels
        self.features = torch.nn.Sequential(*blocks)
        classifier = [
            torch.nn.Flatten(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(channels[-1], hidden or num_classes),
        ]
        if hidden:
            classifier.extend(
                [
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(head_dropout),
                    torch.nn.Linear(hidden, num_classes),
                ]
            )
        self.classifier = torch.nn.Sequential(*classifier)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.nn.functional.adaptive_avg_pool2d(self.features(inputs), (1, 1)))


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
    trials = (
        {
            "name": "small_cnn_original",
            "width": 64,
            "hidden": 256,
            "dropout": 0.30,
            "head_dropout": 0.20,
            "lr": 1e-3,
            "weight_decay": 1e-4,
        },
        {
            "name": "small_cnn_high_dropout",
            "width": 64,
            "hidden": 256,
            "dropout": 0.50,
            "head_dropout": 0.30,
            "lr": 7e-4,
            "weight_decay": 1e-3,
        },
        {
            "name": "sized_cnn_width32",
            "width": 32,
            "hidden": 0,
            "dropout": 0.40,
            "head_dropout": 0.0,
            "lr": 1e-3,
            "weight_decay": 1e-3,
        },
    )
    skipped_trials = [
        {
            "name": "sized_cnn_width64",
            "status": "skipped",
            "reason": "Intentionally skipped at the 2026-08-10 checkpoint.",
        }
    ]
    results = []
    selected = None

    for settings in trials:
        print(f"\n=== {settings['name']} ===", flush=True)
        namespace["set_seed"]()
        model = ScratchCNN(num_classes, **{key: settings[key] for key in ("width", "hidden", "dropout", "head_dropout")})
        training = namespace["train_torch_model"](
            model,
            loaders,
            device=device,
            num_classes=num_classes,
            epochs=25,
            lr=settings["lr"],
            weight_decay=settings["weight_decay"],
            patience=5,
        )
        val_metrics = evaluate(namespace, training["model"], loaders["val"], device)
        record = {
            "name": settings["name"],
            "settings": {**settings, "epochs": 25, "patience": 5, "selection": "raw_val.macro_f1"},
            "params": int(sum(parameter.numel() for parameter in training["model"].parameters())),
            "training": {key: value for key, value in training.items() if key != "model"},
            "val": val_metrics,
        }
        print(f"{settings['name']} val={val_metrics}", flush=True)
        checkpoint_path = namespace["WEIGHTS_DIR"] / f"{settings['name']}.pt"
        torch.save(
            {
                "state_dict": {key: value.detach().cpu() for key, value in training["model"].state_dict().items()},
                "metadata": {
                    "model": settings["name"],
                    "variant": "scratch_cnn_raw_split_trial",
                    "class_names": namespace["class_names"],
                    "split_fingerprint": namespace["SPLIT_SIGNATURE"],
                    "dataset_fingerprint": namespace["_fingerprint"](namespace["all_samples"]),
                    "settings": record["settings"],
                },
                "training": record["training"],
            },
            checkpoint_path,
        )
        record["checkpoint"] = str(checkpoint_path)
        results.append(record)
        if selected is None or record["val"]["macro_f1"] > selected["val"]["macro_f1"]:
            selected = record
        del model, training
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert selected is not None
    selected_model = ScratchCNN(
        num_classes,
        **{key: selected["settings"][key] for key in ("width", "hidden", "dropout", "head_dropout")},
    )
    payload = torch.load(selected["checkpoint"], map_location="cpu")
    selected_model.load_state_dict(payload["state_dict"])
    selected_test = evaluate(namespace, selected_model.to(device), loaders["test"], device)
    selected["test_after_validation_selection"] = selected_test

    output = {
        "status": "completed_with_requested_variant_skipped",
        "dataset_fingerprint": namespace["_fingerprint"](namespace["all_samples"]),
        "split_fingerprint": namespace["SPLIT_SIGNATURE"],
        "selection_metric": "raw_val.macro_f1",
        "trials": results,
        "skipped_trials": skipped_trials,
        "selected_model": selected["name"],
        "selected_test_after_validation_selection": selected_test,
    }
    output_path = namespace["METRICS_DIR"] / "cnn_trials.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {output_path}", flush=True)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
