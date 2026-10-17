"""Compare per-class errors for the selected ResNet50 and scratch CNN.

This is an evaluation-only audit on the frozen raw split. It uses the same
deterministic preprocessing and horizontal-flip test-time averaging as the
training notebooks; it does not retrain or choose a model from the test set.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


os.environ.pop("COMPCLASS_CURATION_MANIFEST", None)
os.environ["COMPCLASS_FORCE_RETRAIN"] = "0"
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
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(channels[-1], hidden or num_classes),
        )
        if hidden:
            self.classifier.extend(
                [
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(head_dropout),
                    torch.nn.Linear(hidden, num_classes),
                ]
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = torch.nn.functional.adaptive_avg_pool2d(self.features(inputs), (1, 1))
        return self.classifier(pooled)


def make_deterministic_loaders(namespace: dict) -> dict:
    class_to_idx = {name: index for index, name in enumerate(namespace["class_names"])}
    loaders = {}
    for split_name, samples in namespace["all_splits"].items():
        dataset = namespace["ComponentDataset"](
            samples,
            class_to_idx,
            transform=namespace["build_transforms"](train=False),
        )
        loaders[split_name] = namespace["DataLoader"](
            dataset,
            batch_size=namespace["BATCH_SIZE"],
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders


def load_models(namespace: dict) -> dict:
    num_classes = len(namespace["class_names"])
    # Checkpoint loading needs the architecture only; avoid a weights download.
    namespace["USE_PRETRAINED"] = False
    resnet50 = namespace["build_backbone"]("resnet50", num_classes)
    scratch = ScratchCNN(num_classes, width=32, hidden=0, dropout=0.40, head_dropout=0.0)
    models = {"resnet50": (resnet50, "resnet50_full.pt"), "scratch_cnn_width32": (scratch, "sized_cnn_width32.pt")}
    loaded = {}
    for name, (model, filename) in models.items():
        checkpoint = namespace["WEIGHTS_DIR"] / filename
        payload = torch.load(checkpoint, map_location="cpu")
        metadata = payload.get("metadata", {})
        if metadata.get("split_fingerprint") != namespace["SPLIT_SIGNATURE"]:
            raise RuntimeError(f"{name} checkpoint has a different split fingerprint")
        if metadata.get("class_names") != namespace["class_names"]:
            raise RuntimeError(f"{name} checkpoint has a different class order")
        model.load_state_dict(payload["state_dict"])
        loaded[name] = model.to(namespace["DEVICE"]).eval()
    return loaded


@torch.no_grad()
def predict_split(model, loader, namespace: dict, split_name: str) -> list[dict]:
    class_names = namespace["class_names"]
    device = namespace["DEVICE"]
    rows = []
    offset = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        flipped_logits = model(torch.flip(inputs, dims=[3]))
        averaged_logits = (logits + flipped_logits) / 2
        probabilities = averaged_logits.softmax(dim=1)
        predictions = probabilities.argmax(dim=1)
        losses = torch.nn.functional.cross_entropy(averaged_logits, targets, reduction="none")
        top_two = probabilities.topk(k=2, dim=1).values
        for index in range(len(targets)):
            sample = loader.dataset.samples[offset + index]
            path = Path(sample.path)
            try:
                relative_path = str(path.relative_to(namespace["WORK_DIR"]))
            except ValueError:
                relative_path = str(path)
            rows.append(
                {
                    "split": split_name,
                    "path": str(path),
                    "relative_path": relative_path,
                    "label": class_names[int(targets[index])],
                    "prediction": class_names[int(predictions[index])],
                    "correct": bool(predictions[index] == targets[index]),
                    "confidence": float(probabilities[index, predictions[index]]),
                    "margin": float(top_two[index, 0] - top_two[index, 1]),
                    "loss": float(losses[index]),
                    "flip_agrees": bool(
                        logits[index].argmax().item() == flipped_logits[index].argmax().item()
                    ),
                }
            )
        offset += len(targets)
    return rows


def summarise(rows: list[dict], class_names: list[str]) -> dict:
    indices = {name: index for index, name in enumerate(class_names)}
    y_true = [indices[row["label"]] for row in rows]
    y_pred = [indices[row["prediction"]] for row in rows]
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    per_class = {}
    for index, name in enumerate(class_names):
        support = int(matrix[index].sum())
        per_class[name] = {
            "support": support,
            "correct": int(matrix[index, index]),
            "errors": support - int(matrix[index, index]),
            "recall": float(matrix[index, index] / support) if support else 0.0,
        }
    confusions = [
        {"true": class_names[row], "predicted": class_names[column], "count": int(matrix[row, column])}
        for row in range(len(class_names))
        for column in range(len(class_names))
        if row != column and matrix[row, column]
    ]
    confusions.sort(key=lambda item: (-item["count"], item["true"], item["predicted"]))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(len(class_names))), zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=list(range(len(class_names))), zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "top_confusions": confusions[:20],
    }


def plot_confusions(summaries: dict, class_names: list[str], path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(18, 16), constrained_layout=True)
    for row_index, split_name in enumerate(("val", "test")):
        for column_index, model_name in enumerate(summaries):
            matrix = np.asarray(summaries[model_name][split_name]["confusion_matrix"], dtype=float)
            row_totals = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(matrix, row_totals, out=np.zeros_like(matrix), where=row_totals != 0)
            axes[row_index, column_index].imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            axes[row_index, column_index].set_title(f"{model_name} · {split_name}")
            axes[row_index, column_index].set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
            axes[row_index, column_index].set_yticks(range(len(class_names)), class_names, fontsize=7)
            axes[row_index, column_index].set_xlabel("predicted")
            axes[row_index, column_index].set_ylabel("true")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    work_dir = Path.cwd().resolve()
    notebook_path = work_dir / "01_primary_comparative.ipynb"
    if not notebook_path.exists():
        notebook_path = work_dir / "notebooks" / "01_primary_comparative.ipynb"
    namespace = load_notebook_namespace(notebook_path)
    loaders = make_deterministic_loaders(namespace)
    models = load_models(namespace)
    all_rows, summaries = [], {}
    by_model_split = {}
    for model_name, model in models.items():
        summaries[model_name] = {}
        for split_name, loader in loaders.items():
            rows = predict_split(model, loader, namespace, split_name)
            by_model_split[(model_name, split_name)] = rows
            all_rows.extend([{**row, "model": model_name} for row in rows])
            summaries[model_name][split_name] = summarise(rows, namespace["class_names"])

    comparison = {}
    for split_name in ("val", "test"):
        left = {row["relative_path"]: row for row in by_model_split[("resnet50", split_name)]}
        right = {row["relative_path"]: row for row in by_model_split[("scratch_cnn_width32", split_name)]}
        comparison[split_name] = {
            "resnet50_only_correct": sum(left[path]["correct"] and not right[path]["correct"] for path in left),
            "scratch_only_correct": sum(right[path]["correct"] and not left[path]["correct"] for path in left),
            "both_correct": sum(left[path]["correct"] and right[path]["correct"] for path in left),
            "both_wrong": sum(not left[path]["correct"] and not right[path]["correct"] for path in left),
        }

    output = {
        "dataset_fingerprint": namespace["_fingerprint"](namespace["all_samples"]),
        "split_fingerprint": namespace["SPLIT_SIGNATURE"],
        "selection_metric": "raw_val.macro_f1",
        "prediction_recipe": "deterministic resize/normalize + horizontal-flip TTA",
        "models": summaries,
        "comparison": comparison,
    }
    metrics_path = namespace["METRICS_DIR"] / "model_error_audit.json"
    metrics_path.write_text(json.dumps(output, indent=2))
    rows_path = namespace["METRICS_DIR"] / "model_error_audit.csv"
    with rows_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "split", "path", "relative_path", "label", "prediction", "correct", "confidence", "margin", "loss", "flip_agrees"])
        writer.writeheader()
        writer.writerows(all_rows)
    plot_confusions(summaries, namespace["class_names"], namespace["FIGURES_DIR"] / "model_error_audit_confusion.png")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {rows_path}")
    print(json.dumps({name: {split: value for split, value in model.items()} for name, model in summaries.items()}, indent=2))


if __name__ == "__main__":
    main()
