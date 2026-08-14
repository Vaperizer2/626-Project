"""Resumable validation-first model-family and size benchmark.

The candidate list is fixed before the held-out test is touched. Each trial
selects its own best epoch on validation macro-F1, writes a progress snapshot
after every validation pass, and gets one test evaluation only after the
validation comparison is complete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import nbformat
import torch
from sklearn.metrics import accuracy_score, f1_score


DEFAULT_MODELS = (
    "resnet34",
    "resnet101",
    "densenet121",
    "efficientnet_b2",
    "mobilenet_v3_large",
    "convnext_tiny",
    "swin_t",
    "resnet152",
    "efficientnet_v2_s",
    "convnext_small",
    "swin_v2_t",
    "convnext_base",
    "efficientnet_v2_m",
    "swin_b",
)
MODEL_SETTINGS = {
    "resnet34": {"family": "resnet", "backbone": "resnet34", "lr": 1e-3, "weight_decay": 1e-4, "freeze_backbone_epochs": 1},
    "resnet101": {"family": "resnet", "backbone": "resnet101", "lr": 7e-4, "weight_decay": 1e-4, "freeze_backbone_epochs": 1},
    "densenet121": {"family": "densenet", "backbone": "densenet121", "lr": 1e-3, "weight_decay": 1e-4, "freeze_backbone_epochs": 1},
    "efficientnet_b2": {"family": "efficientnet", "backbone": "efficientnet_b2", "lr": 1e-3, "weight_decay": 1e-4, "freeze_backbone_epochs": 1},
    "mobilenet_v3_large": {"family": "mobilenet", "backbone": "mobilenet_v3_large", "lr": 1e-3, "weight_decay": 1e-4, "freeze_backbone_epochs": 1},
    "convnext_tiny": {"family": "convnext", "backbone": "convnext_tiny", "lr": 5e-4, "weight_decay": 1e-4, "freeze_backbone_epochs": 1},
    "swin_t": {"family": "swin", "backbone": "swin_t", "lr": 5e-4, "weight_decay": 1e-4, "freeze_backbone_epochs": 1},
    "resnet152": {"family": "resnet", "backbone": "resnet152", "lr": 5e-4, "weight_decay": 1e-4, "freeze_backbone_epochs": 1},
    "efficientnet_v2_s": {"family": "efficientnet_v2", "backbone": "efficientnet_v2_s", "lr": 8e-4, "weight_decay": 1e-4, "freeze_backbone_epochs": 1},
    "convnext_small": {"family": "convnext", "backbone": "convnext_small", "lr": 4e-4, "weight_decay": 1e-4, "freeze_backbone_epochs": 1},
    "swin_v2_t": {"family": "swin_v2", "backbone": "swin_v2_t", "lr": 5e-4, "weight_decay": 1e-4, "freeze_backbone_epochs": 1},
    "convnext_base": {"family": "convnext", "backbone": "convnext_base", "lr": 3e-4, "weight_decay": 1e-4, "freeze_backbone_epochs": 1},
    "efficientnet_v2_m": {"family": "efficientnet_v2", "backbone": "efficientnet_v2_m", "lr": 6e-4, "weight_decay": 1e-4, "freeze_backbone_epochs": 1},
    "swin_b": {"family": "swin", "backbone": "swin_b", "lr": 4e-4, "weight_decay": 1e-4, "freeze_backbone_epochs": 1},
}


def load_notebook_namespace(notebook_path: Path) -> dict:
    notebook = nbformat.read(notebook_path, as_version=4)
    namespace: dict = {"__name__": "__main__"}
    for index in (4, 5, 7):
        source = "".join(notebook.cells[index].source)
        exec(compile(source, f"{notebook_path}:cell-{index}", "exec"), namespace)
    return namespace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=os.getenv("COMPCLASS_BENCHMARK_MODELS", ",".join(DEFAULT_MODELS)))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("COMPCLASS_BENCHMARK_EPOCHS", "15")))
    parser.add_argument("--patience", type=int, default=int(os.getenv("COMPCLASS_BENCHMARK_PATIENCE", "4")))
    parser.add_argument("--profile", default=os.getenv("COMPCLASS_PROFILE", "paper"), choices=("smoke", "standard", "paper"))
    parser.add_argument("--max-train-val-gap", type=float, default=float(os.getenv("COMPCLASS_MAX_TRAIN_VAL_GAP", "0.30")))
    parser.add_argument("--report", default=os.getenv("COMPCLASS_BENCHMARK_REPORT", "model_family_benchmark.json"), help="Report filename under outputs/metrics.")
    parser.add_argument("--defer-test", action="store_true", help="Finish validation selection without touching the held-out test split.")
    parser.add_argument("--recipe", default=os.getenv("COMPCLASS_TRAINING_VARIANT", "robust-v2-tta"), choices=("robust-v2-tta", "strong-regularized", "accuracy-regularized"), help="Training-side recipe recorded in the session manifest and checkpoint metadata.")
    parser.add_argument("--artifact-prefix", default=os.getenv("COMPCLASS_BENCHMARK_ARTIFACT_PREFIX", "family_benchmark"), help="Prefix for checkpoint filenames so separate phases do not overwrite earlier artifacts.")
    parser.add_argument("--training-exclusions", default=os.getenv("COMPCLASS_TRAINING_EXCLUSIONS", ""), help="Optional reviewed JSON manifest; only entries marked train-* are removed from the training loader. Validation and test remain untouched.")
    return parser.parse_args()


def cpu_state(model: torch.nn.Module) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def metrics(namespace: dict, y_true, y_pred) -> dict:
    labels = list(range(len(namespace["class_names"])))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }


def trial_fingerprint(namespace: dict, name: str, settings: dict) -> str:
    payload = {
        "model": name,
        "settings": settings,
        "dataset_fingerprint": namespace["_fingerprint"](namespace["all_samples"]),
        "split_fingerprint": namespace["SPLIT_SIGNATURE"],
        "class_names": namespace["class_names"],
        "training_variant": namespace["TRAINING_VARIANT"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def apply_training_exclusions(loaders: dict, manifest_path: Path) -> dict:
    """Apply only reviewed training exclusions and return an auditable summary."""
    raw = manifest_path.read_bytes()
    payload = json.loads(raw)
    candidates = []
    for row in payload.get("exclude", []):
        review_id = str(row.get("review_id", ""))
        split = str(row.get("split", ""))
        if review_id.startswith("train-") or split == "train":
            relative_path = str(row.get("relative_path", "")).replace("\\", "/").lstrip("/")
            if relative_path:
                candidates.append({"review_id": review_id, "relative_path": relative_path, "reason": row.get("reason", "")})
    by_path = {row["relative_path"]: row for row in candidates}
    original = list(loaders["train"].dataset.samples)
    retained, applied, unmatched = [], [], set(by_path)
    for sample in original:
        sample_path = Path(sample.path).as_posix()
        match = next((row for relative_path, row in by_path.items() if sample_path == relative_path or sample_path.endswith("/" + relative_path)), None)
        if match is None:
            retained.append(sample)
            continue
        applied.append(match)
        unmatched.discard(match["relative_path"])
    loaders["train"].dataset.samples = retained
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "requested_count": len(candidates),
        "applied_count": len(applied),
        "applied_review_ids": [row["review_id"] for row in applied],
        "unmatched_review_ids": [by_path[path]["review_id"] for path in sorted(unmatched)],
        "original_train_size": len(original),
        "remaining_train_size": len(retained),
    }


def build_model(namespace: dict, backbone: str, num_classes: int, *, pretrained: bool) -> torch.nn.Module:
    previous = namespace["USE_PRETRAINED"]
    namespace["USE_PRETRAINED"] = pretrained
    try:
        return namespace["build_backbone"](backbone, num_classes)
    finally:
        namespace["USE_PRETRAINED"] = previous


def save_progress(path: Path, state_dict: dict, metadata: dict, history: list[dict], *, complete: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict, "metadata": {**metadata, "complete": complete}, "training": {"history": history}}, path)


def train_trial(namespace: dict, name: str, settings: dict, loaders: dict, checkpoint: Path, progress: Path, fingerprint: str, *, epochs: int, patience: int) -> dict:
    device = namespace["DEVICE"]
    num_classes = len(namespace["class_names"])
    namespace["set_seed"](namespace["SEED"] + list(MODEL_SETTINGS).index(name))
    model = build_model(namespace, settings["backbone"], num_classes, pretrained=True).to(device)
    train_targets = [loaders["train"].dataset.class_to_idx[sample.label] for sample in loaders["train"].dataset.samples]
    counts = torch.bincount(torch.tensor(train_targets), minlength=num_classes).float()
    class_weight_power = float(settings.get("class_weight_power", 0.5))
    class_weights = None if class_weight_power == 0 else (counts.mean() / counts).pow(class_weight_power).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=float(settings.get("label_smoothing", 0.08)))
    if settings["freeze_backbone_epochs"]:
        namespace["_set_head_or_backbone"](model, False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=settings["lr"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    history, best_state, best_val, stale = [], None, -1.0, 0
    started = time.perf_counter()
    metadata = {
        "fingerprint": fingerprint,
        "model": name,
        "family": settings["family"],
        "settings": {**settings, "epochs": epochs, "patience": patience, "selection": "raw_val_macro_f1"},
        "class_names": namespace["class_names"],
        "image_size": namespace["IMAGE_SIZE"],
        "split_fingerprint": namespace["SPLIT_SIGNATURE"],
        "dataset_fingerprint": namespace["_fingerprint"](namespace["all_samples"]),
    }
    for epoch in range(1, epochs + 1):
        if settings["freeze_backbone_epochs"] and epoch == settings["freeze_backbone_epochs"] + 1:
            namespace["_set_head_or_backbone"](model, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"] * 0.1, weight_decay=settings["weight_decay"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        model.train()
        loss_total, correct, total, batches = 0.0, 0, 0, 0
        for inputs, targets in namespace["tqdm"](loaders["train"], desc=f"{name} epoch {epoch}/{epochs}", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item())
            correct += int(logits.argmax(dim=1).eq(targets).sum())
            total += len(targets)
            batches += 1
        y_val, y_pred = namespace["predict_loader"](model, loaders["val"], device)
        val_metrics = metrics(namespace, y_val, y_pred)
        record = {
            "epoch": epoch,
            "train_loss": loss_total / max(batches, 1),
            "train_accuracy": correct / max(total, 1),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(record)
        scheduler.step(record["val_macro_f1"])
        improved = record["val_macro_f1"] > best_val + 1e-4
        if improved:
            best_val = record["val_macro_f1"]
            best_state = cpu_state(model)
            stale = 0
        else:
            stale += 1
        if best_state is None:
            best_state = cpu_state(model)
        save_progress(progress, best_state, metadata, history, complete=False)
        print(
            f"{name} epoch={epoch:02d} loss={record['train_loss']:.4f} "
            f"train_acc={record['train_accuracy']:.4f} val_acc={record['val_accuracy']:.4f} "
            f"val_macro_f1={record['val_macro_f1']:.4f}",
            flush=True,
        )
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    best_record = max(history, key=lambda row: row["val_macro_f1"])
    training = {
        "history": history,
        "params": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best_record["epoch"],
        "best_val_macro_f1": best_record["val_macro_f1"],
        "best_val_accuracy": best_record["val_accuracy"],
        "best_train_accuracy": best_record["train_accuracy"],
        "train_val_accuracy_gap": best_record["train_accuracy"] - best_record["val_accuracy"],
        "wall_time_seconds": time.perf_counter() - started,
    }
    save_progress(checkpoint, cpu_state(model), metadata, history, complete=True)
    if progress.exists():
        progress.unlink()
    return {"name": name, "settings": metadata["settings"], "params": training["params"], "training": training, "val": {"accuracy": best_record["val_accuracy"], "macro_f1": best_record["val_macro_f1"], "weighted_f1": best_record["val_weighted_f1"]}, "checkpoint": str(checkpoint)}


def load_completed(namespace: dict, name: str, checkpoint: Path, fingerprint: str) -> dict | None:
    if not checkpoint.exists():
        return None
    payload = torch.load(checkpoint, map_location="cpu")
    if payload.get("metadata", {}).get("fingerprint") != fingerprint or not payload.get("metadata", {}).get("complete"):
        return None
    training = payload.get("training", {})
    history = training.get("history", [])
    best = max(history, key=lambda row: row["val_macro_f1"])
    return {
        "name": name,
        "settings": payload["metadata"].get("settings", {}),
        "params": int(training.get("params", sum(value.numel() for key, value in payload["state_dict"].items() if not key.endswith("num_batches_tracked")))),
        "training": {
            **training,
            "best_epoch": best["epoch"],
            "best_val_macro_f1": best["val_macro_f1"],
            "best_val_accuracy": best["val_accuracy"],
            "best_train_accuracy": best["train_accuracy"],
            "train_val_accuracy_gap": best["train_accuracy"] - best["val_accuracy"],
        },
        "val": {"accuracy": best["val_accuracy"], "macro_f1": best["val_macro_f1"], "weighted_f1": best.get("val_weighted_f1")},
        "checkpoint": str(checkpoint),
        "loaded": True,
    }


def evaluate_checkpoint(namespace: dict, record: dict, loaders: dict, split: str) -> dict:
    model = build_model(namespace, record["settings"]["backbone"], len(namespace["class_names"]), pretrained=False)
    payload = torch.load(record["checkpoint"], map_location="cpu")
    model.load_state_dict(payload["state_dict"])
    model = model.to(namespace["DEVICE"])
    y_true, y_pred = namespace["predict_loader"](model, loaders[split], namespace["DEVICE"])
    return metrics(namespace, y_true, y_pred)


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    os.environ["COMPCLASS_PROFILE"] = args.profile
    os.environ["COMPCLASS_TRAINING_VARIANT"] = args.recipe
    os.environ["COMPCLASS_FORCE_RETRAIN"] = "0"
    os.environ.pop("COMPCLASS_CURATION_MANIFEST", None)
    work_dir = Path.cwd().resolve()
    notebook_path = work_dir / "01_primary_comparative.ipynb"
    if not notebook_path.exists():
        notebook_path = work_dir / "notebooks" / "01_primary_comparative.ipynb"
    namespace = load_notebook_namespace(notebook_path)
    loaders = namespace["make_loaders"](namespace["splits"], namespace["class_names"], batch_size=namespace["BATCH_SIZE"], num_workers=0)
    exclusion_info = None
    if args.training_exclusions.strip():
        requested_path = Path(args.training_exclusions).expanduser()
        candidates = [requested_path]
        if not requested_path.is_absolute():
            candidates.extend((work_dir / requested_path, notebook_path.parent / requested_path))
        manifest_path = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if manifest_path is None:
            raise FileNotFoundError(f"Training exclusion manifest not found: {args.training_exclusions}")
        exclusion_info = apply_training_exclusions(loaders, manifest_path)
        print(json.dumps({"training_exclusions": exclusion_info}, indent=2), flush=True)
    names = tuple(name.strip() for name in args.models.split(",") if name.strip())
    unknown = sorted(set(names) - set(MODEL_SETTINGS))
    if unknown:
        raise ValueError(f"Unknown benchmark models: {unknown}; choose from {sorted(MODEL_SETTINGS)}")
    report_name = Path(args.report)
    report_path = report_name if report_name.is_absolute() else namespace["METRICS_DIR"] / report_name
    report = {
        "status": "running",
        "dataset_fingerprint": namespace["_fingerprint"](namespace["all_samples"]),
        "split_fingerprint": namespace["SPLIT_SIGNATURE"],
        "split_sizes": {name: len(rows) for name, rows in namespace["all_splits"].items()},
        "selection_metric": "raw_val_macro_f1",
        "test_policy": "held-out test is deferred until the final validation selection" if args.defer_test else "held-out test is evaluated only after all requested validation decisions",
        "test_deferred": args.defer_test,
        "target_test_accuracy": 0.80,
        "max_train_val_accuracy_gap": args.max_train_val_gap,
        "training_variant": args.recipe,
        "artifact_prefix": args.artifact_prefix,
        "epochs": args.epochs,
        "patience": args.patience,
        "requested_models": list(names),
        "training_exclusions": exclusion_info,
        "trials": [],
    }
    existing = {}
    if report_path.exists():
        try:
            old = json.loads(report_path.read_text())
            if old.get("dataset_fingerprint") == report["dataset_fingerprint"] and old.get("split_fingerprint") == report["split_fingerprint"]:
                existing = {trial["name"]: trial for trial in old.get("trials", [])}
        except json.JSONDecodeError:
            pass
    for name in names:
        settings = {
            **MODEL_SETTINGS[name],
            "epochs": args.epochs,
            "patience": args.patience,
            "selection": "raw_val_macro_f1",
            "training_variant": args.recipe,
            "label_smoothing": 0.08,
            "class_weight_power": 0.5,
            "training_exclusions_sha256": exclusion_info["manifest_sha256"] if exclusion_info else None,
            "training_exclusions_applied_count": exclusion_info["applied_count"] if exclusion_info else 0,
        }
        if args.recipe == "strong-regularized":
            settings["weight_decay"] *= 5
            settings["freeze_backbone_epochs"] = max(settings["freeze_backbone_epochs"], 2)
            settings["label_smoothing"] = 0.12
        elif args.recipe == "accuracy-regularized":
            settings["weight_decay"] *= 5
            settings["freeze_backbone_epochs"] = max(settings["freeze_backbone_epochs"], 2)
            settings["label_smoothing"] = 0.05
            settings["class_weight_power"] = 0.0
        fingerprint = trial_fingerprint(namespace, name, settings)
        checkpoint = namespace["WEIGHTS_DIR"] / f"{args.artifact_prefix}_{name}.pt"
        progress = namespace["WEIGHTS_DIR"] / f"{args.artifact_prefix}_{name}.progress.pt"
        record = load_completed(namespace, name, checkpoint, fingerprint)
        if record is None:
            print(f"\n=== {name} ===", flush=True)
            record = train_trial(namespace, name, settings, loaders, checkpoint, progress, fingerprint, epochs=args.epochs, patience=args.patience)
        else:
            print(f"Loaded completed {checkpoint.name}", flush=True)
        record["fingerprint"] = fingerprint
        record["overfit_flag"] = record["training"]["train_val_accuracy_gap"] > args.max_train_val_gap
        report["trials"].append(record)
        report["trials"] = sorted(report["trials"], key=lambda item: names.index(item["name"]))
        write_report(report_path, report)
        print(json.dumps({"model": name, "val": record["val"], "gap": record["training"]["train_val_accuracy_gap"], "overfit_flag": record["overfit_flag"]}, indent=2), flush=True)

    eligible = [trial for trial in report["trials"] if not trial["overfit_flag"]]
    if not eligible:
        eligible = report["trials"]
        report["selection_warning"] = "All completed trials exceeded the train/validation gap guard; selected highest validation macro-F1 for diagnosis only."
    selected = max(eligible, key=lambda item: item["val"]["macro_f1"])
    report["selected_model"] = selected["name"]
    if args.defer_test:
        report["selected_test_after_validation_selection"] = None
        report["status"] = "validation_complete_test_deferred"
    else:
        report["selected_test_after_validation_selection"] = evaluate_checkpoint(namespace, selected, loaders, "test")
        report["status"] = "target_reached" if report["selected_test_after_validation_selection"]["accuracy"] >= report["target_test_accuracy"] else "completed_below_target"
    write_report(report_path, report)
    print(json.dumps({"selected_model": report["selected_model"], "selected_test_after_validation_selection": report["selected_test_after_validation_selection"], "status": report["status"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
