"""Grouped cross-fold benchmark for the registered classifier set.

The default list remains the original 14-model sweep; an explicitly requested
baseline such as ``resnet18`` can be added without changing that default order.
The legacy test split is deliberately excluded.  Five folds are built from
the legacy train+validation pool, keeping exact byte-identical images together
so the fold estimates do not put a known copy on both sides of a split. Each
model writes out-of-fold logits; those predictions are then combined with
pre-specified uniform probability, logit, and vote aggregation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean, pstdev

import torch
from sklearn.model_selection import StratifiedGroupKFold


TOP3_MODELS = ("convnext_tiny", "convnext_base", "convnext_small")
AUGMENTATION_RECIPES = {
    "standard": "robust-v2-tta",
    "strong": "strong-regularized",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", choices=("smoke", "standard", "paper"), default="paper")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--augmentation", choices=tuple(AUGMENTATION_RECIPES), default="strong")
    parser.add_argument("--models", default="")
    parser.add_argument("--max-per-class", type=int, default=0, help="Limit the development pool per class; smoke profile defaults to 2.")
    parser.add_argument("--report", default="cross_fold_augmentation.json")
    parser.add_argument("--artifact-prefix", default="cross_fold")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--no-tta", action="store_false", dest="tta", default=True)
    parser.add_argument("--external-manifest", default="", help="Optional composed candidate_manifest.csv; selected rows are training-only augmentation.")
    parser.add_argument("--external-source", default="electrocom61", help="Source value to accept from --external-manifest.")
    parser.add_argument("--external-split", default="train", help="Split value to accept from --external-manifest.")
    parser.add_argument("--external-min-side", type=int, default=64, help="Minimum image side for external crops; 0 disables the size filter.")
    parser.add_argument("--external-max-per-class", type=int, default=0, help="Optional deterministic cap on external rows per class; 0 keeps all rows.")
    return parser.parse_args()


def resolve_paths() -> tuple[Path, Path]:
    requested = Path(os.getenv("COMPCLASS_WORK_DIR", Path.cwd())).expanduser().resolve()
    if (requested / "01_primary_comparative.ipynb").exists():
        return requested, requested / "01_primary_comparative.ipynb"
    notebook = requested / "notebooks" / "01_primary_comparative.ipynb"
    if notebook.exists():
        return notebook.parent, notebook
    raise FileNotFoundError(f"Could not find 01_primary_comparative.ipynb under {requested}")


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def short_hash(payload: object) -> str:
    return hashlib.sha256(stable_json(payload).encode()).hexdigest()[:16]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    return float(fmean(values)), float(pstdev(values)) if len(values) > 1 else 0.0


def load_external_training_rows(namespace: dict, args: argparse.Namespace, class_names: list[str]) -> tuple[list, dict | None]:
    """Load explicitly selected external rows without touching validation/test.

    External candidates are appended to every fold's training loader. They are
    never passed to ``make_folds`` and therefore cannot become OOF rows.
    """
    if not args.external_manifest:
        return [], None
    manifest_path = Path(args.external_manifest).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"External manifest not found: {manifest_path}")
    if args.external_min_side < 0 or args.external_max_per_class < 0:
        raise ValueError("external size and per-class caps cannot be negative")
    root = manifest_path.parent.parent
    allowed = set(class_names)
    candidates = []
    skipped = Counter()
    from PIL import Image

    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("source") != args.external_source or row.get("split") != args.external_split:
                skipped["source_or_split"] += 1
                continue
            if row.get("status") != "candidate" or row.get("canonical_label") not in allowed:
                skipped["status_or_label"] += 1
                continue
            image_path = (root / row["output_path"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"External candidate is missing: {image_path}")
            with Image.open(image_path) as image:
                width, height = image.size
            if args.external_min_side and min(width, height) < args.external_min_side:
                skipped["min_side"] += 1
                continue
            candidates.append((row["canonical_label"], str(image_path), row.get("source_sha256", "")))

    candidates.sort(key=lambda item: (item[0], item[1]))
    if args.external_max_per_class:
        counts = Counter()
        capped = []
        for label, path, source_hash in candidates:
            if counts[label] >= args.external_max_per_class:
                skipped["per_class_cap"] += 1
                continue
            counts[label] += 1
            capped.append((label, path, source_hash))
        candidates = capped
    rows = [namespace["Sample"](path=path, label=label) for label, path, _ in candidates]
    info = {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(str(manifest_path)),
        "root": str(root),
        "source": args.external_source,
        "split": args.external_split,
        "min_side": args.external_min_side,
        "max_per_class": args.external_max_per_class or None,
        "selected_rows": len(rows),
        "selected_by_class": dict(sorted(Counter(row.label for row in rows).items())),
        "skipped": dict(sorted(skipped.items())),
        "scope": "training-only; appended to every fold train loader",
    }
    print(f"External training augmentation: {len(rows)} rows from {manifest_path}", flush=True)
    return rows, info


def make_folds(rows, *, folds: int, seed: int, class_names: list[str]) -> tuple[list[dict], dict]:
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    groups = [sha256_file(sample.path) for sample in rows]
    labels = [sample.label for sample in rows]
    class_counts = Counter(labels)
    if class_counts and min(class_counts.values()) < folds:
        raise ValueError(
            f"Each class needs at least {folds} development rows for {folds}-fold CV; "
            f"smallest class has {min(class_counts.values())}"
        )
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignments = [-1] * len(rows)
    fold_rows = []
    for fold, (train_indices, val_indices) in enumerate(splitter.split(rows, labels, groups)):
        train_indices = train_indices.tolist()
        val_indices = val_indices.tolist()
        for index in val_indices:
            if assignments[index] != -1:
                raise AssertionError(f"row {index} appeared in multiple validation folds")
            assignments[index] = fold
        train_groups = {groups[index] for index in train_indices}
        val_groups = {groups[index] for index in val_indices}
        if train_groups & val_groups:
            raise AssertionError(f"exact-content group leaked into fold {fold}")
        fold_rows.append(
            {
                "fold": fold,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "train_size": len(train_indices),
                "val_size": len(val_indices),
                "train_class_counts": dict(Counter(labels[index] for index in train_indices)),
                "val_class_counts": dict(Counter(labels[index] for index in val_indices)),
            }
        )
    if any(assignment < 0 for assignment in assignments):
        raise AssertionError("some development rows were not assigned to a fold")
    group_labels = {}
    for group, label in zip(groups, labels, strict=True):
        group_labels.setdefault(group, set()).add(label)
    group_counts = Counter(groups)
    grouping = {
        "method": "StratifiedGroupKFold",
        "group_key": "sha256_file_bytes",
        "folds": folds,
        "seed": seed,
        "rows": len(rows),
        "exact_content_groups": len(group_labels),
        "duplicate_groups": sum(count > 1 for count in group_counts.values()),
        "conflicting_label_groups": sum(len(labels_for_group) > 1 for labels_for_group in group_labels.values()),
        "class_names": class_names,
    }
    return fold_rows, {"groups": groups, "assignments": assignments, "grouping": grouping}


@torch.no_grad()
def collect_logits(model: torch.nn.Module, loader, device: torch.device, *, tta: bool) -> tuple[list[int], torch.Tensor]:
    model.eval()
    targets, batches = [], []
    for inputs, batch_targets in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        if tta:
            logits = (logits + model(torch.flip(inputs, dims=[3]))) / 2
        batches.append(logits.cpu())
        targets.extend(batch_targets.tolist())
    return targets, torch.cat(batches, dim=0)


def load_checkpoint_logits(runner, namespace: dict, checkpoint: Path, loader, *, tta: bool) -> torch.Tensor:
    payload = torch.load(checkpoint, map_location="cpu")
    settings = payload["metadata"]["settings"]
    model = runner.build_model(namespace, settings["backbone"], len(namespace["class_names"]), pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model = model.to(namespace["DEVICE"])
    targets, logits = collect_logits(model, loader, namespace["DEVICE"], tta=tta)
    expected = [namespace["class_names"].index(sample.label) for sample in loader.dataset.samples]
    if targets != expected:
        raise AssertionError("validation loader order changed while collecting out-of-fold logits")
    del model
    gc.collect()
    if namespace["DEVICE"].type == "cuda":
        torch.cuda.empty_cache()
    return logits


def aggregation_predictions(logits: list[torch.Tensor], mode: str) -> torch.Tensor:
    stack = torch.stack(logits)
    if mode == "mean_logits":
        return stack.mean(dim=0).argmax(dim=1)
    if mode == "mean_probability":
        return stack.softmax(dim=2).mean(dim=0).argmax(dim=1)
    if mode == "majority_vote":
        votes = torch.nn.functional.one_hot(stack.argmax(dim=2), num_classes=stack.shape[2]).sum(dim=0)
        return votes.argmax(dim=1)
    raise ValueError(f"Unknown aggregation mode: {mode}")


def aggregation_summary(runner, namespace: dict, model_names: list[str], oof_logits: dict[str, torch.Tensor], y_dev: list[int], fold_rows: list[dict], *, name: str, mode: str) -> dict:
    selected = [oof_logits[model] for model in model_names]
    predictions = aggregation_predictions(selected, mode)
    oof = runner.metrics(namespace, y_dev, predictions.tolist())
    per_fold = []
    for fold in fold_rows:
        indices = fold["val_indices"]
        per_fold.append(
            {
                "fold": fold["fold"],
                **runner.metrics(namespace, [y_dev[index] for index in indices], predictions[indices].tolist()),
            }
        )
    fold_f1_mean, fold_f1_std = mean_std([row["macro_f1"] for row in per_fold])
    return {
        "name": name,
        "mode": mode,
        "models": model_names,
        "fold_metrics": per_fold,
        "fold_macro_f1_mean": fold_f1_mean,
        "fold_macro_f1_std": fold_f1_std,
        "oof": oof,
    }


def recipe_settings(runner, name: str, args: argparse.Namespace, recipe: str) -> dict:
    settings = {
        **runner.MODEL_SETTINGS[name],
        "epochs": args.epochs,
        "patience": args.patience,
        "selection": "fold_val_macro_f1",
        "training_variant": recipe,
        "label_smoothing": 0.08,
        "class_weight_power": 0.5,
    }
    if recipe == "strong-regularized":
        settings["weight_decay"] *= 5
        settings["freeze_backbone_epochs"] = max(settings["freeze_backbone_epochs"], 2)
        settings["label_smoothing"] = 0.12
    return settings


def model_summary(runner, namespace: dict, name: str, record: list[dict], logits: torch.Tensor, y_dev: list[int], fold_rows: list[dict]) -> dict:
    f1_mean, f1_std = mean_std([row["oof_val"]["macro_f1"] for row in record])
    gap_values = [row["training"]["train_val_accuracy_gap"] for row in record]
    gap_mean, gap_std = mean_std(gap_values)
    oof = runner.metrics(namespace, y_dev, logits.argmax(dim=1).tolist())
    return {
        "model": name,
        "family": runner.MODEL_SETTINGS[name]["family"],
        "fold_count": len(fold_rows),
        "fold_macro_f1_mean": f1_mean,
        "fold_macro_f1_std": f1_std,
        "fold_accuracy_mean": float(fmean([row["oof_val"]["accuracy"] for row in record])),
        "mean_train_val_accuracy_gap": gap_mean,
        "std_train_val_accuracy_gap": gap_std,
        "oof": oof,
        "folds": record,
    }


def save_oof(path: Path, oof_logits: dict[str, torch.Tensor], y_dev: list[int], fold_rows: list[dict], signature: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "signature": signature,
            "oof_logits": {name: logits.cpu() for name, logits in oof_logits.items()},
            "targets": y_dev,
            "fold_rows": fold_rows,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.folds < 2 or args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("folds, epochs, patience, and batch-size must be positive; num-workers cannot be negative")
    recipe = AUGMENTATION_RECIPES[args.augmentation]
    os.environ["COMPCLASS_PROFILE"] = args.profile
    os.environ["COMPCLASS_SEED"] = str(args.seed)
    os.environ["COMPCLASS_BATCH_SIZE"] = str(args.batch_size)
    os.environ["COMPCLASS_TRAINING_VARIANT"] = recipe
    os.environ["COMPCLASS_FORCE_RETRAIN"] = "0"
    work_dir, notebook_path = resolve_paths()
    os.environ["COMPCLASS_WORK_DIR"] = str(work_dir)
    if str(work_dir) not in sys.path:
        sys.path.insert(0, str(work_dir))
    runner = importlib.import_module("12_model_family_benchmark")
    namespace = runner.load_notebook_namespace(notebook_path)
    namespace["SEED"] = args.seed
    requested = tuple(name.strip() for name in (args.models or ",".join(runner.DEFAULT_MODELS)).split(",") if name.strip())
    model_names = list(dict.fromkeys(requested))
    unknown = sorted(set(model_names) - set(runner.MODEL_SETTINGS))
    if unknown or not model_names:
        raise ValueError(f"Unknown or empty model list: {unknown or model_names}")

    development_rows = list(namespace["all_splits"]["train"]) + list(namespace["all_splits"]["val"])
    max_per_class = args.max_per_class or (2 if args.profile == "smoke" else 0)
    if max_per_class:
        if max_per_class < 1:
            raise ValueError("--max-per-class must be positive")
        grouped = {}
        for sample in development_rows:
            grouped.setdefault(sample.label, []).append(sample)
        development_rows = [
            sample
            for label in sorted(grouped)
            for sample in sorted(grouped[label], key=lambda item: item.path)[:max_per_class]
        ]
    external_rows, external_info = load_external_training_rows(namespace, args, namespace["class_names"])
    if external_rows:
        # Include the selected training-only material in checkpoint fingerprints,
        # while keeping the OOF development rows and held-out policy unchanged.
        namespace["all_samples"] = list(namespace["all_samples"]) + external_rows
    class_to_idx = {label: index for index, label in enumerate(namespace["class_names"])}
    y_dev = [class_to_idx[sample.label] for sample in development_rows]
    fold_rows, fold_info = make_folds(development_rows, folds=args.folds, seed=args.seed, class_names=namespace["class_names"])
    run_signature = short_hash(
        {
            "dataset_fingerprint": namespace["_fingerprint"](namespace["all_samples"]),
            "development_rows": [(sample.label, sha256_file(sample.path)) for sample in development_rows],
            "folds": args.folds,
            "seed": args.seed,
            "models": model_names,
            "augmentation": args.augmentation,
            "recipe": recipe,
            "epochs": args.epochs,
            "patience": args.patience,
            "tta": args.tta,
            "external_augmentation": external_info,
        }
    )
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = namespace["METRICS_DIR"] / report_path
    oof_path = report_path.with_name(report_path.stem + "_oof.pt")
    csv_path = report_path.with_suffix(".csv")
    existing = {}
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing.get("run_signature") not in (None, run_signature):
            raise RuntimeError(f"Existing report has a different run signature: {report_path}; use a new --report")
    oof_logits = {name: torch.full((len(development_rows), len(namespace["class_names"])), float("nan")) for name in model_names}
    completed_records = existing.get("fold_records", [])
    completed = {(row["model"], row["fold"]) for row in completed_records}
    if oof_path.exists():
        payload = torch.load(oof_path, map_location="cpu")
        if payload.get("signature") == run_signature:
            for name, logits in payload.get("oof_logits", {}).items():
                if name in oof_logits and tuple(logits.shape) == tuple(oof_logits[name].shape):
                    oof_logits[name] = logits

    report = {
        "status": "running",
        "run_signature": run_signature,
        "augmentation": {
            "name": args.augmentation,
            "training_variant": recipe,
            "scope": "train fold only",
            "tta": args.tta,
        },
        "models": model_names,
        "folds": args.folds,
        "epochs": args.epochs,
        "patience": args.patience,
        "profile": args.profile,
        "max_per_class": max_per_class or None,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "dataset_fingerprint": namespace["_fingerprint"](namespace["all_samples"]),
        "development_pool": "legacy raw train + validation only",
        "external_augmentation": external_info,
        "legacy_test_policy": "legacy raw test was not loaded into a DataLoader, evaluated, or used for selection",
        "grouping": fold_info["grouping"],
        "fold_assignment": [
            {key: value for key, value in fold.items() if key not in {"train_indices", "val_indices"}}
            for fold in fold_rows
        ],
        "fold_records": completed_records,
        "model_summaries": [],
        "aggregations": [],
    }
    write_json(report_path, report)

    for model_name in model_names:
        for fold in fold_rows:
            fold_number = fold["fold"]
            key = (model_name, fold_number)
            fold_logits = oof_logits[model_name][fold["val_indices"]]
            if key in completed and not torch.isnan(fold_logits).any():
                continue
            settings = recipe_settings(runner, model_name, args, recipe)
            settings.update({"cv_fold": fold_number, "cv_folds": args.folds, "augmentation_name": args.augmentation, "tta": args.tta})
            namespace["SEED"] = args.seed + fold_number * 1000
            namespace["SPLIT_SIGNATURE"] = short_hash(
                {
                    "development_pool": [(sample.label, sha256_file(sample.path)) for sample in development_rows],
                    "fold": fold_number,
                    "train_indices": fold["train_indices"],
                    "val_indices": fold["val_indices"],
                }
            )
            fold_splits = {
                "train": [development_rows[index] for index in fold["train_indices"]] + external_rows,
                "val": [development_rows[index] for index in fold["val_indices"]],
            }
            loaders = namespace["make_loaders"](fold_splits, namespace["class_names"], batch_size=args.batch_size, num_workers=args.num_workers)
            checkpoint = namespace["WEIGHTS_DIR"] / f"{args.artifact_prefix}_{args.augmentation}_{model_name}_fold{fold_number}.pt"
            progress = checkpoint.with_suffix(".progress.pt")
            fingerprint = short_hash({"run_signature": run_signature, "model": model_name, "fold": fold_number, "settings": settings})
            print(f"=== {model_name} fold={fold_number + 1}/{args.folds} ===", flush=True)
            training = runner.train_trial(
                namespace,
                model_name,
                settings,
                loaders,
                checkpoint,
                progress,
                fingerprint,
                epochs=args.epochs,
                patience=args.patience,
            )
            logits = load_checkpoint_logits(runner, namespace, checkpoint, loaders["val"], tta=args.tta)
            val_indices = fold["val_indices"]
            oof_logits[model_name][val_indices] = logits
            oof_val = runner.metrics(
                namespace,
                [y_dev[index] for index in val_indices],
                logits.argmax(dim=1).tolist(),
            )
            fold_record = {
                "model": model_name,
                "family": runner.MODEL_SETTINGS[model_name]["family"],
                "fold": fold_number,
                "base_train_size": fold["train_size"],
                "external_train_size": len(external_rows),
                "train_size": fold["train_size"] + len(external_rows),
                "val_size": fold["val_size"],
                "selection_val": training["val"],
                "oof_val": oof_val,
                "training": training["training"],
                "checkpoint": str(checkpoint),
                "checkpoint_retained": args.keep_checkpoints,
            }
            report["fold_records"] = [
                row for row in report["fold_records"] if (row["model"], row["fold"]) != key
            ] + [fold_record]
            report["fold_records"].sort(key=lambda row: (model_names.index(row["model"]), row["fold"]))
            save_oof(oof_path, oof_logits, y_dev, fold_rows, run_signature)
            write_json(report_path, report)
            if not args.keep_checkpoints and checkpoint.exists():
                checkpoint.unlink()
            del loaders, logits, training
            gc.collect()
            if namespace["DEVICE"].type == "cuda":
                torch.cuda.empty_cache()

    if any(torch.isnan(logits).any() for logits in oof_logits.values()):
        raise RuntimeError("Run ended without complete out-of-fold predictions")
    report["model_summaries"] = [
        model_summary(
            runner,
            namespace,
            name,
            [row for row in report["fold_records"] if row["model"] == name],
            oof_logits[name],
            y_dev,
            fold_rows,
        )
        for name in model_names
    ]
    all_models_label = f"all_{len(model_names)}"
    report["aggregations"] = [
        aggregation_summary(runner, namespace, model_names, oof_logits, y_dev, fold_rows, name=f"{all_models_label}_mean_probability", mode="mean_probability"),
        aggregation_summary(runner, namespace, model_names, oof_logits, y_dev, fold_rows, name=f"{all_models_label}_mean_logits", mode="mean_logits"),
        aggregation_summary(runner, namespace, model_names, oof_logits, y_dev, fold_rows, name=f"{all_models_label}_majority_vote", mode="majority_vote"),
    ]
    if all(model in model_names for model in TOP3_MODELS):
        report["aggregations"].append(
            aggregation_summary(runner, namespace, list(TOP3_MODELS), oof_logits, y_dev, fold_rows, name="top3_convnext_mean_probability", mode="mean_probability")
        )
    report["status"] = "complete_test_deferred"
    report["best_single_model_by_oof_macro_f1"] = max(report["model_summaries"], key=lambda row: row["oof"]["macro_f1"])["model"]
    report["best_aggregation_by_oof_macro_f1"] = max(report["aggregations"], key=lambda row: row["oof"]["macro_f1"])["name"]
    write_json(report_path, report)
    table_rows = []
    for row in report["model_summaries"]:
        table_rows.append(
            {
                "kind": "model",
                "name": row["model"],
                "family": row["family"],
                "fold_macro_f1_mean": row["fold_macro_f1_mean"],
                "fold_macro_f1_std": row["fold_macro_f1_std"],
                "oof_accuracy": row["oof"]["accuracy"],
                "oof_macro_f1": row["oof"]["macro_f1"],
                "oof_weighted_f1": row["oof"]["weighted_f1"],
                "mean_train_val_accuracy_gap": row["mean_train_val_accuracy_gap"],
            }
        )
    for row in report["aggregations"]:
        table_rows.append(
            {
                "kind": "aggregation",
                "name": row["name"],
                "family": "ensemble",
                "fold_macro_f1_mean": row["fold_macro_f1_mean"],
                "fold_macro_f1_std": row["fold_macro_f1_std"],
                "oof_accuracy": row["oof"]["accuracy"],
                "oof_macro_f1": row["oof"]["macro_f1"],
                "oof_weighted_f1": row["oof"]["weighted_f1"],
                "mean_train_val_accuracy_gap": math.nan,
            }
        )
    write_csv(csv_path, table_rows)
    print(json.dumps({"status": report["status"], "report": str(report_path), "csv": str(csv_path), "best_single": report["best_single_model_by_oof_macro_f1"], "best_aggregation": report["best_aggregation_by_oof_macro_f1"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
