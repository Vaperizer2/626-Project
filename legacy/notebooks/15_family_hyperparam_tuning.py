"""Validation-only Optuna tuning for the top model-family candidates.

The candidate set is frozen from the logged 14-model family/size sweep:
the top three overall plus the best candidate from each family. The held-out
test split is intentionally not loaded or evaluated by this script.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import time
from pathlib import Path

import optuna
import torch


TOP_THREE_OVERALL = [
    {"model": "convnext_tiny", "val_macro_f1": 0.7427, "rank": 1},
    {"model": "convnext_base", "val_macro_f1": 0.7407, "rank": 2},
    {"model": "convnext_small", "val_macro_f1": 0.7266, "rank": 3},
]
FAMILY_PICKS = [
    {"family": "ResNet", "model": "resnet152", "val_macro_f1": 0.7257},
    {"family": "DenseNet", "model": "densenet121", "val_macro_f1": 0.6783},
    {"family": "EfficientNet", "model": "efficientnet_v2_s", "val_macro_f1": 0.7082},
    {"family": "MobileNet", "model": "mobilenet_v3_large", "val_macro_f1": 0.6725},
    {"family": "ConvNeXt", "model": "convnext_tiny", "val_macro_f1": 0.7427},
    {"family": "Swin", "model": "swin_b", "val_macro_f1": 0.7234},
]
DEFAULT_MODELS = tuple(
    dict.fromkeys(
        [row["model"] for row in TOP_THREE_OVERALL]
        + [row["model"] for row in FAMILY_PICKS]
    )
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated candidates; defaults to top-three plus family picks.",
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--profile", choices=("smoke", "standard", "paper"), default="paper")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=0, help="Override the profile batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers on the compute host.")
    parser.add_argument("--threads", type=int, default=0, help="Torch CPU threads; 0 leaves the default unchanged.")
    parser.add_argument("--report", default="family_hyperparam_tuning.json")
    parser.add_argument("--artifact-prefix", default="family_hparam")
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow random weights when pretrained weights are unavailable; useful only for smoke checks.",
    )
    return parser.parse_args()


def resolve_paths() -> tuple[Path, Path]:
    requested = Path(os.getenv("COMPCLASS_WORK_DIR", Path.cwd())).expanduser().resolve()
    if (requested / "01_primary_comparative.ipynb").exists():
        return requested, requested / "01_primary_comparative.ipynb"
    notebook_path = requested / "notebooks" / "01_primary_comparative.ipynb"
    if notebook_path.exists():
        return notebook_path.parent, notebook_path
    raise FileNotFoundError(f"Could not find 01_primary_comparative.ipynb under {requested}")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def model_family(model: str) -> str:
    for row in FAMILY_PICKS:
        if row["model"] == model:
            return row["family"]
    for row in TOP_THREE_OVERALL:
        if row["model"] == model:
            return "ConvNeXt"
    return "unknown"


def train_trial(
    *,
    trial: optuna.Trial,
    runner,
    namespace: dict,
    model_name: str,
    loaders: dict,
    epochs: int,
    patience: int,
    pretrained_state: dict[str, torch.Tensor] | None,
    best_so_far: dict,
    checkpoint_path: Path,
    seed: int,
) -> float:
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True),
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        "class_weight_power": trial.suggest_float("class_weight_power", 0.0, 0.75),
        "freeze_backbone_epochs": trial.suggest_int("freeze_backbone_epochs", 0, 2),
    }
    namespace["set_seed"](seed + trial.number)
    model = runner.build_model(
        namespace,
        model_name,
        len(namespace["class_names"]),
        pretrained=False,
    )
    if pretrained_state is not None:
        state = model.state_dict()
        for key, value in pretrained_state.items():
            if key in state and state[key].shape == value.shape:
                state[key] = value
        model.load_state_dict(state)
    model = model.to(namespace["DEVICE"])

    train_targets = [
        loaders["train"].dataset.class_to_idx[sample.label]
        for sample in loaders["train"].dataset.samples
    ]
    counts = torch.bincount(
        torch.tensor(train_targets), minlength=len(namespace["class_names"])
    ).float()
    class_weights = None
    if params["class_weight_power"] > 0:
        class_weights = (
            counts.mean() / counts.clamp_min(1)
        ).pow(params["class_weight_power"]).to(namespace["DEVICE"])
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=params["label_smoothing"],
    )

    if params["freeze_backbone_epochs"]:
        namespace["_set_head_or_backbone"](model, False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    best_val = -1.0
    best_state = None
    best_history = []
    history = []
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if params["freeze_backbone_epochs"] and epoch == params["freeze_backbone_epochs"] + 1:
            namespace["_set_head_or_backbone"](model, True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=params["lr"] * 0.1,
                weight_decay=params["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2
            )

        model.train()
        loss_total, correct, total, batches = 0.0, 0, 0, 0
        for inputs, targets in namespace["tqdm"](
            loaders["train"], desc=f"{model_name} trial={trial.number} epoch={epoch}/{epochs}", leave=False
        ):
            inputs, targets = inputs.to(namespace["DEVICE"]), targets.to(namespace["DEVICE"])
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            loss_total += float(loss.item())
            correct += int(logits.argmax(dim=1).eq(targets).sum())
            total += len(targets)
            batches += 1

        y_val, y_pred = namespace["predict_loader"](
            model, loaders["val"], namespace["DEVICE"]
        )
        val_metrics = runner.metrics(namespace, y_val, y_pred)
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
        trial.report(record["val_macro_f1"], epoch)
        if record["val_macro_f1"] > best_val + 1e-4:
            best_val = record["val_macro_f1"]
            best_state = runner.cpu_state(model)
            best_history = list(history)
            stale = 0
        else:
            stale += 1
        print(
            f"{model_name} trial={trial.number:02d} epoch={epoch:02d} "
            f"val_acc={record['val_accuracy']:.4f} val_macro_f1={record['val_macro_f1']:.4f}",
            flush=True,
        )
        if trial.should_prune():
            raise optuna.TrialPruned()
        if stale >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"No validation checkpoint was produced for {model_name}, trial {trial.number}")
    best_record = max(best_history, key=lambda row: row["val_macro_f1"])
    trial.set_user_attr("best_epoch", best_record["epoch"])
    trial.set_user_attr("val_accuracy", best_record["val_accuracy"])
    trial.set_user_attr("val_weighted_f1", best_record["val_weighted_f1"])
    trial.set_user_attr("train_accuracy", best_record["train_accuracy"])
    trial.set_user_attr("train_val_accuracy_gap", best_record["train_accuracy"] - best_record["val_accuracy"])
    trial.set_user_attr("wall_time_seconds", time.perf_counter() - started)

    if best_val > best_so_far.get("value", -1.0):
        best_so_far.update({"value": best_val, "trial": trial.number})
        torch.save(
            {
                "state_dict": best_state,
                "metadata": {
                    "model": model_name,
                    "family": model_family(model_name),
                    "settings": {**params, "epochs": epochs, "patience": patience},
                    "selection": "raw_val_macro_f1",
                    "dataset_fingerprint": namespace["_fingerprint"](namespace["all_samples"]),
                    "split_fingerprint": namespace["SPLIT_SIGNATURE"],
                    "class_names": namespace["class_names"],
                    "image_size": namespace["IMAGE_SIZE"],
                },
                "training": {"history": best_history},
            },
            checkpoint_path,
        )
    del model
    gc.collect()
    if namespace["DEVICE"].type == "cuda":
        torch.cuda.empty_cache()
    return best_val


def build_pretrained_state(runner, namespace: dict, model_name: str, allow_random_init: bool):
    if allow_random_init:
        return None
    try:
        source = runner.build_model(
            namespace,
            model_name,
            len(namespace["class_names"]),
            pretrained=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Pretrained weights for {model_name} are unavailable. "
            "Use a machine with the torchvision weights cached/network-enabled, "
            "or pass --allow-random-init only for a smoke check."
        ) from exc
    state = {key: value.detach().cpu().clone() for key, value in source.state_dict().items()}
    del source
    gc.collect()
    return state


def main() -> None:
    args = parse_args()
    if args.trials < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("trials, epochs, and patience must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if args.threads:
        torch.set_num_threads(args.threads)
    os.environ["COMPCLASS_PROFILE"] = args.profile
    os.environ["COMPCLASS_SEED"] = str(args.seed)
    os.environ["COMPCLASS_FORCE_RETRAIN"] = "0"
    if args.batch_size:
        if args.batch_size < 1:
            raise ValueError("batch-size must be positive")
        os.environ["COMPCLASS_BATCH_SIZE"] = str(args.batch_size)
    work_dir, notebook_path = resolve_paths()
    os.environ["COMPCLASS_WORK_DIR"] = str(work_dir)
    runner = importlib.import_module("12_model_family_benchmark")
    namespace = runner.load_notebook_namespace(notebook_path)
    train_val_splits = {name: namespace["splits"][name] for name in ("train", "val")}
    loaders = namespace["make_loaders"](
        train_val_splits,
        namespace["class_names"],
        batch_size=namespace["BATCH_SIZE"],
        num_workers=args.num_workers,
    )
    known_models = set(runner.MODEL_SETTINGS)
    models = tuple(dict.fromkeys(name.strip() for name in args.models.split(",") if name.strip()))
    unknown = sorted(set(models) - known_models)
    if unknown:
        raise ValueError(f"Unknown models: {unknown}; choose from {sorted(known_models)}")
    if not models:
        raise ValueError("At least one model is required")

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = namespace["METRICS_DIR"] / report_path
    database_path = namespace["METRICS_DIR"] / f"{args.artifact_prefix}.db"
    storage = f"sqlite:///{database_path.resolve()}"
    report = {
        "status": "running",
        "selection_metric": "raw_val_macro_f1",
        "test_deferred": True,
        "test_policy": "held-out test is not loaded by this tuner",
        "dataset_fingerprint": namespace["_fingerprint"](namespace["all_samples"]),
        "split_fingerprint": namespace["SPLIT_SIGNATURE"],
        "split_sizes": {name: len(rows) for name, rows in namespace["all_splits"].items()},
        "top_3_overall": TOP_THREE_OVERALL,
        "family_picks": FAMILY_PICKS,
        "requested_models": list(models),
        "trials_per_model": args.trials,
        "epochs": args.epochs,
        "patience": args.patience,
        "profile": args.profile,
        "batch_size": namespace["BATCH_SIZE"],
        "num_workers": args.num_workers,
        "pretrained": not args.allow_random_init,
        "models": [],
    }
    write_json(report_path, report)

    for model_name in models:
        print(f"\n=== {model_name} ({model_family(model_name)}) ===", flush=True)
        pretrained_state = build_pretrained_state(runner, namespace, model_name, args.allow_random_init)
        checkpoint_path = namespace["WEIGHTS_DIR"] / f"{args.artifact_prefix}_{model_name}.pt"
        best_so_far = {"value": -1.0, "trial": None}
        study = optuna.create_study(
            study_name=f"{args.artifact_prefix}_{model_name}",
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2),
        )
        study.optimize(
            lambda trial: train_trial(
                trial=trial,
                runner=runner,
                namespace=namespace,
                model_name=model_name,
                loaders=loaders,
                epochs=args.epochs,
                patience=args.patience,
                pretrained_state=pretrained_state,
                best_so_far=best_so_far,
                checkpoint_path=checkpoint_path,
                seed=args.seed,
            ),
            n_trials=args.trials,
            gc_after_trial=True,
        )
        completed = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
        best = study.best_trial
        row = {
            "model": model_name,
            "family": model_family(model_name),
            "study": study.study_name,
            "completed_trials": len(completed),
            "best_trial": best.number,
            "best_val_macro_f1": best.value,
            "best_val_accuracy": best.user_attrs.get("val_accuracy"),
            "best_val_weighted_f1": best.user_attrs.get("val_weighted_f1"),
            "best_epoch": best.user_attrs.get("best_epoch"),
            "train_val_accuracy_gap": best.user_attrs.get("train_val_accuracy_gap"),
            "best_params": best.params,
            "baseline_reference_val_macro_f1": next(
                (item["val_macro_f1"] for item in TOP_THREE_OVERALL + FAMILY_PICKS if item["model"] == model_name),
                None,
            ),
            "checkpoint": str(checkpoint_path),
        }
        row["improvement_over_logged_reference"] = (
            row["best_val_macro_f1"] - row["baseline_reference_val_macro_f1"]
            if row["baseline_reference_val_macro_f1"] is not None
            else None
        )
        report["models"] = [item for item in report["models"] if item["model"] != model_name]
        report["models"].append(row)
        write_json(report_path, report)
        del pretrained_state
        gc.collect()

    report["models"].sort(key=lambda row: row["best_val_macro_f1"], reverse=True)
    report["status"] = "validation_complete_test_deferred"
    report["best_model"] = report["models"][0]["model"]
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], "best_model": report["best_model"], "report": str(report_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
