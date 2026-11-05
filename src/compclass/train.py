"""Generic fine-tuning loop shared by every family.

    uv run python -m compclass.train --config configs/convnext_tiny.yaml --fold 0
    uv run python -m compclass.train --config configs/vit_b16.yaml --fold 0 1 2 3 4 --set train.lr=5e-5

Recipe (each knob lives in the YAML ``train:`` / ``aug:`` sections, documented in configs/base.yaml):

* AdamW, no weight decay on norms/biases; optional **layer-wise LR decay** (early layers learn slower --
  they hold the most general features) via timm's parameter grouping.
* Linear warm-up, then cosine decay, stepped per optimiser step.
* Optional **frozen-backbone warm-up**: the first ``freeze_epochs`` train only the new head, so random head
  gradients do not wreck the pretrained features.
* Label smoothing, or Mixup/CutMix (soft targets) when enabled.
* Mixed precision (``amp: fp16|bf16|none``) and **gradient accumulation** (effective batch =
  ``batch_size * accum_steps``) so ViT-B fits in 8 GB.
* Validation every epoch; the best macro-F1 checkpoint is kept, training stops after ``patience`` epochs
  without improvement. One line per run is appended to ``results/runs.jsonl``.

Per-epoch ``history`` (learning curves): ``train_loss`` is the optimised objective (label-smoothed / soft-target
cross-entropy on the augmented training batches, mean over batches); ``train_ce`` is the plain, *unsmoothed*
cross-entropy and ``train_acc`` the accuracy on those same augmented batches (computed from the logits of the
forward pass used for the update, against the original labels; ``train_ce`` is None when Mixup/CutMix is on);
``val_loss`` is the plain cross-entropy (no label smoothing) on the validation split with eval preprocessing,
plus ``val_acc`` and ``val_macro_f1``. Compare ``train_ce`` (not ``train_loss``) with ``val_loss``.

``--test`` evaluates the best checkpoint once on the sealed test split after training and writes
``test_preds.npz`` (ids, y, probs) and ``test_metrics.json`` next to ``best.pt``.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import Config, config_hash, load_config
from .data import ImageDataset, build_mixup, build_transforms, get_fold, get_test, load_manifest
from .evaluate import compute_metrics, log_run, predict, write_eval
from .models import build_model

# Fewer fragmentation OOMs on 8 GB cards (must be set before CUDA initialises).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
AMP = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None, None: None}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(model: nn.Module, t: dict[str, Any]) -> torch.optim.Optimizer:
    """AdamW with timm's parameter groups (no decay on 1-D params; per-layer LR scale if ``layer_decay``)."""
    from timm.optim import create_optimizer_v2

    layer_decay = t.get("layer_decay") if hasattr(model, "group_matcher") else None
    opt = create_optimizer_v2(model, opt="adamw", lr=t["lr"], weight_decay=t["weight_decay"],
                              layer_decay=layer_decay)
    for g in opt.param_groups:  # bake timm's per-group scale into lr so a plain LambdaLR respects it
        g["lr"] = g["lr"] * g.pop("lr_scale", 1.0)
    return opt


def warmup_cosine(opt: torch.optim.Optimizer, warmup: int, total: int,
                  frozen: int = 0) -> torch.optim.lr_scheduler.LambdaLR:
    """LR multiplier: linear 0->1 over ``warmup`` steps, then cosine 1->0 at ``total``.

    If the backbone is frozen for the first ``frozen`` steps (head-only warm-up), the head ramps up during
    that phase and the warm-up restarts when the backbone unfreezes. Otherwise the freshly unfrozen
    pretrained layers would receive the full peak LR on their very first update, which can wreck the
    pretrained features they were chosen for.
    """

    def f(step: int) -> float:
        if step < frozen:
            return (step + 1) / frozen
        s = step - frozen
        if s < warmup:
            return (s + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total - frozen - warmup))))

    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def permanently_frozen(model: nn.Module, trainable_blocks: int | None) -> set[int]:
    """Parameter ids that stay frozen for the whole run ("partial fine-tuning").

    With ``trainable_blocks = N`` on a transformer that has a ``blocks`` list (timm ViT / DINOv2 / CLIP /
    SigLIP), only the last N blocks plus everything after them (final norm, attention-pool, head) are
    trained; the patch embedding, position embedding and earlier blocks keep their pretrained values. Frozen
    parameters get no gradients and no AdamW state, which is what lets a 400M-parameter model fit in 8 GB.
    """
    if not trainable_blocks or not hasattr(model, "blocks"):
        return set()
    n = len(model.blocks)
    keep = {id(p) for b in list(model.blocks)[max(0, n - trainable_blocks):] for p in b.parameters()}
    early = {id(p) for b in list(model.blocks)[: max(0, n - trainable_blocks)] for p in b.parameters()}
    stem = {id(p) for name, p in model.named_parameters()
            if name.split(".")[0] in {"patch_embed", "pos_embed", "cls_token", "reg_token", "norm_pre"}}
    return (early | stem) - keep


def set_backbone_trainable(model: nn.Module, trainable: bool, frozen: set[int] = frozenset()) -> None:
    """Freeze/unfreeze everything except the classifier head (``get_classifier()``, as in timm).

    Parameters in ``frozen`` stay frozen regardless (see ``permanently_frozen``).
    """
    head = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad_((trainable or id(p) in head) and id(p) not in frozen)


class _WithIndex(torch.utils.data.Dataset):
    """(x, y) -> (x, y, i) so a batch can look up its rows' precomputed teacher logits."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y, i


def load_teacher(distill: dict[str, Any], fold: int, train_rows, fingerprint: str, device) -> torch.Tensor:
    """Teacher logits aligned with ``train_rows`` from ``distill.teacher`` (a path with ``{fold}``).

    Refuses a teacher built on another manifest or one that was fitted on this fold's validation rows.
    """
    tf = np.load(str(distill["teacher"]).format(fold=fold), allow_pickle=False)
    if str(tf["manifest"]) != fingerprint:
        raise SystemExit(f"teacher manifest {tf['manifest']} != {fingerprint}")
    pos = {i: j for j, i in enumerate(tf["ids"].tolist())}
    idx = np.array([pos[r["id"]] for r in train_rows])
    if not bool(tf["is_train"][idx].all()):
        raise SystemExit("teacher was fitted on rows outside this fold's training split")
    return torch.tensor(tf["logits"][idx], dtype=torch.float32, device=device)


def run(cfg: Config, fold: int, test: bool = False) -> dict[str, Any]:
    """Train one model on one CV fold; returns the record appended to the run log.

    ``test=True``: afterwards score the best checkpoint on the sealed test split (see module docstring).
    """
    d, t, a = cfg["data"], cfg["train"], cfg.get("aug", {})
    seed_everything(t.get("seed", 0))
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu":
        torch.set_num_threads(t.get("cpu_threads", min(8, os.cpu_count() or 1)))
    torch.backends.cudnn.benchmark = True
    amp_dtype = AMP[t.get("amp", "fp16")] if device.type == "cuda" else None

    m = load_manifest(d["manifest"])
    train_rows, val_rows = get_fold(m, fold)
    h = config_hash(cfg)
    run_dir = Path(cfg["output"]["runs_dir"]) / f"{cfg['name']}-{h}" / f"fold{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)

    mc = cfg["model"]
    model, spec = build_model(mc["key"], len(m.classes), mc.get("pretrained", True), **mc.get("kwargs", {}))
    if t.get("grad_checkpointing") and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)  # recompute activations in backward: less VRAM, ~30% slower
    if t.get("head_init"):  # LP-FT (Kumar et al. 2022): start from a fitted linear probe instead of a random head;
        hi = np.load(str(t["head_init"]).format(fold=fold))  # {weight [C, D], bias [C]}, e.g. tools/views/probe_heads.py
        head = model.get_classifier()
        with torch.no_grad():
            head.weight.copy_(torch.as_tensor(hi["weight"], dtype=head.weight.dtype))
            head.bias.copy_(torch.as_tensor(hi["bias"], dtype=head.bias.dtype))
        print(f"[{cfg['name']} f{fold}] head initialised from {str(t['head_init']).format(fold=fold)}", flush=True)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    nw = d.get("num_workers", 4)
    common = dict(num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    distill = t.get("distill")
    train_ds: torch.utils.data.Dataset = ImageDataset(train_rows, m.root, m.class_to_idx, build_transforms(spec, a, True))
    if distill:
        train_ds = _WithIndex(train_ds)
    train_dl = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, drop_last=True, **common)
    val_dl = DataLoader(ImageDataset(val_rows, m.root, m.class_to_idx, build_transforms(spec, a, False)),
                        batch_size=t.get("eval_batch_size", 2 * t["batch_size"]), **common)

    if hasattr(model, "init_from_data"):  # data-dependent initialisation (e.g. whitened stem), before any step
        model.init_from_data(train_dl, device)

    mixup = build_mixup(a, len(m.classes), t.get("label_smoothing", 0.0))
    if mixup is not None:
        from timm.loss import SoftTargetCrossEntropy

        criterion: nn.Module = SoftTargetCrossEntropy()  # mixup already produced smoothed soft targets
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=t.get("label_smoothing", 0.0))

    teacher = None
    if distill:  # knowledge distillation from precomputed per-image teacher logits (tools/make_teacher.py)
        if mixup is not None:
            raise SystemExit("distillation is not combined with mixup/cutmix")
        teacher = load_teacher(distill, fold, train_rows, m.fingerprint, device)
        kd_alpha, kd_temp = float(distill.get("alpha", 0.5)), float(distill.get("temperature", 4.0))
        print(f"[{cfg['name']} f{fold}] distilling from {distill['teacher']}  alpha {kd_alpha}  T {kd_temp}", flush=True)
    accum = t.get("accum_steps", 1)
    steps_per_epoch = min(len(train_dl), t.get("max_steps_per_epoch") or len(train_dl))
    opt_steps = math.ceil(steps_per_epoch / accum)
    frozen = permanently_frozen(model, t.get("trainable_blocks")) | set(getattr(model, "frozen_param_ids", ()))
    set_backbone_trainable(model, True, frozen)  # timm's optimizer skips requires_grad=False params
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = make_optimizer(model, t)
    sched = warmup_cosine(opt, int(t.get("warmup_epochs", 0) * opt_steps), t["epochs"] * opt_steps,
                          frozen=int(t.get("freeze_epochs", 0) * opt_steps))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    ema = None
    if t.get("ema"):  # exponential moving average of the weights; evaluated and saved instead of the raw model
        from timm.utils import ModelEmaV3

        ema = ModelEmaV3(model, decay=float(t["ema"]), use_warmup=True, device=device)
    eval_model = ema.module if ema is not None else model

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    best_f1, best_epoch, bad, history = -1.0, -1, 0, []
    n_seen, train_time = 0, 0.0
    for epoch in range(t["epochs"]):
        backbone_on = epoch >= t.get("freeze_epochs", 0)
        set_backbone_trainable(model, backbone_on, frozen)
        model.train()
        # A frozen CNN backbone should not keep updating its BatchNorm running statistics; ``train.freeze_bn`` keeps
        # them frozen for the whole run (small-batch fine-tuning of a BN network; the affine parameters still train).
        if not backbone_on or t.get("freeze_bn"):
            for mod in model.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        t0, loss_sum, ce_sum, n_correct, n_epoch = time.perf_counter(), 0.0, 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_dl):
            if step >= steps_per_epoch:
                break
            x, y = batch[0], batch[1]
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_hard = y  # original labels, kept for train accuracy / plain CE
            if mixup is not None:
                x, y = mixup(x, y)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
                loss = criterion(logits, y)
                if teacher is not None:  # Hinton et al.: KL(teacher^T || student^T) * T^2, mixed with the CE
                    tl = teacher[batch[2].to(device)]
                    kd = F.kl_div(F.log_softmax(logits.float() / kd_temp, 1), F.softmax(tl / kd_temp, 1),
                                  reduction="batchmean") * kd_temp ** 2
                    loss = (1 - kd_alpha) * loss + kd_alpha * kd
                loss = loss / accum
            scaler.scale(loss).backward()
            if (step + 1) % accum == 0 or step + 1 == steps_per_epoch:
                scaler.unscale_(opt)  # so clipping sees true gradient magnitudes
                if t.get("grad_clip"):
                    nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema is not None:
                    ema.update(model, step=epoch * opt_steps + (step + 1) // accum)
            loss_sum += loss.item() * accum
            with torch.no_grad():  # monitoring only: no effect on the update
                lf = logits.detach().float()
                n_correct += int((lf.argmax(1) == y_hard).sum())
                if mixup is None:
                    ce_sum += float(F.cross_entropy(lf, y_hard, reduction="sum"))
            n_epoch += x.shape[0]
            n_seen += x.shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.perf_counter() - t0

        probs, y_val = predict(eval_model, val_dl, device, amp_dtype)
        vm = compute_metrics(y_val, probs.argmax(1), m.classes)
        f1 = vm["macro_f1"]
        val_loss = float(-np.log(np.clip(probs[np.arange(len(y_val)), y_val], 1e-12, None)).mean())
        history.append({"epoch": epoch, "train_loss": loss_sum / steps_per_epoch,
                        "train_ce": ce_sum / n_epoch if mixup is None else None, "train_acc": n_correct / n_epoch,
                        "val_loss": val_loss, "val_acc": vm["accuracy"], "val_macro_f1": f1,
                        "lr": opt.param_groups[-1]["lr"]})
        print(f"[{cfg['name']} f{fold}] epoch {epoch:3d}  loss {loss_sum / steps_per_epoch:.4f}  "
              f"train acc {n_correct / n_epoch:.4f}  val loss {val_loss:.4f}  val acc {vm['accuracy']:.4f}  "
              f"val macro-F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1, best_epoch, bad = f1, epoch, 0
            torch.save({"model": eval_model.state_dict(), "config": cfg, "config_hash": h, "fold": fold,
                        "classes": m.classes, "preprocess": spec.to_dict(), "manifest_fingerprint": m.fingerprint,
                        "epoch": epoch, "val_macro_f1": f1}, run_dir / "best.pt")
        else:
            bad += 1
            if bad >= t.get("patience", 10):
                print(f"early stop: no improvement for {bad} epochs", flush=True)
                break

    # Final numbers always come from the best checkpoint, not the last epoch.
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False)["model"])
    probs, y_val = predict(model, val_dl, device, amp_dtype)
    metrics = compute_metrics(y_val, probs.argmax(1), m.classes)
    np.savez_compressed(run_dir / "val_preds.npz", ids=np.array([r["id"] for r in val_rows]), y=y_val, probs=probs)
    record = {"method": "finetune", "name": cfg["name"], "model": mc["key"], "config_hash": h, "fold": fold,
              "manifest": m.fingerprint, "params": n_params, "trainable_params": n_trainable,
              "input_size": spec.input_size,
              "epochs_run": len(history), "best_epoch": best_epoch, "train_time_s": round(train_time, 1),
              "train_img_per_s": round(n_seen / max(train_time, 1e-9), 1),
              "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
              "effective_batch": t["batch_size"] * accum, "amp": t.get("amp") if amp_dtype else "none",
              "run_dir": str(run_dir), "history": history, "config": cfg, **metrics}
    log_run(record, cfg["output"]["results"])
    print(f"[{cfg['name']} f{fold}] best epoch {best_epoch}: acc {metrics['accuracy']:.4f}  "
          f"macro-F1 {metrics['macro_f1']:.4f}  -> {run_dir}", flush=True)
    if test:
        test_rows = get_test(m, confirm=True)
        test_dl = DataLoader(ImageDataset(test_rows, m.root, m.class_to_idx, build_transforms(spec, a, False)),
                             batch_size=t.get("eval_batch_size", 2 * t["batch_size"]), **common)
        tp, ty = predict(model, test_dl, device, amp_dtype)
        tm = write_eval(run_dir, "test", [r["id"] for r in test_rows], ty, tp, m.classes,
                        {"name": cfg["name"], "config_hash": h, "fold": fold, "best_epoch": best_epoch,
                         "manifest": m.fingerprint})
        record["test"] = {k: tm[k] for k in ("accuracy", "macro_f1")}
        print(f"[{cfg['name']} f{fold}] TEST: acc {tm['accuracy']:.4f}  macro-F1 {tm['macro_f1']:.4f}", flush=True)
    return record


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, nargs="+", required=True, help="one or more CV folds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides")
    ap.add_argument("--test", action="store_true",
                    help="after training, score the best checkpoint on the sealed test split (final comparison only)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    for k in args.fold:
        run(cfg, k, test=args.test)


if __name__ == "__main__":
    main()
