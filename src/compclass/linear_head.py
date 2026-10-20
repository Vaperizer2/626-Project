"""A softmax linear head trained with gradient descent on frozen backbone features.

This is the "transfer learning by feature extraction" recipe written out as an ordinary PyTorch training loop,
so it produces the same per-epoch learning curves as the fine-tuned networks (``compclass.train``):

    frozen backbone (e.g. SigLIP So400m/14 @384) -> f in R^D  (cached, eval preprocessing, no augmentation)
    z = (f - mu_train) / sd_train                              (standardised with TRAIN-split statistics only)
    logits = W z + b,  W in R^{C x D}                          (one nn.Linear; the only trainable parameters)
    loss = cross-entropy (no label smoothing), AdamW, constant LR, mini-batches, early stopping on val macro-F1

``compclass.features.fit_probe`` (scikit-learn L-BFGS logistic regression, C picked by grouped CV) solves
essentially the same convex problem with a full-batch quasi-Newton solver (no learning-rate schedule, no
epochs); this module exists to log the optimisation per epoch and
to make the weight-decay choice on the validation split explicit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .evaluate import compute_metrics


@dataclass
class HeadResult:
    model: nn.Linear
    mu: np.ndarray
    sd: np.ndarray
    history: list[dict[str, Any]]
    best_epoch: int
    best_val_macro_f1: float
    settings: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            z = torch.as_tensor((X - self.mu) / self.sd, dtype=torch.float32)
            return self.model.eval()(z).softmax(1).numpy()

    def state(self) -> dict[str, Any]:
        """Everything needed to rebuild the classifier on top of the frozen backbone."""
        return {"weight": self.model.weight.detach().cpu(), "bias": self.model.bias.detach().cpu(),
                "mu": torch.as_tensor(self.mu), "sd": torch.as_tensor(self.sd), "settings": self.settings,
                "best_epoch": self.best_epoch, "best_val_macro_f1": self.best_val_macro_f1}


def standardiser(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean / std (population std, zero std -> 1), as sklearn's StandardScaler."""
    mu = X.mean(0).astype(np.float32)
    sd = X.std(0).astype(np.float32)
    sd[sd == 0] = 1.0
    return mu, sd


def _eval(model: nn.Module, Z: torch.Tensor, y: torch.Tensor, classes: list[str]) -> tuple[float, float, float]:
    with torch.no_grad():
        logits = model.eval()(Z)
        loss = float(F.cross_entropy(logits, y))
        m = compute_metrics(y.numpy(), logits.argmax(1).numpy(), classes)
    return loss, m["accuracy"], m["macro_f1"]


def train_head(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, classes: list[str], *,
               lr: float = 1e-3, weight_decay: float = 1e-3, epochs: int = 200, patience: int = 20,
               batch_size: int = 256, seed: int = 0, verbose: bool = False) -> HeadResult:
    """Train ``nn.Linear(D, C)`` on standardised features; keep the epoch with the best validation macro-F1.

    History per epoch: ``train_loss`` (mean mini-batch cross-entropy during the epoch), ``train_ce`` /
    ``train_acc`` / ``train_macro_f1`` (full pass over the training rows after the epoch), ``val_loss``,
    ``val_acc``, ``val_macro_f1``. No augmentation and no label smoothing, so train and val losses are directly
    comparable. Ties in macro-F1 keep the earlier epoch.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    mu, sd = standardiser(Xtr)
    Ztr = torch.as_tensor((Xtr - mu) / sd, dtype=torch.float32)
    Zva = torch.as_tensor((Xva - mu) / sd, dtype=torch.float32)
    ytr_t, yva_t = torch.as_tensor(ytr, dtype=torch.long), torch.as_tensor(yva, dtype=torch.long)
    model = nn.Linear(Ztr.shape[1], len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_f1, best_epoch, best_state, bad, history = -1.0, -1, None, 0, []
    n = len(Ztr)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(model(Ztr[idx]), ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        tr_ce, tr_acc, tr_f1 = _eval(model, Ztr, ytr_t, classes)
        va_loss, va_acc, va_f1 = _eval(model, Zva, yva_t, classes)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "train_ce": tr_ce, "train_acc": tr_acc,
                        "train_macro_f1": tr_f1, "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1,
                        "lr": lr})
        if verbose:
            print(f"[head wd={weight_decay:g}] epoch {epoch:3d}  train {tr_ce:.4f}/{tr_acc:.4f}  "
                  f"val {va_loss:.4f}/{va_acc:.4f}/{va_f1:.4f}", flush=True)
        if va_f1 > best_f1:
            best_f1, best_epoch, best_state, bad = va_f1, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    settings = {"lr": lr, "weight_decay": weight_decay, "max_epochs": epochs, "patience": patience,
                "batch_size": batch_size, "seed": seed, "optimizer": "AdamW", "schedule": "constant",
                "loss": "cross-entropy", "epochs_run": len(history)}
    return HeadResult(model.eval(), mu, sd, history, best_epoch, best_f1, settings)


def select_weight_decay(Xtr, ytr, Xva, yva, classes, grid=(1e-4, 1e-3, 1e-2, 1e-1), **kw
                        ) -> tuple[HeadResult, list[dict[str, Any]]]:
    """Train one head per weight decay; pick the best validation macro-F1 (ties: higher val accuracy, then the
    smaller weight decay). Uses only the training and validation splits."""
    results, table = [], []
    for wd in grid:
        r = train_head(Xtr, ytr, Xva, yva, classes, weight_decay=wd, **kw)
        h = r.history[r.best_epoch]
        results.append(r)
        table.append({"weight_decay": wd, "best_epoch": r.best_epoch, "epochs_run": len(r.history),
                      "val_macro_f1": h["val_macro_f1"], "val_acc": h["val_acc"], "val_loss": h["val_loss"]})
    order = sorted(range(len(grid)), key=lambda i: (-table[i]["val_macro_f1"], -table[i]["val_acc"], grid[i]))
    for i, row in enumerate(table):
        row["selected"] = i == order[0]
    return results[order[0]], table
