"""Distil a student into the SigLIP So400m-384 pooled-embedding space from cached teacher views.

    uv run python tools/distill/train.py --student siglip2_b16_256 --pool POOL --teacher TEACHER --out RUN

Each epoch visits every training pool image once, through one of its cached views drawn at random, so the student
sees heavily augmented inputs whose exact teacher embedding is known. Pool-val images (common.is_pool_val) are
never trained on; teacher fidelity on their evaluation views is logged every epoch for the raw and EMA weights.
RUN/last.pt holds the latest state (resumable); RUN/final.pt holds the weights chosen for export.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TEACHER_DIM, build_student, distill_loss, is_pool_val, load_pool, render_view, to_tensor  # noqa: E402


class StudentViews(Dataset):
    """Item i of epoch e: image ``images[i]`` through a random cached view; returns (pixels, teacher row)."""

    def __init__(self, rows, images: np.ndarray, view_rows: list[np.ndarray], views: list[np.ndarray],
                 pre: dict, seed: int, eval_view: bool = False):
        self.rows, self.images, self.view_rows, self.views = rows, images, view_rows, views
        self.pre, self.seed, self.eval_view, self.epoch = pre, seed, eval_view, 0

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        k = self.images[i]
        j = 0 if self.eval_view else np.random.default_rng((self.seed, self.epoch, i)).integers(len(self.views[k]))
        row = self.rows[k]
        im = Image.open(row["path"]).convert("RGB")
        x = to_tensor(render_view(im, row["id"], int(self.views[k][j]), self.pre["input_size"]),
                      self.pre["mean"], self.pre["std"])
        return x, int(self.view_rows[k][j])


def load_source(pool_dir: Path, teacher_dir: Path):
    """Pool rows (paths made absolute) with their cached teacher embeddings and per-image view lists."""
    rows, _ = load_pool(pool_dir)
    rows = [{**r, "path": str(pool_dir / r["path"])} for r in rows]
    T, view_rows, views, ids = load_teacher(teacher_dir, len(rows))
    assert list(ids) == [r["id"] for r in rows], f"{teacher_dir} does not match {pool_dir}"
    return rows, T, view_rows, views


def load_teacher(teacher_dir: Path, n_images: int):
    idx = np.load(teacher_dir / "index.npz")
    shards = sorted(teacher_dir.glob("shard_[0-9][0-9][0-9].npy"))
    T = np.concatenate([np.load(s) for s in shards])
    if len(T) < len(idx["items"]):
        print(f"teacher cache partial: training on the first {len(T)} of {len(idx['items'])} views", flush=True)
    view_rows: list[list[int]] = [[] for _ in range(n_images)]
    views: list[list[int]] = [[] for _ in range(n_images)]
    for m, (k, v) in enumerate(idx["items"][:len(T)]):
        view_rows[k].append(m)
        views[k].append(int(v))
    return torch.from_numpy(T), [np.array(r) for r in view_rows], [np.array(v) for v in views], idx["ids"]


@torch.no_grad()
def fidelity(model, dl, T: torch.Tensor, dev) -> dict:
    model.eval()
    S, rows = [], []
    for x, r in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            S.append(model(x.to(dev, non_blocking=True)).float().cpu())
        rows.append(r)
    S, t = torch.cat(S), T[torch.cat(rows)].float()
    cos = torch.nn.functional.cosine_similarity(S, t, dim=1)
    return {"cos_mean": float(cos.mean()), "cos_p05": float(cos.quantile(0.05)),
            "rel_mse": float((S - t).pow(2).sum(1).mean() / t.pow(2).sum(1).mean())}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--text-weights", required=True, help="npy [20, 1152]: the deployed zero-shot text embeddings")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--layer-decay", type=float, default=0.8)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--ema", type=float, default=0.9995)
    ap.add_argument("--w-mse", type=float, default=1.0)
    ap.add_argument("--w-cos", type=float, default=1.0)
    ap.add_argument("--w-zs", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--grad-checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--component-repeat", type=int, default=2, help="in-domain images per epoch, vs 1 for generic")
    ap.add_argument("--smoke", action="store_true", help="~512 train / 128 val images, to check the loop runs")
    ap.add_argument("--init", help="final.pt of an earlier run to start from")
    ap.add_argument("--extra-pool", help="second pool trained on alongside --pool (paths may be absolute)")
    ap.add_argument("--extra-teacher", help="teacher cache of --extra-pool")
    ap.add_argument("--extra-repeat", type=int, default=1, help="visits per epoch of each --extra-pool image")
    a = ap.parse_args(argv)

    from timm.optim import param_groups_layer_decay
    from timm.utils import ModelEmaV3

    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, T, view_rows, views = load_source(Path(a.pool), Path(a.teacher))
    n_main, mu, sd = len(rows), T.float().mean(0), T.float().std(0)
    val = np.array([is_pool_val(r["id"]) for r in rows])
    cached = np.array([len(v) > 0 for v in views])
    comp = np.array([r["domain"] == "component" for r in rows])
    train_imgs = np.concatenate([np.flatnonzero(~val & cached)] + [np.flatnonzero(~val & cached & comp)] *
                                (a.component_repeat - 1))
    val_imgs = np.flatnonzero(val & cached)
    if a.extra_pool:  # e.g. the training-fold dev images of one fold (fold_pool.py); never used as pool-val
        xr, xT, xvr, xv = load_source(Path(a.extra_pool), Path(a.extra_teacher))
        rows, views = rows + xr, views + xv
        view_rows = view_rows + [v + len(T) for v in xvr]
        T = torch.cat([T, xT])
        train_imgs = np.concatenate([train_imgs] + [np.arange(n_main, len(rows))] * a.extra_repeat)
    if a.smoke:
        train_imgs, val_imgs = train_imgs[:: len(train_imgs) // 512], val_imgs[:128]
    dev = torch.device("cuda")
    W = torch.from_numpy(np.load(a.text_weights)).float().to(dev)
    mu_d, sd_d = mu.to(dev), sd.to(dev)

    kw = {"drop_path_rate": a.drop_path} if a.drop_path else {}
    model, pre = build_student(a.student, **kw)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model"])
    if a.grad_checkpointing:
        model.backbone.set_grad_checkpointing(True)
    model.to(dev)
    groups = param_groups_layer_decay(model.backbone, weight_decay=a.wd, layer_decay=a.layer_decay)
    for g in groups:
        g["lr"] = a.lr * g.pop("lr_scale", 1.0)
    groups.append({"params": list(model.proj.parameters()), "lr": a.head_lr, "weight_decay": a.wd})
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    ema = ModelEmaV3(model, decay=a.ema)

    ds = StudentViews(rows, train_imgs, view_rows, views, pre, a.seed)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=True, drop_last=True,
                    persistent_workers=False)
    val_dl = DataLoader(StudentViews(rows, val_imgs, view_rows, views, pre, a.seed, eval_view=True),
                        batch_size=128, num_workers=a.workers)
    total = a.epochs * len(dl)
    start_epoch, step = 0, 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        ema.module.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        print(f"resumed at epoch {start_epoch}", flush=True)
    (out / "config.json").write_text(json.dumps({**vars(a), "preprocess": pre, "n_train": len(train_imgs),
                                                 "n_val": len(val_imgs), "steps": total}, indent=1))
    log = open(out / "log.jsonl", "a")
    for epoch in range(start_epoch, a.epochs):
        model.train()
        ds.epoch = epoch
        t0, agg = time.time(), {"loss": 0.0, "mse": 0.0, "cos": 0.0, "zs": 0.0}
        for x, r in dl:
            lr_f = min(1.0, (step + 1) / a.warmup_steps) * 0.5 * (1 + np.cos(np.pi * min(step / total, 1.0)))
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * lr_f
            t = T[r].to(dev, non_blocking=True).float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s = model(x.to(dev, non_blocking=True))
            loss, parts = distill_loss(s.float(), t, mu_d, sd_d, W, a.w_mse, a.w_cos, a.w_zs)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model, step=step)
            step += 1
            agg["loss"] += loss.item()
            for k, v in parts.items():
                agg[k] += v
            if step % 200 == 0:
                print(f"ep {epoch} step {step}/{total} loss {float(loss):.4f} {parts} "
                      f"{(time.time() - t0) / (step - epoch * len(dl)) * 1000:.0f} ms/step", flush=True)
        rec = {"epoch": epoch, "step": step, "train_s": round(time.time() - t0),
               **{f"train_{k}": v / len(dl) for k, v in agg.items()},
               "val_raw": fidelity(model, val_dl, T, dev), "val_ema": fidelity(ema.module, val_dl, T, dev)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        torch.save({"model": model.state_dict(), "ema": ema.module.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "step": step}, out / "last.tmp")
        (out / "last.tmp").rename(out / "last.pt")
    last = json.loads(open(out / "log.jsonl").read().splitlines()[-1])
    pick = "ema" if last["val_ema"]["cos_mean"] >= last["val_raw"]["cos_mean"] else "raw"
    ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    torch.save({"model": ck["ema" if pick == "ema" else "model"], "student": a.student, "preprocess": pre,
                "dim": TEACHER_DIM, "picked": pick, "val": last[f"val_{pick}"]}, out / "final.pt")
    print(f"final.pt <- {pick} weights, pool-val fidelity {last[f'val_{pick}']}", flush=True)


if __name__ == "__main__":
    main()
