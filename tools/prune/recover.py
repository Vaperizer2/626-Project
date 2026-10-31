"""Recover a pruned So400m by feature distillation onto the deployed teacher's cached pool embeddings (carlpc2).

    recover.py check     --candidate DIR                   # cosine to the teacher cache on pool-val eval views
    recover.py calibrate --candidate DIR --out ACT.npz     # mean |GELU output| per MLP neuron, component pool images
    recover.py train     --candidate DIR --out RUN [--steps 1800 --batch 16 --lr 3e-5 --train mlp]

The student is DIR's graph weights in timm's `vit_so400m_patch14_siglip_384` (MLP widths taken from DIR). Data is
the distill track's pool (disjoint from every data/v2 and web-holdout image) through its cached views, each with
the So400m-384 embedding of the deployed trunk as the target. Pool-val images are never trained on. Weights that
are zero in DIR stay zero (2:4 / unstructured masks). `train` writes RUN/state.npz for `build.py load`.
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
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import so400m  # noqa: E402
from views import distill_loss, is_pool_val, load_pool, render_view, to_tensor  # noqa: E402

D = Path("/home/carl/626-runs/distill")
POOL, TEACHER, TEXT = D / "pool", D / "teacher_v8g2", D / "v2meta/text_weights.npy"
SIZE, MEAN, STD = 384, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
GROUPS = {"mlp": ("mlp.fc1", "mlp.fc2"), "attn": ("attn.qkv", "attn.proj")}


def build(cand: Path) -> nn.Module:
    import timm

    sd = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in
          so400m.Weights(so400m.load(cand)).torch_state().items()}
    m = timm.create_model("vit_so400m_patch14_siglip_384", pretrained=False, num_classes=0)
    for i, b in enumerate(m.blocks):
        h = sd[f"blocks.{i}.mlp.fc1.weight"].shape[0]
        if h != b.mlp.fc1.out_features:
            b.mlp.fc1, b.mlp.fc2 = nn.Linear(b.mlp.fc1.in_features, h), nn.Linear(h, b.mlp.fc2.out_features)
    m.load_state_dict(sd, strict=True)
    return m


class Pool:
    """Pool rows, cached teacher embeddings T [views, 1152] and, per image, its cached (view, T row) pairs."""

    def __init__(self):
        self.rows, _ = load_pool(POOL)
        idx = np.load(TEACHER / "index.npz")
        assert list(idx["ids"]) == [r["id"] for r in self.rows], "teacher cache does not match the pool"
        self.T = torch.from_numpy(np.concatenate([np.load(s) for s in sorted(TEACHER.glob("shard_[0-9][0-9][0-9].npy"))]))
        assert len(self.T) == len(idx["items"]), "teacher cache incomplete"
        self.views: list[list[tuple[int, int]]] = [[] for _ in self.rows]
        for t_row, (k, v) in enumerate(idx["items"]):
            self.views[k].append((int(v), t_row))
        self.val = np.array([is_pool_val(r["id"]) for r in self.rows])
        self.comp = np.array([r["domain"] == "component" for r in self.rows])


class Views(Dataset):
    """Item i: image `images[i]` through cached view `pick(i)` -> (pixels, teacher row)."""

    def __init__(self, pool: Pool, images: np.ndarray, eval_view: bool, seed: int = 0):
        self.pool, self.images, self.eval_view, self.seed = pool, images, eval_view, seed

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        k = int(self.images[i])
        vs = self.pool.views[k]
        v, t_row = vs[0] if self.eval_view else vs[np.random.default_rng((self.seed, i)).integers(len(vs))]
        row = self.pool.rows[k]
        im = Image.open(POOL / row["path"]).convert("RGB")
        return to_tensor(render_view(im, row["id"], v, SIZE), MEAN, STD), t_row


@torch.no_grad()
def fidelity(m: nn.Module, pool: Pool, images: np.ndarray, dev, workers: int) -> dict:
    m.eval()
    S, R = [], []
    for x, r in DataLoader(Views(pool, images, eval_view=True), batch_size=32, num_workers=workers):
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            S.append(m(x.to(dev)).float().cpu())
        R.append(r)
    S, t = torch.cat(S), pool.T[torch.cat(R)].float()
    cos = nn.functional.cosine_similarity(S, t, dim=1)
    return {"n": len(images), "cos_mean": round(float(cos.mean()), 5), "cos_p05": round(float(cos.quantile(0.05)), 5),
            "rel_mse": round(float((S - t).pow(2).sum(1).mean() / t.pow(2).sum(1).mean()), 5)}


def val_images(pool: Pool, n: int | None) -> np.ndarray:
    imgs = np.flatnonzero(pool.val)
    return imgs[:n] if n else imgs


def cmd_check(a, dev) -> None:
    pool = Pool()
    print(json.dumps(fidelity(build(a.candidate).to(dev), pool, val_images(pool, a.n), dev, a.workers)))


@torch.no_grad()
def cmd_calibrate(a, dev) -> None:
    pool = Pool()
    m = build(a.candidate).to(dev).eval()
    imgs = np.random.default_rng(0).permutation(np.flatnonzero(pool.comp & ~pool.val))[:a.n]
    acc = [torch.zeros(b.mlp.fc1.out_features, dtype=torch.float64, device=dev) for b in m.blocks]
    def hook(i):
        def f(mod, inp, out):
            acc[i].add_(out.abs().float().sum((0, 1)))
        return f

    for i, b in enumerate(m.blocks):
        b.mlp.act.register_forward_hook(hook(i))
    n_tok = 0
    for x, _ in DataLoader(Views(pool, imgs, eval_view=True), batch_size=32, num_workers=a.workers):
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            m(x.to(dev))
        n_tok += x.shape[0] * (SIZE // 14) ** 2
    act = torch.stack(acc).cpu().numpy() / n_tok
    np.savez(a.out, mean_abs_act=act, image_ids=np.array([pool.rows[k]["id"] for k in imgs]))
    print(f"{a.out}: {len(imgs)} component pool images; mean |act| quantiles per block (p10, p50, p90): "
          f"{np.round(np.quantile(act, [0.1, 0.5, 0.9], axis=1).T[[0, 13, 26]], 4).tolist()}")


def cmd_train(a, dev) -> None:
    torch.manual_seed(a.seed)
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    pool = Pool()
    mu, sd = pool.T.float().mean(0).to(dev), pool.T.float().std(0).to(dev)
    W = torch.from_numpy(np.load(TEXT)).float().to(dev)
    m = build(a.candidate)
    names = tuple(n for g in a.train.split(",") for n in GROUPS[g])
    for p in m.parameters():
        p.requires_grad_(False)
    params, masks = [], {}
    for i, b in enumerate(m.blocks):
        for n in names:
            lin = b.get_submodule(n)
            for p in (lin.weight, lin.bias):
                p.requires_grad_(True)
                params.append(p)
            zero = lin.weight == 0
            if zero.float().mean() > 0.01:
                masks[lin.weight] = ~zero
    m.to(dev)
    masks = {p: mk.to(dev) for p, mk in masks.items()}  # Module.to() keeps Parameter identity
    m.set_grad_checkpointing(True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
    train = np.flatnonzero(~pool.val)
    train = np.concatenate([train] + [np.flatnonzero(~pool.val & pool.comp)] * (a.component_repeat - 1))
    order = np.random.default_rng(a.seed).permutation(train)[:a.steps * a.batch]
    dl = DataLoader(Views(pool, order, eval_view=False, seed=a.seed), batch_size=a.batch, shuffle=False,
                    num_workers=a.workers, pin_memory=True, drop_last=True)
    val = val_images(pool, a.val_n)
    log = open(out / "log.jsonl", "a")
    rec0 = {"step": 0, "val": fidelity(m, pool, val, dev, a.workers)}
    (out / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(a).items()}, "n_trainable":
                                                 sum(p.numel() for p in params), "n_masked": len(masks),
                                                 "n_train_items": len(order), "n_val": len(val)}, indent=1))
    print(json.dumps(rec0), flush=True)
    log.write(json.dumps(rec0) + "\n")
    m.train()
    t0, step, agg = time.time(), 0, {"loss": 0.0, "mse": 0.0, "cos": 0.0, "zs": 0.0}
    for x, r in dl:
        f = min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + np.cos(np.pi * step / a.steps))
        for g in opt.param_groups:
            g["lr"] = a.lr * f
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            s = m(x.to(dev, non_blocking=True))
        loss, parts = distill_loss(s.float(), pool.T[r].to(dev).float(), mu, sd, W, 1.0, 1.0, 0.5)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for p, mk in masks.items():
            p.grad.mul_(mk)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        with torch.no_grad():
            for p, mk in masks.items():
                p.mul_(mk)
        step += 1
        agg["loss"] += loss.item()
        for k, v in parts.items():
            agg[k] += v
        if step % 50 == 0:
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
            rec = {"step": step, **{k: round(v / 50, 5) for k, v in agg.items()}, "lr": a.lr * f,
                   "s_per_step": round(el / step, 3), "mem_gb": round(mem, 2)}
            agg = dict.fromkeys(agg, 0.0)
            if step % a.val_every == 0:
                rec["val"] = fidelity(m, pool, val, dev, a.workers)
                m.train()
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            if el > a.max_minutes * 60:
                print(f"stopping at step {step}: time budget {a.max_minutes} min", flush=True)
                break
        if step >= a.steps:
            break
    rec = {"step": step, "train_s": round(time.time() - t0), "val": fidelity(m, pool, val_images(pool, None), dev,
                                                                                a.workers)}
    print(json.dumps(rec), flush=True)
    log.write(json.dumps(rec) + "\n")
    m.cpu()
    state = {f"blocks.{i}.{n}.{p}": getattr(b.get_submodule(n), p).detach().numpy()
             for i, b in enumerate(m.blocks) for n in (*GROUPS["mlp"], *GROUPS["attn"]) for p in ("weight", "bias")}
    for p, mk in masks.items():
        assert bool((p.detach()[~mk.cpu()] == 0).all()), "a masked weight moved"
    np.savez(out / "state.npz", **state)
    print(f"{out}/state.npz written", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "calibrate", "train"):
        p = sub.add_parser(name)
        p.add_argument("--candidate", type=Path, required=True)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--cpu", action="store_true")
    sub.choices["check"].add_argument("--n", type=int, default=256)
    sub.choices["calibrate"].add_argument("--out", type=Path, required=True)
    sub.choices["calibrate"].add_argument("--n", type=int, default=512)
    t = sub.choices["train"]
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=1800)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--train", default="mlp")
    t.add_argument("--component-repeat", type=int, default=2)
    t.add_argument("--val-n", type=int, default=256)
    t.add_argument("--val-every", type=int, default=300)
    t.add_argument("--max-minutes", type=float, default=60)
    t.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cpu" if a.cpu else "cuda")
    {"check": cmd_check, "calibrate": cmd_calibrate, "train": cmd_train}[a.cmd](a, dev)


if __name__ == "__main__":
    main()
