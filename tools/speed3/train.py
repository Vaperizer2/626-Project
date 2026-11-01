"""Label-free feature distillation of the MLP-pruned student s294 onto the frozen unpruned SigLIP So400m (carlpc2).

    train.py <out_dir> --data dev|pool|devout:K [--teacher t294|t384] [--train mlp|fc2|lora:R|lora_all:R]
             [--init STATE.npz] [--steps 1500 --batch 12 --lr 2e-5] [--w-cos 1 --w-mse 1 --w-tok 0]

Every step renders a fresh augmented view of each image (tools/prune/views.py; view 0, the harness transform, with
probability --p-view0) at 294 px for the student and at the teacher's size for the teacher, same crop. The loss uses
only the teacher's outputs on that view:
  cos  1 - cosine of the pooled embeddings (the direction the zero-shot text head reads)
  mse  squared error of the per-dimension standardised embeddings (the space the refitted probe reads); mean and sd
       from the teacher on 256 images of the training source
  tok  1 - cosine per token of every block's output (t294 only: same 21x21 grid), mean over tokens and blocks
  zs   KL from the teacher's to the student's distribution under the fixed zero-shot text head the deployed model
       fuses (exports/final head.npz W, T = 50): the 20 class-prompt directions of SigLIP's text space. Teacher
       outputs only; no image label enters.
Trainable: `mlp` fc1+fc2 of all 27 blocks, `fc2` only, `lora:R` rank-R adapters on fc1/fc2 (`lora_all:R` also on
qkv/proj), merged into the weights at the end so shapes stay exactly s294's. --qat runs the student through
fakequant.py's replica of the speed2 INT8 recipe (straight-through rounding), recalibrated on the pool calibration
images every --recal steps, so it learns to match the fp32 teacher through the INT8 error it will be deployed with.
Writes out_dir/state.npz (the float weights; tools/speed3/remote_int8.sh quantises them with the real recipe).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from common import GRAPHS, block_state, graph_size, load_block_state, loader, source, trunk

HEAD, T_ZS = Path("/home/carl/626-runs/distill/v2meta/head.npz"), 50.0  # byte-identical copy of exports/final/head.npz


class LoRA(nn.Module):
    """W + B A as a Linear-like module (weight, bias, in_features), so fakequant.QLinear can wrap it."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base, self.in_features = base, base.in_features
        self.A = nn.Parameter(torch.randn(rank, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight + self.B @ self.A

    @property
    def bias(self) -> torch.Tensor:
        return self.base.bias

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        self.base.weight.copy_(self.weight)
        return self.base


def trainable(m: nn.Module, how: str) -> list[nn.Parameter]:
    for p in m.parameters():
        p.requires_grad_(False)
    kind, _, rank = how.partition(":")
    names = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2") if kind == "lora_all" else \
        ("mlp.fc2",) if kind == "fc2" else ("mlp.fc1", "mlp.fc2")
    params = []
    for b in m.blocks:
        for n in names:
            parent, leaf = n.split(".")
            lin = getattr(b.get_submodule(parent), leaf)
            if kind.startswith("lora"):
                lora = LoRA(lin, int(rank))
                setattr(b.get_submodule(parent), leaf, lora)
                params += [lora.A, lora.B]
            else:
                lin.weight.requires_grad_(True)
                lin.bias.requires_grad_(True)
                params += [lin.weight, lin.bias]
    return params


def merge(m: nn.Module) -> None:
    for b in m.blocks:
        for parent in (b.attn, b.mlp):
            for leaf, mod in list(parent.named_children()):
                if isinstance(mod, LoRA):
                    setattr(parent, leaf, mod.merged())


def run(m: nn.Module, x: torch.Tensor, ckpt: bool) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Pooled embedding and every block's output tokens."""
    h = m.norm_pre(m._pos_embed(m.patch_embed(x)))
    outs = []
    for blk in m.blocks:
        h = checkpoint(blk, h, use_reentrant=False) if ckpt else blk(h)
        outs.append(h)
    return m.forward_head(m.norm(h)), outs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--data", required=True)
    ap.add_argument("--teacher", choices=("t294", "t384"), default="t294")
    ap.add_argument("--train", default="mlp")
    ap.add_argument("--init", type=Path)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--w-cos", type=float, default=1.0)
    ap.add_argument("--w-mse", type=float, default=1.0)
    ap.add_argument("--w-tok", type=float, default=0.0)
    ap.add_argument("--w-zs", type=float, default=0.0)
    ap.add_argument("--p-view0", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--qat", action="store_true")
    ap.add_argument("--recal", type=int, default=250)
    ap.add_argument("--bf16-base", action="store_true",
                    help="frozen weights in bf16 on the GPU; trained weights or LoRA deltas go into fp32 ones at the end")
    a = ap.parse_args()
    assert not (a.w_tok and a.teacher != "t294"), "token loss needs the teacher on the student's grid"
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    s = trunk(GRAPHS["s294"])
    if a.init:
        load_block_state(s, a.init)
    params = trainable(s, a.train)
    s.to(dev).train()
    if a.bf16_base:
        for p in s.parameters():
            if not p.requires_grad:
                p.data = p.data.bfloat16()
    if a.qat:
        import fakequant

        fakequant.quantize(s, fakequant.calibrate(s, bf16=a.bf16_base))
    t = trunk(GRAPHS[a.teacher]).to(dev, torch.bfloat16)
    sizes = (294, graph_size(GRAPHS[a.teacher]))
    items = source(a.data)
    rng = np.random.default_rng(a.seed)
    n_items = a.steps * a.batch
    order = np.concatenate([rng.permutation(len(items)) for _ in range(n_items // len(items) + 1)])[:n_items]
    views = np.where(rng.random(n_items) < a.p_view0, 0, rng.integers(1, 2**31, n_items))

    with torch.no_grad():
        k = rng.permutation(len(items))[:256]
        T0 = []
        for _, xt in loader([items[j] for j in k], np.zeros(len(k), int), sizes, 32, a.workers):
            T0.append(t(xt.to(dev, torch.bfloat16)).float())
        T0 = torch.cat(T0)
        mu, sd = T0.mean(0), T0.std(0)
        x0 = xt[:4].to(dev, torch.bfloat16)
        assert F.cosine_similarity(run(t, x0, ckpt=False)[0].float(), t(x0).float()).min() > 0.9999, "run() != forward"

    Wt = torch.from_numpy(np.load(HEAD)["W"]).float().to(dev)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0, fused=True)
    dl = loader([items[j] for j in order], views, sizes, a.batch, a.workers)
    cfg = {**{k: str(v) for k, v in vars(a).items()}, "n_trainable": sum(p.numel() for p in params),
           "n_source_images": len(items), "n_views": n_items}
    (a.out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg), flush=True)
    log = open(a.out / "log.jsonl", "w")
    t0, agg = time.time(), {"loss": 0.0, "cos": 0.0, "mse": 0.0, "tok": 0.0, "zs": 0.0}
    for step, (xs, xt) in enumerate(dl):
        f = min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        for g in opt.param_groups:
            g["lr"] = a.lr * f
        xs, xt = xs.to(dev, non_blocking=True), xt.to(dev, torch.bfloat16, non_blocking=True)
        with torch.no_grad():
            te, ttok = run(t, xt, ckpt=False) if a.w_tok else (t(xt), [])
            te = te.float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            se, stok = run(s, xs, ckpt=True)
        se = se.float()
        cos = 1 - F.cosine_similarity(se, te, dim=1).mean()
        mse = ((se - te) / sd).pow(2).mean()
        loss = a.w_cos * cos + a.w_mse * mse
        tok = torch.zeros((), device=dev)
        if a.w_tok:
            tok = torch.stack([1 - F.cosine_similarity(u.float(), v.float(), dim=-1).mean()
                               for u, v in zip(stok, ttok)]).mean()
            loss = loss + a.w_tok * tok
        zs = torch.zeros((), device=dev)
        if a.w_zs:
            ls = F.log_softmax(T_ZS * F.normalize(se, dim=1) @ Wt.T, dim=1)
            lt = F.log_softmax(T_ZS * F.normalize(te, dim=1) @ Wt.T, dim=1)
            zs = F.kl_div(ls, lt, log_target=True, reduction="batchmean")
            loss = loss + a.w_zs * zs
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if a.qat and (step + 1) % a.recal == 0 and step + 1 < a.steps:
            fakequant.quantize(s, fakequant.calibrate(s, bf16=a.bf16_base))
        for k_, v_ in (("loss", loss), ("cos", cos), ("mse", mse), ("tok", tok), ("zs", zs)):
            agg[k_] += float(v_)
        if (step + 1) % 50 == 0:
            rec = {"step": step + 1, **{k_: round(v_ / 50, 6) for k_, v_ in agg.items()}, "lr": a.lr * f,
                   "s_per_step": round((time.time() - t0) / (step + 1), 3),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            agg = dict.fromkeys(agg, 0.0)
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
    if a.qat:
        fakequant.unwrap(s)
    if a.bf16_base:
        lins = [(i, n, b.get_submodule(n)) for i, b in enumerate(s.blocks)
                for n in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")]
        deltas = {(i, n): (mod.B @ mod.A).detach().float().cpu() for i, n, mod in lins if isinstance(mod, LoRA)}
        trained = {(i, n): (mod.weight.detach().float().cpu(), mod.bias.detach().float().cpu()) for i, n, mod in lins
                   if not isinstance(mod, LoRA) and mod.weight.requires_grad}
        s = trunk(GRAPHS["s294"])
        if a.init:
            load_block_state(s, a.init)
        with torch.no_grad():
            for (i, n), d in deltas.items():
                s.blocks[i].get_submodule(n).weight += d
            for (i, n), (w, b) in trained.items():
                s.blocks[i].get_submodule(n).weight.copy_(w)
                s.blocks[i].get_submodule(n).bias.copy_(b)
    else:
        merge(s)
    np.savez(a.out / "state.npz", **block_state(s))
    print(f"{a.out}/state.npz written, {round(time.time() - t0)} s", flush=True)


if __name__ == "__main__":
    main()
