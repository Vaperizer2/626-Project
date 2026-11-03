"""The speed2 INT8 recipe as torch fake quantisation, for screening and quantisation-aware distillation.

tools/quant/qdq.py builds the deployed INT8 graph as: SmoothQuant alpha 0.5 on the qkv, fc1 (LayerNorm inputs) and
proj (V output) GEMMs of every block, s = cmax(x)^0.5 / max_out|W|^0.5 per input channel; activation scale
amax / 127 per tensor (max calibration, p99.999 for every `mlp.fc2` role); weights int8 symmetric per output channel.
All 112 GEMMs: 27 x (qkv, proj, fc1, fc2) and the attention pool's kv, proj, mlp.fc1, mlp.fc2. Here the same
statistics come from torch hooks on the same 512 pool calibration images (tools/quant/calib_set.py, never dev), and
each GEMM runs on the integer grid: round(clamp(x / s / sx)) @ round(W s / sw) * sx * sw + b, with straight-through
gradients for the rounding.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import CALIB, loader

ALPHA, TOPK, P_FC2 = 0.5, 1024, 99.999


def calib_items(n: int = 512) -> list[tuple[str, Path]]:
    rows = json.loads((CALIB / "split.json").read_text())["calib"][:n]
    return [(r["id"], CALIB / r["path"]) for r in rows]


def gemms(m: nn.Module) -> dict[str, nn.Module]:
    """Role -> the module computing that GEMM (a Linear, or a QLinear wrapping one)."""
    out = {}
    for i, b in enumerate(m.blocks):
        for n in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            out[f"blocks.{i}.{n}"] = b.get_submodule(n)
    for n in ("kv", "proj", "mlp.fc1", "mlp.fc2"):
        out[f"attn_pool.{n}"] = m.attn_pool.get_submodule(n)
    return out


def smoothed(role: str) -> bool:
    return role.startswith("blocks.") and not role.endswith("mlp.fc2")


@torch.no_grad()
def calibrate(m: nn.Module, n: int = 512, bf16: bool = False) -> dict[str, dict]:
    """Per role: smoothing factors s ([in], ones if unsmoothed) and the activation amax, from fp32 hooks (bf16 autocast
    for a model whose frozen weights are bf16)."""
    lins = {r: (mod.lin if isinstance(mod, QLinear) else mod) for r, mod in gemms(m).items()}
    cmax_img, top = {r: [] for r in lins}, {r: [] for r in lins}
    numel = {}
    hooks = []
    for r, lin in lins.items():
        def f(mod, inp, r=r):
            x = inp[0].float().abs()
            cmax_img[r].append(x.flatten(1, -2).amax(1))  # [B, C]
            top[r].append(x.flatten(1).topk(min(TOPK, x[0].numel()), dim=1).values)
            numel[r] = x[0].numel()
        hooks.append(lin.register_forward_pre_hook(f))
    was = {r: mod for r, mod in gemms(m).items() if isinstance(mod, QLinear)}
    for mod in was.values():
        mod.enabled = False
    dev = next(m.parameters()).device
    items = calib_items(n)
    for (x,) in loader(items, np.zeros(len(items), int), (294,), 16, workers=6):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            m(x.to(dev))
    for h in hooks:
        h.remove()
    for mod in was.values():
        mod.enabled = True
    out = {}
    for r, lin in lins.items():
        c = torch.cat(cmax_img[r])  # [images, C]
        t = torch.cat(top[r])
        if smoothed(r):
            a = c.amax(0).clamp_min(1e-5)
            wr = lin.weight.float().abs().amax(0).clamp_min(1e-5)  # per input channel
            s = a ** ALPHA / wr ** (1 - ALPHA)
            amax = float((c / s).amax())
        else:
            s = torch.ones(lin.in_features, device=dev)
            if r.endswith("mlp.fc2"):
                rank = int(round((1 - P_FC2 / 100) * len(t) * numel[r]))
                amax = float(t.flatten().sort(descending=True).values[rank])
            else:
                amax = float(t[:, 0].max())
        out[r] = {"s": s, "amax": amax}
    return out


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


class QLinear(nn.Module):
    def __init__(self, lin: nn.Linear, s: torch.Tensor, amax: float):
        super().__init__()
        self.lin, self.enabled = lin, True
        self.register_buffer("s", s.detach().float().clone())
        self.register_buffer("amax", torch.tensor(float(amax)))

    def forward(self, x):
        if not self.enabled:
            return self.lin(x)
        sx = self.amax / 127
        xi = RoundSTE.apply(torch.clamp(x.float() / self.s / sx, -128, 127))
        w = self.lin.weight.float() * self.s
        sw = (w.abs().amax(1).clamp_min(1e-8) / 127).detach()
        wi = RoundSTE.apply(w / sw[:, None])
        y = F.linear(xi, wi)  # integer-valued operands: exact in bf16/TF32, accumulated in fp32 like an INT8 GEMM
        return (y.float() * (sx * sw) + self.lin.bias.float()).to(x.dtype)


def quantize(m: nn.Module, cal: dict[str, dict]) -> nn.Module:
    """Wrap every GEMM in a QLinear (or refresh the scales of existing wrappers) in place."""
    for r, mod in gemms(m).items():
        if isinstance(mod, QLinear):
            mod.s.copy_(cal[r]["s"])
            mod.amax.fill_(cal[r]["amax"])
            continue
        parent_name, _, leaf = r.rpartition(".")
        parent = m.get_submodule(parent_name)
        setattr(parent, leaf, QLinear(mod, cal[r]["s"].to(mod.weight.device), cal[r]["amax"]))
    return m


def unwrap(m: nn.Module) -> nn.Module:
    for r, mod in gemms(m).items():
        if isinstance(mod, QLinear):
            parent_name, _, leaf = r.rpartition(".")
            setattr(m.get_submodule(parent_name), leaf, mod.lin)
    return m


def amax_table(cal: dict[str, dict]) -> dict[str, float]:
    return {r: round(v["amax"], 6) for r, v in cal.items()}
