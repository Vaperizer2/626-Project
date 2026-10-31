"""The deployed SigLIP So400m trunk rebuilt in torch from the baseline ONNX weights, with token merging.

Mirrors the baseline graph op for op (pre-LN blocks, 16 heads of 72 with 72**-0.25 on q and k, erf GELU,
LayerNorm eps 1e-6, MAP attention pooling), so with no merging it reproduces the baseline embedding.

Merging is ToMe's bipartite soft matching (Bolya et al., ICLR 2023): after a block's attention, split the tokens
into alternating sets A and B, match each A token to its most similar B token by cosine of the head-averaged keys,
and fold the r best-matched A tokens into their B partners as a size-weighted mean. Later attention adds log(size)
to the logits (proportional attention), the MAP pooling included. The scatter in the reference implementation
becomes GatherElements plus a one-hot [B, r] matmul, ops TensorRT runs without falling back, and the size weights
are fractions (each <= 1) so fp16 cannot overflow on SigLIP's large residual activations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
from onnx import numpy_helper

HEADS, HEAD_DIM, WIDTH, EPS = 16, 72, 1152, 1e-6
SCALE = HEAD_DIM ** -0.25


def onnx_weights(path: Path) -> dict[str, np.ndarray]:
    """Named weights of the baseline graph. MatMul weights are stored [in, out] under '<layer>.w', keyed by the
    bias their output is added to; the MAP head's latent query is the 'expand' initializer."""
    m = onnx.load(path)
    g = m.graph
    arr = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    producer = {o: n for n in g.node for o in n.output}
    out = {k: v for k, v in arr.items() if k.startswith("trunk.")}
    for n in g.node:
        if n.op_type == "Add" and n.input[1].startswith("trunk.") and n.input[1].endswith(".bias"):
            mm = producer.get(n.input[0])
            if mm is not None and mm.op_type == "MatMul":
                out[n.input[1][:-len(".bias")] + ".w"] = arr[mm.input[1]]
    out["trunk.attn_pool.latent"] = arr["expand"]
    return {k[len("trunk."):]: v for k, v in out.items()}


def resample_pos(pos: torch.Tensor, grid: int) -> torch.Tensor:
    """timm.resample_abs_pos_embed: bicubic, antialiased, on the square token grid."""
    old = int(pos.shape[1] ** 0.5)
    if old == grid:
        return pos
    p = pos.reshape(1, old, old, WIDTH).permute(0, 3, 1, 2)
    p = F.interpolate(p, size=(grid, grid), mode="bicubic", antialias=True, align_corners=False)
    return p.permute(0, 2, 3, 1).reshape(1, grid * grid, WIDTH)


class Trunk(torch.nn.Module):
    """pixel_values [N,3,S,S] -> pooled embedding [N,1152]. ``merge[i]`` tokens are merged after block i's
    attention; token counts are static per block, so the exported graph has static shapes past the batch."""

    def __init__(self, w: dict[str, np.ndarray], size: int, merge: tuple[int, ...]):
        super().__init__()
        self.depth = sum(1 for k in w if k.startswith("blocks.") and k.endswith(".norm1.weight"))
        assert len(merge) == self.depth, f"merge schedule needs {self.depth} entries"
        self.grid, self.merge = size // 14, merge
        for k, v in w.items():
            t = torch.from_numpy(v.copy())
            if k == "pos_embed":
                t = resample_pos(t, self.grid)
            self.register_buffer(k.replace(".", "__"), t)

    def p(self, name: str) -> torch.Tensor:
        return getattr(self, name.replace(".", "__"))

    def lin(self, x, name):
        return x @ self.p(name + ".w") + self.p(name + ".bias")

    def ln(self, x, name):
        return F.layer_norm(x, (WIDTH,), self.p(name + ".weight"), self.p(name + ".bias"), EPS)

    def mlp(self, x, name):
        return self.lin(F.gelu(self.lin(x, name + ".fc1")), name + ".fc2")

    def attend(self, q, k, v, logsize):
        s = (q * SCALE) @ (k * SCALE).transpose(-1, -2)
        if logsize is not None:
            s = s + logsize[:, None, None, :]
        return s.softmax(-1) @ v

    def attn(self, x, t, i, logsize):
        qkv = self.lin(x, f"blocks.{i}.attn.qkv").reshape(-1, t, 3, HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = self.attend(q, k, v, logsize).transpose(1, 2).reshape(-1, t, WIDTH)
        return self.lin(o, f"blocks.{i}.attn.proj"), k.mean(1)

    def forward(self, pixel_values):
        t = self.grid * self.grid
        x = F.conv2d(pixel_values, self.p("patch_embed.proj.weight"), self.p("patch_embed.proj.bias"), stride=14)
        x = x.reshape(-1, WIDTH, t).transpose(1, 2) + self.p("pos_embed")
        size = None
        for i, r in enumerate(self.merge):
            logsize = None if size is None else size[..., 0].log()
            a, metric = self.attn(self.ln(x, f"blocks.{i}.norm1"), t, i, logsize)
            x = x + a
            if r:
                if size is None:
                    size = torch.ones_like(x[..., :1])
                x, size = bipartite_merge(x, size, metric, t, r)
                t -= r
            x = x + self.mlp(self.ln(x, f"blocks.{i}.norm2"), f"blocks.{i}.mlp")
        x = self.ln(x, "norm")
        kv = self.lin(x, "attn_pool.kv").reshape(-1, t, 2, HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
        q = self.lin(self.p("attn_pool.latent"), "attn_pool.q").reshape(1, 1, HEADS, HEAD_DIM).transpose(1, 2)
        o = self.attend(q, kv[0], kv[1], None if size is None else size[..., 0].log())
        o = self.lin(o.transpose(1, 2).reshape(-1, 1, WIDTH), "attn_pool.proj")
        o = o + self.mlp(self.ln(o, "attn_pool.norm"), "attn_pool.mlp")
        return o[:, 0]


def bipartite_merge(x, size, metric, t: int, r: int):
    """ToMe merge of r tokens out of t (static ints). x holds size-weighted token means, size their counts."""
    tb = t // 2
    assert 0 < r <= (t + 1) // 2

    def take(v, idx):
        return v.gather(1, idx[..., None].expand(-1, -1, v.shape[-1]))

    m = metric / metric.norm(dim=-1, keepdim=True)
    node_max, node_idx = (m[:, ::2] @ m[:, 1::2].transpose(-1, -2)).max(-1)
    order = node_max.argsort(-1, descending=True)
    src, unm = order[:, :r], order[:, r:]
    dst = node_idx.gather(1, src)
    to_dst = (dst[..., None] == torch.arange(tb)[None, None]).to(x.dtype).transpose(1, 2)
    xa, xb, sa, sb = x[:, ::2], x[:, 1::2], size[:, ::2], size[:, 1::2]
    s_src = take(sa, src)
    s_dst = sb + to_dst @ s_src
    merged = xb * (sb / s_dst) + to_dst @ (take(xa, src) * (s_src / take(s_dst, dst)))
    return torch.cat([take(xa, unm), merged], 1), torch.cat([take(sa, unm), s_dst], 1)


def export(model: Trunk, size: int) -> onnx.ModelProto:
    """TorchScript-based export (the harness venv has no onnxscript), example batch 2 so the batch stays symbolic."""
    import io

    f = io.BytesIO()
    torch.onnx.export(model, (torch.zeros(2, 3, size, size),), f, dynamo=False, opset_version=18,
                      input_names=["pixel_values"], output_names=["embedding"],
                      dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}})
    m = onnx.load_from_string(f.getvalue())
    m.ir_version = 10
    return m
