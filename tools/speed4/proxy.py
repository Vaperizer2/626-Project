"""Label-free accuracy proxy for merge schedules: how far a schedule's fp32 embedding moves from the deployed one, on
unlabelled pool images that never touch the development folds.

    proxy.py prep [--n 1024]          # pick the proxy images, preprocess them at every size used
    proxy.py run <spec> ...           # embed with tools/tokens/siglip.Trunk on the GPU, append metrics per schedule

Images: the distillation pool (gated against every data/v2 image and the web holdout, quant/calib_set.py), component
domain, short side >= 224, minus the calibration and sensitivity splits, balanced by source, seed 1. Reference: the
deployed 384 px trunk in fp32 (the torch rebuild reproduces the baseline graph). Metrics per schedule:
cosine to the reference (mean, 5th and 1st percentile), cosine to the unmerged trunk at the same size, and top-1
agreement with the reference under the deployed fused head (exports/final/head.npz). Run on carlpc2 with the project
venv; results go to $SPEED4_PROXY/results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tokens"))
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))

from sched import Schedule  # noqa: E402

OUT = Path(os.environ.get("SPEED4_PROXY", "/home/carl/626-runs/speed4/proxy"))
POOL = Path(os.environ.get("SPEED4_POOL", "/home/carl/626-runs/distill/pool"))
CALIB_SPLIT = Path(os.environ.get("SPEED4_CALIB_SPLIT", "/home/carl/626-runs/quant/art/calib/split.json"))
BASELINE = Path(os.environ.get("TOKENS_BASELINE", "/home/carl/626-runs/harness/candidates/baseline_fp32"))
HEAD = OUT / "head.npz"


def pick(n: int) -> list[dict]:
    used = {r["id"] for k in ("calib", "sens") for r in json.loads(CALIB_SPLIT.read_text())[k]}
    rows = sorted((r for r in json.loads((POOL / "pool.json").read_text())["rows"]
                   if r["domain"] == "component" and min(r["w"], r["h"]) >= 224 and r["id"] not in used),
                  key=lambda r: r["id"])
    rng = random.Random(1)
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["source"]].append(r)
    for v in by_src.values():
        rng.shuffle(v)
    out = []
    while len(out) < n and any(by_src.values()):
        for src in sorted(by_src):
            if by_src[src] and len(out) < n:
                out.append(by_src[src].pop())
    return out


def images(size: int) -> np.ndarray:
    f = OUT / f"x{size}.npy"
    if not f.exists():
        from PIL import Image

        from compclass.data import build_transforms

        rows = json.loads((OUT / "images.json").read_text())
        spec = SimpleNamespace(input_size=size, crop_pct=0.9, interpolation="bicubic", mean=[0.5] * 3, std=[0.5] * 3)
        tf = build_transforms(spec, None, train=False)
        x = np.stack([tf(Image.open(POOL / r["path"]).convert("RGB")).numpy() for r in rows]).astype(np.float32)
        np.save(f, x)
    return np.load(f, mmap_mode="r")


def gpu_merge(x, size, metric, t: int, r: int):
    """siglip.bipartite_merge with its one-hot built on the input's device."""
    import torch

    from siglip import take

    tb = t // 2
    m = metric / metric.norm(dim=-1, keepdim=True)
    node_max, node_idx = (m[:, ::2] @ m[:, 1::2].transpose(-1, -2)).max(-1)
    order = node_max.argsort(-1, descending=True)
    src, unm = order[:, :r], order[:, r:]
    dst = node_idx.gather(1, src)
    to_dst = (dst[..., None] == torch.arange(tb, device=x.device)[None, None]).to(x.dtype).transpose(1, 2)
    xa, xb, sa, sb = x[:, ::2], x[:, 1::2], size[:, ::2], size[:, 1::2]
    s_src = take(sa, src)
    s_dst = sb + to_dst @ s_src
    merged = xb * (sb / s_dst) + to_dst @ (take(xa, src) * (s_src / take(s_dst, dst)))
    return torch.cat([take(xa, unm), merged], 1), torch.cat([take(sa, unm), s_dst], 1)


class Embedder:
    def __init__(self):
        import torch

        import siglip

        siglip.bipartite_merge = gpu_merge
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.torch, self.siglip = torch, siglip
        self.w = siglip.onnx_weights(BASELINE / "model.onnx")
        self.models: dict[int, object] = {}

    def model(self, size: int):
        if size not in self.models:
            self.models.clear()
            self.torch.cuda.empty_cache()
            self.models[size] = self.siglip.Trunk(self.w, size, (0,) * 27).eval().cuda()
        return self.models[size]

    def tail(self, m, x, size, t: int, merge: tuple[int, ...], start: int):
        """siglip.Trunk.forward from block ``start`` on, given the residual stream entering it."""
        from siglip import HEAD_DIM, HEADS, WIDTH

        for i in range(start, len(merge)):
            r = merge[i]
            a, metric = m.attn(m.ln(x, f"blocks.{i}.norm1"), t, i, None if size is None else size[..., 0].log(), r > 0)
            x = x + a
            if r:
                size = self.torch.ones_like(x[..., :1]) if size is None else size
                x, size = gpu_merge(x, size, metric, t, r)
                t -= r
            x = x + m.mlp(m.ln(x, f"blocks.{i}.norm2"), f"blocks.{i}.mlp")
        x = m.ln(x, "norm")
        kv = m.lin(x, "attn_pool.kv").reshape(-1, t, 2, HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
        q = m.lin(m.p("attn_pool.latent"), "attn_pool.q").reshape(1, 1, HEADS, HEAD_DIM).transpose(1, 2)
        o, _ = m.attend(q, kv[0], kv[1], None if size is None else size[..., 0].log())
        o = m.lin(o.transpose(1, 2).reshape(-1, 1, WIDTH), "attn_pool.proj")
        return (o + m.mlp(m.ln(o, "attn_pool.norm"), "attn_pool.mlp"))[:, 0]

    def __call__(self, scheds: list[Schedule], batch: int = 16) -> dict[str, np.ndarray]:
        """Embeddings for schedules of one input size. The unmerged prefix runs once per batch and each schedule
        branches off at its first merge block; the first batch is checked against Trunk.forward."""
        import torch.nn.functional as F

        from siglip import WIDTH

        torch = self.torch
        m = self.model(scheds[0].size)
        first = {s.id: next((b for b, r in enumerate(s.merge()) if r), 27) for s in scheds}
        x_all = images(scheds[0].size)
        out = {s.id: [] for s in scheds}
        t0 = m.grid * m.grid
        with torch.no_grad():
            for i in range(0, len(x_all), batch):
                px = torch.from_numpy(np.ascontiguousarray(x_all[i:i + batch])).cuda()
                x = F.conv2d(px, m.p("patch_embed.proj.weight"), m.p("patch_embed.proj.bias"), stride=14)
                x = x.reshape(-1, WIDTH, t0).transpose(1, 2) + m.p("pos_embed")
                for k in range(28):
                    for s in scheds:
                        if first[s.id] == k:
                            out[s.id].append(self.tail(m, x, None, t0, s.merge(), k).cpu().numpy())
                    if k < 27:
                        x = x + m.attn(m.ln(x, f"blocks.{k}.norm1"), t0, k, None, False)[0]
                        x = x + m.mlp(m.ln(x, f"blocks.{k}.norm2"), f"blocks.{k}.mlp")
                if i == 0:
                    m.merge = scheds[-1].merge()
                    d = float(np.abs(m(px).cpu().numpy() - out[scheds[-1].id][0]).max())
                    assert d < 1e-3, f"branching forward differs from Trunk.forward by {d}"
        return {k: np.concatenate(v) for k, v in out.items()}


def fused_top1(e: np.ndarray, h) -> np.ndarray:
    def lsm(z):
        z = z - z.max(1, keepdims=True)
        return z - np.log(np.exp(z).sum(1, keepdims=True))

    lp = lsm(((e - h["mu"]) / h["sd"]) @ h["coef"].T + h["intercept"])
    fn = e / np.linalg.norm(e, axis=1, keepdims=True)
    return (lp + h["w_zs"] * lsm(h["t_zs"] * fn @ h["W"].T)).argmax(1)


def cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def embeddings(emb: Embedder, scheds: list[Schedule]) -> dict[str, np.ndarray]:
    d = OUT / "emb"
    d.mkdir(parents=True, exist_ok=True)
    todo = [s for s in scheds if not (d / f"{s.id}.npy").exists()]
    for size in sorted({s.size for s in todo}, reverse=True):
        for k, v in emb([s for s in todo if s.size == size]).items():
            np.save(d / f"{k}.npy", v)
    return {s.id: np.load(d / f"{s.id}.npy") for s in scheds}


def run(specs: list[str]) -> None:
    emb = Embedder()
    h = np.load(HEAD)
    scheds = list({s.id: s for s in map(Schedule.parse, specs)}.values())
    t0 = time.time()
    e = embeddings(emb, [Schedule(384)] + [Schedule(sz) for sz in {s.size for s in scheds}] + scheds)
    ref = e["s384"]
    ref_top1 = fused_top1(ref, h)
    for s in scheds:
        c, cs = cos(e[s.id], ref), cos(e[s.id], e[Schedule(s.size).id])
        rec = {"id": s.id, "spec": s.spec, "n": len(c), "mean_tokens": float(np.mean(s.mlp_tokens())),
               "cos_ref_mean": float(c.mean()), "cos_ref_p05": float(np.percentile(c, 5)),
               "cos_ref_p01": float(np.percentile(c, 1)), "cos_same_mean": float(cs.mean()),
               "cos_same_p05": float(np.percentile(cs, 5)),
               "agree_ref": float((fused_top1(e[s.id], h) == ref_top1).mean()),
               "finite": bool(np.isfinite(e[s.id]).all())}
        with open(OUT / "results.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    print(f"{len(scheds)} schedules in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep").add_argument("--n", type=int, default=1024)
    sub.add_parser("run").add_argument("specs", nargs="+")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cmd == "prep":
        rows = pick(a.n)
        (OUT / "images.json").write_text(json.dumps(rows, indent=1))
        print(len(rows), "images from", len({r["source"] for r in rows}), "sources")
    else:
        run(a.specs)


if __name__ == "__main__":
    main()
