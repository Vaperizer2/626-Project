"""Closed-form recovery of the MLP-pruned student: per block, refit fc2 by least squares on unlabeled images.

    compensate.py <out_state.npz> --data pool|dev|devout:K [--n 512] [--target self|teacher] [--damp 0.01]
                  [--select flap|obs]

The student is s294 (MLP 4304 -> 3840 per block, kept neurons and fc1 unchanged). Block by block, on the student's
own residual stream over N view-0 images at 294 px (441 tokens each):
  self     regress the full teacher MLP output (all 4304 neurons on the student's input) on the 3840 kept GELU
           outputs and a constant. This is bias compensation generalised to a linear map: the dropped neurons are
           predicted from the kept ones. Only the block's own MLP is the teacher.
  teacher  regress what the unpruned t294 block adds at that depth (teacher block output minus the student's
           post-attention stream) on the same features, so fc2 also absorbs drift from the blocks before it.
Ridge: fc2 = (C_rk + l W0)(C_kk + l I)^-1 with W0 the self solution, l = damp x mean diag(C_kk). Then the next
block sees the corrected stream. --select obs (self only) also re-chooses which 464 neurons to drop, greedily by the
optimal-brain-surgeon saliency ||W2[:, j]||^2 / [C^-1]_jj on the same covariance with the exact rank-1 updates, so
the kept set is the one the least-squares refit can best stand in for; fc1 then takes the teacher's rows of that set. Writes every block Linear (qkv, proj,
fc1 as in s294) so `tools/prune/build.py load` can write the state into the s294 graph.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from common import GRAPHS, block_state, loader, source, trunk

CHUNK = 16


def stream0(m, X: torch.Tensor) -> torch.Tensor:
    return m.norm_pre(m._pos_embed(m.patch_embed(X)))


@torch.no_grad()
def obs_keep(C: torch.Tensor, W2: torch.Tensor, n_keep: int, lam: torch.Tensor) -> np.ndarray:
    Hi = torch.linalg.inv(C + lam * torch.eye(len(C), dtype=C.dtype, device=C.device))
    W = W2.clone()
    alive = torch.ones(len(C), dtype=torch.bool, device=C.device)
    for _ in range(len(C) - n_keep):
        sal = (W ** 2).sum(0) / Hi.diagonal()
        sal[~alive] = torch.inf
        j = int(sal.argmin())
        W -= torch.outer(W[:, j], Hi[j]) / Hi[j, j]
        Hi -= torch.outer(Hi[:, j], Hi[j]) / Hi[j, j]
        alive[j] = False
    return torch.nonzero(alive).flatten().cpu().numpy()


def compensate(s, t, kept: list[np.ndarray], images: torch.Tensor, target: str, damp: float,
               select: str = "flap") -> list[dict]:
    """Only the current block pair and one chunk live on the GPU (carlpc2's GPU is shared); streams stay in pinned
    host memory."""
    dev = torch.device("cuda")
    for mod in (s.patch_embed, s.norm_pre):
        mod.to(dev)
    s.pos_embed.data = s.pos_embed.data.to(dev)
    xs = torch.cat([stream0(s, images[j:j + CHUNK].to(dev)).cpu() for j in range(0, len(images), CHUNK)]).pin_memory()
    xt = xs.clone() if target == "teacher" else None
    log = []
    for i, (bs, bt) in enumerate(zip(s.blocks, t.blocks)):
        bs.to(dev)
        bt.to(dev)
        K = torch.from_numpy(kept[i]).to(dev)
        D = torch.from_numpy(np.setdiff1d(np.arange(bt.mlp.fc1.out_features), kept[i])).to(dev)
        n, d = 0, bt.mlp.fc1.out_features
        S_hh = torch.zeros(d, d, dtype=torch.float64, device=dev)
        s_h = torch.zeros(d, dtype=torch.float64, device=dev)
        if target == "teacher":
            S_rk = torch.zeros(bt.mlp.fc2.out_features, len(K), dtype=torch.float64, device=dev)
            s_r = torch.zeros(bt.mlp.fc2.out_features, dtype=torch.float64, device=dev)
        for j in range(0, len(xs), CHUNK):
            x = xs[j:j + CHUNK].to(dev, non_blocking=True)
            a = x + bs.attn(bs.norm1(x))
            h = bt.mlp.act(bt.mlp.fc1(bs.norm2(a))).flatten(0, 1)
            S_hh += (h.T @ h).double()
            s_h += h.sum(0).double()
            n += len(h)
            if target == "teacher":
                r = (bt(xt[j:j + CHUNK].to(dev, non_blocking=True)) - a).flatten(0, 1)
                S_rk += (r.T @ h[:, K]).double()
                s_r += r.sum(0).double()
        mu = s_h / n
        C = S_hh / n - torch.outer(mu, mu)
        if select == "obs":
            lam0 = damp * C.diagonal().mean()
            k_new = obs_keep(C, bt.mlp.fc2.weight.double(), len(K), lam0)
            rec_sel = {"kept_overlap_flap": round(len(np.intersect1d(k_new, kept[i])) / len(K), 4)}
            kept[i] = k_new
            K = torch.from_numpy(k_new).to(dev)
            D = torch.from_numpy(np.setdiff1d(np.arange(d), k_new)).to(dev)
            bs.mlp.fc1.weight.copy_(bt.mlp.fc1.weight[K])
            bs.mlp.fc1.bias.copy_(bt.mlp.fc1.bias[K])
        Ckk = C[K][:, K]
        lam = damp * Ckk.diagonal().mean()
        W2 = bt.mlp.fc2.weight.double()
        b2 = bt.mlp.fc2.bias.double()
        # self: regress the dropped neurons on the kept ones (B = C_dk C_kk^-1), fold W2_d B into the kept columns
        L = torch.linalg.cholesky(Ckk + lam * torch.eye(len(K), dtype=torch.float64, device=dev))
        B = torch.cholesky_solve(C[D][:, K].T, L).T
        W_self = W2[:, K] + W2[:, D] @ B
        b_self = b2 + W2[:, D] @ (mu[D] - B @ mu[K])
        if target == "self":
            W, b = W_self, b_self
        else:
            C_rk = S_rk / n - torch.outer(s_r / n, mu[K])
            W = torch.cholesky_solve((C_rk + lam * W_self).T, L).T
            b = s_r / n - W @ mu[K]
        old_w, old_b = bs.mlp.fc2.weight.double().clone(), bs.mlp.fc2.bias.double().clone()
        bs.mlp.fc2.weight.copy_(W.float())
        bs.mlp.fc2.bias.copy_(b.float())
        rel = float((W - old_w).norm() / old_w.norm())
        cs = []
        for j in range(0, len(xs), CHUNK):
            if target == "teacher":
                xt[j:j + CHUNK] = bt(xt[j:j + CHUNK].to(dev)).cpu()
            xs[j:j + CHUNK] = bs(xs[j:j + CHUNK].to(dev)).cpu()
            if target == "teacher":
                cs.append(torch.nn.functional.cosine_similarity(xs[j:j + CHUNK].flatten(0, 1),
                                                                xt[j:j + CHUNK].flatten(0, 1), dim=1))
        rec = {"block": i, "tokens": n, "lambda": float(lam), "fc2_rel_change": round(rel, 4),
               "fc2_bias_change": round(float((b - old_b).norm()), 4), **(rec_sel if select == "obs" else {})}
        if target == "teacher":
            rec["stream_cos_mean"] = round(float(torch.cat(cs).mean()), 5)
        log.append(rec)
        print(json.dumps(rec), flush=True)
        bs.cpu()
        bt.cpu()
    s.cpu()
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--target", choices=("self", "teacher"), default="self")
    ap.add_argument("--damp", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select", choices=("flap", "obs"), default="flap")
    a = ap.parse_args()
    assert a.select == "flap" or a.target == "self", "obs selection is defined for the self target"
    torch.backends.cuda.matmul.allow_tf32 = False
    items = source(a.data)
    pick = np.sort(np.random.default_rng(a.seed).permutation(len(items))[:a.n])
    items = [items[k] for k in pick]
    s, t = trunk(GRAPHS["s294"]), trunk(GRAPHS["t294"])
    kept = [np.asarray(v) for v in json.loads((GRAPHS["s294"] / "prune.json").read_text())["stats"]["kept"].values()]
    assert len(kept) == len(s.blocks) and all(len(k) == b.mlp.fc1.out_features for k, b in zip(kept, s.blocks))
    for k, bs, bt in zip(kept, s.blocks, t.blocks):
        assert torch.equal(bs.mlp.fc1.weight, bt.mlp.fc1.weight[torch.from_numpy(k)]), "kept set mismatch"
    images = torch.cat([x for (x,) in loader(items, np.zeros(len(items), int), (294,), 32)])
    t0 = time.time()
    log = compensate(s, t, kept, images, a.target, a.damp, a.select)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **block_state(s))
    meta = {**{k: str(v) for k, v in vars(a).items()}, "n_images": len(items), "seconds": round(time.time() - t0),
            "blocks": log, "kept": [k.tolist() for k in kept]}
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    print(f"{a.out} written ({meta['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
