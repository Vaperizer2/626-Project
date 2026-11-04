"""Token-merge schedules of the So400m trunk, their wave counts on the RTX 3060 Laptop, and predicted latency.

A schedule is written ``<px>:<block>=<tokens>,...``, for example ``308:13=384,20=256``: 308 px input (22x22 = 484
patch tokens), ToMe-merged down to 384 tokens right after block 13's attention and to 256 after block 20's. Block
i's attention sees the count before its merge and its MLP the count after (tools/tokens/siglip.py). The id form
replaces ``:`` and ``,`` (``s308_13t384_20t256``).

    sched.py id <spec>                             # the schedule's id form
    sched.py waves [T ...]                         # CTAs and waves per role at each token count
    sched.py predict <costs.json> <spec> ...       # predicted ms of layers and harness ratio per schedule

The wave model (speed2 NOTES): legion's GPU has 30 SMs. TensorRT's INT8 GEMMs here run 128x128 tiles, two CTAs per
SM (60 slots); attn.proj runs 64x64 tiles, three per SM (90 slots); the fused attention runs one CTA per head and
64-query tile (60 slots), each CTA looping over all T keys. A GEMM costs about one CTA time per wave, so latency
steps when ceil(T / tile) crosses a multiple of the slot count, not in proportion to FLOPs.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

DEPTH, PATCH, HEADS, WIDTH, MLP = 27, 14, 16, 1152, 4304
SMS = 30


@dataclass(frozen=True)
class Schedule:
    size: int
    steps: tuple[tuple[int, int], ...] = ()

    @classmethod
    def parse(cls, spec: str) -> "Schedule":
        spec = spec.removeprefix("s")
        if "_" in spec and ":" not in spec:
            px, *rest = spec.split("_")
            spec = px + ":" + ",".join(r.replace("t", "=") for r in rest)
        px, _, rest = spec.partition(":")
        steps = tuple(tuple(int(x) for x in s.split("=")) for s in rest.split(",") if s)
        s = cls(int(px), tuple(sorted(steps)))
        s.merge()
        return s

    @property
    def tokens0(self) -> int:
        return (self.size // PATCH) ** 2

    @property
    def spec(self) -> str:
        return f"{self.size}:" + ",".join(f"{b}={t}" for b, t in self.steps)

    @property
    def id(self) -> str:
        return f"s{self.size}" + "".join(f"_{b}t{t}" for b, t in self.steps)

    def merge(self) -> tuple[int, ...]:
        """Tokens merged after each block's attention (the siglip.Trunk merge tuple)."""
        r, t = [0] * DEPTH, self.tokens0
        for b, after in self.steps:
            assert 0 <= b < DEPTH and after < t, f"{self.spec}: block {b} must reduce {t} tokens"
            assert t - after <= (t + 1) // 2, f"{self.spec}: bipartite merge removes at most half of {t} tokens"
            r[b], t = t - after, after
        return tuple(r)

    def mlp_tokens(self) -> list[int]:
        t, out = self.tokens0, []
        for r in self.merge():
            t -= r
            out.append(t)
        return out

    def attn_tokens(self) -> list[int]:
        return [self.tokens0] + self.mlp_tokens()[:-1]


def gemm_waves(t: int, n: int, tile: int = 128, slots: int = 2 * SMS) -> tuple[int, int]:
    ctas = math.ceil(t / tile) * math.ceil(n / tile)
    return ctas, math.ceil(ctas / slots)


def waves(t: int, mlp: int = MLP) -> dict[str, tuple[int, int]]:
    """(CTAs, waves) per role at T tokens. fc2's 9 output tiles never fill a wave; its cost is one CTA's K loop,
    doubled where more than 30 CTAs put two on some SMs."""
    return {"qkv": gemm_waves(t, 3 * WIDTH), "proj": gemm_waves(t, WIDTH, 64, 3 * SMS),
            "fc1": gemm_waves(t, mlp), "fc2": gemm_waves(t, WIDTH, slots=SMS),
            "mha": (HEADS * math.ceil(t / 64), math.ceil(HEADS * math.ceil(t / 64) / (2 * SMS)))}


@dataclass
class Costs:
    """Per-block costs measured on a ladder probe (fit_costs.py): attention side A(T) (qkv, fused attention, proj),
    MLP side plus everything else M(T), merge ops G(T) (by the count before the merge), and the fixed rest."""
    A: dict[int, float]
    M: dict[int, float]
    G: dict[int, float]
    fixed: dict
    harness: dict

    @classmethod
    def load(cls, p: Path) -> "Costs":
        d = json.loads(p.read_text())
        return cls(*({int(k): v for k, v in d[x].items()} for x in "AMG"), d["fixed"], d["harness"])

    @staticmethod
    def at(table: dict[int, float], t: int) -> float:
        """Exact entry, else linear between the neighbours that share t's wave counts (the model's claim is that
        cost is piecewise linear in T inside one wave class), else the nearest same-class entry scaled by T."""
        if t in table:
            return table[t]
        cls = waves(t)
        same = sorted(k for k in table if waves(k) == cls)
        lo = [k for k in same if k < t]
        hi = [k for k in same if k > t]
        if lo and hi:
            a, b = lo[-1], hi[0]
            return table[a] + (table[b] - table[a]) * (t - a) / (b - a)
        if same:
            k = min(same, key=lambda k: abs(k - t))
            return table[k] * t / k
        raise KeyError(f"no measured token count shares the wave class of {t}")

    def ms(self, s: Schedule) -> float:
        tot = self.fixed["f0"] + self.fixed["f1"] * s.mlp_tokens()[-1]
        for b, (ti, to) in enumerate(zip(s.attn_tokens(), s.mlp_tokens())):
            tot += self.at(self.A, ti) + self.at(self.M, to) + (self.at(self.G, ti) if ti != to else 0.0)
        return tot

    def ratio(self, s: Schedule) -> float:
        """Harness paired ratio vs fp32, from a line through the reference candidates' (probe ms, harness ratio)."""
        h = self.harness
        return h["slope"] * self.ms(s) + h["intercept"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("waves")
    w.add_argument("tokens", nargs="*", type=int, default=[484, 448, 441, 416, 384, 352, 320, 288, 256, 224, 192,
                                                            160, 128])
    w.add_argument("--mlp", type=int, default=MLP)
    sub.add_parser("id").add_argument("spec")
    p = sub.add_parser("predict")
    p.add_argument("costs", type=Path)
    p.add_argument("specs", nargs="+")
    a = ap.parse_args()
    if a.cmd == "id":
        print(Schedule.parse(a.spec).id)
        return
    if a.cmd == "waves":
        for t in a.tokens:
            print(f"T={t:4d} " + "  ".join(f"{k} {c:3d}/{n}w" for k, (c, n) in waves(t, a.mlp).items()))
        return
    c = Costs.load(a.costs)
    for spec in a.specs:
        s = Schedule.parse(spec)
        mt = s.mlp_tokens()
        print(f"{s.id:32s} ms {c.ms(s):6.3f}  ratio {c.ratio(s):.4f}  mean tokens {sum(mt) / DEPTH:.0f}")


if __name__ == "__main__":
    main()
