"""Proxy scores beside predicted speed, and beside the harness CV delta where an earlier track measured one.

    table.py [--sort ratio|proxy] [--only-308]

Reads the proxy results fetched from carlpc2 (626-artifacts/speed4/proxy/results.jsonl, latest record per schedule)
and the cost table (626-artifacts/speed4/costs_ladder308.json). KNOWN lists harness CV deltas vs fp32 recorded by the
tokens (fp16 TensorRT), compose, trtbugs and speed2 (INT8) tracks; they calibrate the proxy and are never re-spent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sched import Costs, Schedule  # noqa: E402

ART = Path("/home/vaper/Work/Prototypes/626-artifacts/speed4")
R6 = "308:" + ",".join(f"{i}={484 - 6 * (i + 1)}" for i in range(27))
R16 = "384:" + ",".join(f"{i}={729 - 16 * (i + 1)}" for i in range(27))
KNOWN = {"336:": "-0.08 fp16", "308:": "-0.08 fp16, +0.00 int8", "294:": "-0.28 int8", "280:": "-0.76 fp16",
         "252:": "-1.51 fp16", "224:": "-3.54 fp16", "308:2=384": "-0.96 fp16", "308:9=384": "-0.84 fp16",
         "308:13=384": "-0.12 fp16", "308:13=384,20=256": "-0.28 fp16, -0.16 int8",
         "308:13=384,20=256,23=128": "-0.80 int8", R6: "-0.68 fp16", R16: "-0.68 fp16"}


def latest(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[r["id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", default="ratio", choices=["ratio", "proxy"])
    ap.add_argument("--only-308", action="store_true")
    a = ap.parse_args()
    c = Costs.load(ART / "costs_ladder308.json")
    known = {Schedule.parse(k).id: v for k, v in KNOWN.items()}
    rows = []
    for r in latest(ART / "proxy" / "results.jsonl").values():
        s = Schedule.parse(r["spec"])
        if a.only_308 and s.size != 308:
            continue
        try:
            ratio = c.ratio(s)
        except KeyError:
            ratio = float("nan")
        rows.append((ratio, r, known.get(s.id, "")))
    rows.sort(key=lambda x: (x[0] != x[0], x[0]) if a.sort == "ratio" else (0, -x[1]["cos_ref_mean"]))
    print(f"{'schedule':34s} {'pred':>6s} {'cos_ref':>7s} {'p05':>6s} {'p01':>6s} {'cos_same':>8s} {'p05':>6s} "
          f"{'agree':>6s}  harness dCV")
    for ratio, r, k in rows:
        sid = r["id"] if len(r["id"]) <= 34 else r["id"][:22] + "..." + r["id"][-9:]
        print(f"{sid:34s} {ratio:6.4f} {r['cos_ref_mean']:7.4f} {r['cos_ref_p05']:6.4f} {r['cos_ref_p01']:6.4f} "
              f"{r['cos_same_mean']:8.4f} {r['cos_same_p05']:6.4f} {r['agree_ref']:6.3f}  {k}")


if __name__ == "__main__":
    main()
