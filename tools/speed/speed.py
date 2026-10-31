"""Frozen speed harness for the So400m classifier hillclimb. See README.md in this directory.

    tools/speed/speed export-baseline
    tools/speed/speed eval <candidate_dir>
    tools/speed/speed time <candidate_dir>
    tools/speed/speed compare <reference_dir> <candidate_dir>
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export-baseline", help="deployed model -> candidates/baseline_fp32")
    sub.add_parser("eval", help="dev-fold CV accuracy through the candidate's graph").add_argument("candidate")
    t = sub.add_parser("time", help="batch-1 latency and batch-8 throughput on the GPU")
    t.add_argument("candidate")
    c = sub.add_parser("compare", help="accuracy delta, McNemar, latency ratio, verdict")
    c.add_argument("reference")
    c.add_argument("candidate")
    a = ap.parse_args()
    if a.cmd == "export-baseline":
        from baseline import export_baseline

        export_baseline()
    elif a.cmd == "eval":
        from evaluate import run_eval

        run_eval(a.candidate)
    elif a.cmd == "time":
        from timing import run_time

        run_time(a.candidate)
    else:
        from compare import run_compare

        run_compare(a.reference, a.candidate)


if __name__ == "__main__":
    main()
