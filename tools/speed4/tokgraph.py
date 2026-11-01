"""Export the token-merged So400m trunk for one schedule (sched.py) with tools/tokens/build.py's exporter.

    tokgraph.py <spec> <out_parent>      # writes <out_parent>/<schedule id>/ (model.onnx, weights relinked)

Run where torch is (carlpc2), with TOKENS_BASELINE pointing at the fp32 baseline candidate.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tokens"))
sys.path.insert(0, str(HERE))

import build as tokens  # noqa: E402

from sched import Schedule  # noqa: E402

if __name__ == "__main__":
    s = Schedule.parse(sys.argv[1])
    tokens.OUT = Path(sys.argv[2])
    tokens.RECIPES[s.id] = tokens.Recipe(f"speed4 schedule {s.spec}", s.size, merge=s.merge())
    tokens.build(s.id)
