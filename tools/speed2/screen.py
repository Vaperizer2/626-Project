"""Label-free screen: cosine of each graph's embedding to a reference graph's on pool ``sens`` images (CUDA EP,
TF32 off, so float graphs run in fp32 and QDQ graphs simulate INT8). Never development images. A screen, not a
result; the harness decides.

    screen.py <ref_dir> <graph_dir> [<graph_dir> ...] [--n 64]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quant"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "compose"))

from common import pool_images, preprocess, spec_of  # noqa: E402
from sanity import agreement, embed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", type=Path)
    ap.add_argument("graphs", type=Path, nargs="+")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    import onnxruntime as ort

    ort.preload_dlls()
    X = preprocess(pool_images("sens", a.n), spec_of(a.ref))
    ref = embed(a.ref, X)
    rep = {"ref": str(a.ref), "n": len(X)}
    for g in a.graphs:
        assert spec_of(g) == spec_of(a.ref), f"{g}: preprocessing differs from the reference"
        rep[str(g)] = agreement(ref, embed(g, X))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
