"""Prove compclass.final.fit_head reproduces the deployed head from the features it was fitted on.

    check_fit.py <dev_features.npy> <deployed_dir> <manifest.json>

<dev_features.npy> is the So400m-384 dev feature cache (torch, fp16 autocast) behind exports/final; <deployed_dir> holds
its head.npz and fit_report.json. The text embeddings W are taken from head.npz, as the fast export does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main() -> None:
    from compclass.data import get_dev, load_manifest
    from compclass.final import fit_head

    feats, deployed, manifest = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    ref = dict(np.load(deployed / "head.npz"))
    ref_rep = json.loads((deployed / "fit_report.json").read_text())
    m = load_manifest(manifest)
    head, rep = fit_head(np.load(feats), m, get_dev(m), ref["W"])
    out = {"arrays_max_abs_diff": {k: float(np.abs(np.asarray(head[k], np.float64) - ref[k]).max()) for k in ref},
           "report_diff": {k: [rep[k], ref_rep[k]] for k in ref_rep
                           if k != "abstain_table" and rep[k] != ref_rep[k]},
           "abstain_table_equal": rep["abstain_table"] == ref_rep["abstain_table"]}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
