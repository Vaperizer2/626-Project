"""THE one look at the sealed test split for the packaged Small model (exports/mobileclip2_s3), as compclass.final test.

    uv run python tools/views/test_small.py --confirm [--out exports/mobileclip2_s3]

Scores the exact exported pipeline (head.npz: probe on all dev rows + zero-shot head through the checkpoint's
projection, fused, temperature, abstain threshold) on the test rows' cached features and writes
results/final_test_mobileclip2_s3.json and _preds.npz, the same fields as compclass.final's cmd_test. Run once;
record it in EXPERIMENT_LOG.md; never rescore.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "src")
from compclass.data import get_test, load_manifest
from compclass.evaluate import compute_metrics
from compclass.features import extract_features
from compclass.final import ece, log_softmax


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm", action="store_true", help="acknowledge spending the sealed-test look")
    ap.add_argument("--out", default="exports/mobileclip2_s3")
    ap.add_argument("--manifest", default="data/v2/manifest.json")
    a = ap.parse_args()
    m = load_manifest(a.manifest)
    rows = get_test(m, confirm=a.confirm)
    out = Path(a.out)
    head = dict(np.load(out / "head.npz"))
    key = json.load(open(out / "preprocess.json"))["model"]
    F = extract_features(key, rows, m, num_workers=0).astype(np.float32)
    y = np.array([m.class_to_idx[r["label"]] for r in rows])
    lp = log_softmax(((F - head["mu"]) / head["sd"]) @ head["coef"].T + head["intercept"])
    e = F @ head["proj_w"].T + head["proj_b"]; e = e / np.linalg.norm(e, axis=1, keepdims=True)
    fused = lp + float(head["w_zs"]) * log_softmax(float(head["t_zs"]) * e @ head["W"].T)
    probs = np.exp(log_softmax(fused / float(head["temperature"])))
    probe_only = lp.argmax(1)
    pred, conf = probs.argmax(1), probs.max(1)
    met = compute_metrics(y, pred, m.classes)
    ood = np.array(["ood_suspect" in r.get("flags", []) for r in rows])
    thr = float(head["abstain_threshold"]); acc_ = conf >= thr
    rep = {"model": key, "export": str(out), "manifest": a.manifest, "fingerprint": m.fingerprint, "n": len(rows), **met,
           "probe_only_accuracy": float((probe_only == y).mean()),
           "probe_only_macro_f1": compute_metrics(y, probe_only, m.classes)["macro_f1"],
           "inscope_n": int((~ood).sum()),
           "inscope_accuracy": float((pred[~ood] == y[~ood]).mean()) if (~ood).any() else None,
           "inscope_macro_f1": compute_metrics(y[~ood], pred[~ood], m.classes)["macro_f1"] if (~ood).any() else None,
           "abstain_threshold": thr, "coverage": float(acc_.mean()),
           "accepted_accuracy": float((pred[acc_] == y[acc_]).mean()) if acc_.any() else None, "ece": ece(probs, y)}
    name = key
    Path("results").mkdir(exist_ok=True)
    Path("results") / f"final_test_{name}.json"
    (Path("results") / f"final_test_{name}.json").write_text(json.dumps(rep, indent=1, default=float))
    np.savez(Path("results") / f"final_test_{name}_preds.npz", ids=np.array([r["id"] for r in rows]), y=y, probs=probs)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("per_class", "confusion")}, indent=1, default=float))


if __name__ == "__main__":
    main()
