"""Export the fixed ITRI626 train/validation/test split of data v2 as CSV files (stdlib + pandas only).

    uv run python tools/export_splits.py            # writes data/v2/splits/*.csv
    uv run python tools/export_splits.py --check    # recompute and compare with the files on disk

The split is NOT re-randomised here. It is read from ``data/v2/manifest.json`` (built once by
tools/build_dataset_v2.py: exact-duplicate collapse, near-duplicate clustering, then a cluster-grouped
StratifiedGroupKFold with seed 42 that put one outer fold aside as the sealed test set and cut the rest
into 5 development folds):

* train      = dev rows with ``fold`` in 1..4
* validation = dev rows with ``fold`` == 0
* test       = rows with ``split`` == "test" (the sealed test set; 418 images)

This is exactly ``compclass.data.get_fold(manifest, 0)`` plus ``get_test``. The script re-computes the
manifest's split fingerprint, checks that no duplicate cluster crosses the train/validation/test boundary,
and writes

* ``data/v2/splits/{train,val,test}.csv``  id, path (relative to data/v2/), label, cluster, fold, ood_suspect
* ``data/v2/splits/split_summary.csv``     per class: raw files, unique after exact de-duplication, files in
  excluded conflicting-label groups, final rows, train/val/test counts, OOD-flagged rows (+ a TOTAL row)
* ``data/v2/splits/split_info.json``       seed, fingerprint, sizes, the rule above, provenance of the counts

Raw per-class file counts come from ``data/raw/electronic-component-recognizer/<class>/<class>/*`` when that
folder exists (all files, any suffix, hidden files skipped) and are cross-checked against the manifest's
``src_paths`` + ``excluded`` records; without the raw folder the manifest records alone are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "v2" / "manifest.json"
RAW = ROOT / "data" / "raw" / "electronic-component-recognizer"
OUT = ROOT / "data" / "v2" / "splits"
EXPECTED_FINGERPRINT = "431315e87d2612cb"
VAL_FOLD = 0
COLUMNS = ["id", "path", "label", "cluster", "fold", "ood_suspect"]


def fingerprint(rows: list[dict]) -> str:
    """Same rule as compclass.data.compute_fingerprint / tools/build_dataset_v2.py."""
    lines = sorted(f"{r['id']}\t{r['label']}\t{r['split']}\t{'' if r['fold'] is None else r['fold']}" for r in rows)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]


def assign(r: dict) -> str:
    if r["split"] == "test":
        return "test"
    return "val" if r["fold"] == VAL_FOLD else "train"


def raw_counts_from_disk(classes: list[str]) -> Counter | None:
    if not RAW.is_dir():
        return None
    out = Counter()
    for c in classes:
        d = RAW / c / c
        out[c] = sum(1 for p in d.iterdir() if p.is_file() and not p.name.startswith(".")) if d.is_dir() else 0
    return out


def build(manifest_path: Path = MANIFEST) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    m = json.loads(manifest_path.read_text())
    rows, classes = m["rows"], m["classes"]
    fp = fingerprint(rows)
    assert fp == m["fingerprint"] == EXPECTED_FINGERPRINT, (fp, m["fingerprint"])
    assert m["split_policy"]["seed"] == 42 and m["split_policy"]["group_key"] == "cluster"

    for r in rows:
        r["_split"] = assign(r)
    # leakage guard: a near-duplicate cluster must live in exactly one of train/val/test
    home: dict[str, str] = {}
    for r in rows:
        assert home.setdefault(r["cluster"], r["_split"]) == r["_split"], f"cluster {r['cluster']} crosses splits"
    ids = Counter(r["id"] for r in rows)
    assert max(ids.values()) == 1, "duplicate row ids"
    sha = Counter(r["sha256"] for r in rows)
    assert max(sha.values()) == 1, "exact duplicate images survived de-duplication"

    frames = {}
    for s in ("train", "val", "test"):
        sel = [r for r in rows if r["_split"] == s]
        frames[s] = pd.DataFrame({
            "id": [r["id"] for r in sel], "path": [r["path"] for r in sel], "label": [r["label"] for r in sel],
            "cluster": [r["cluster"] for r in sel],
            "fold": pd.array([r["fold"] for r in sel], dtype="Int64"),
            "ood_suspect": [int("ood_suspect" in r.get("flags", [])) for r in sel]})[COLUMNS]
    for a in frames:
        for b in frames:
            if a < b:
                assert not set(frames[a]["cluster"]) & set(frames[b]["cluster"]), (a, b)

    # ---- per-class counts before / after cleaning
    # Every raw file is accounted for exactly once: either in some kept row's src_paths (identical bytes are
    # collapsed into one row) or in an excluded group (identical bytes saved under conflicting class folders).
    raw_from_manifest, excluded_files, unique_exact = Counter(), Counter(), Counter()
    for r in rows:
        unique_exact[r["label"]] += 1
        for p in r["src_paths"]:
            raw_from_manifest[p.split("/")[0]] += 1
    for e in m.get("excluded", []):
        for c in set(e["labels"]):
            unique_exact[c] += 1  # the image is one unique file per class folder it appeared in
        for p in e["src_paths"]:
            raw_from_manifest[p.split("/")[0]] += 1
            excluded_files[p.split("/")[0]] += 1
    disk = raw_counts_from_disk(classes)
    if disk is not None:
        assert disk == raw_from_manifest, {c: (disk[c], raw_from_manifest[c]) for c in classes
                                           if disk[c] != raw_from_manifest[c]}
    raw = disk if disk is not None else raw_from_manifest

    per = []
    for c in classes:
        rr = [r for r in rows if r["label"] == c]
        per.append({"class": c, "raw_files": raw[c], "unique_after_exact_dedup": unique_exact[c],
                    "excluded_conflict_files": excluded_files[c], "final_rows": len(rr),
                    "train": sum(r["_split"] == "train" for r in rr), "val": sum(r["_split"] == "val" for r in rr),
                    "test": sum(r["_split"] == "test" for r in rr),
                    "ood_flagged": sum("ood_suspect" in r.get("flags", []) for r in rr)})
    summary = pd.DataFrame(per)
    total = {"class": "TOTAL", **{k: int(summary[k].sum()) for k in summary.columns if k != "class"}}
    summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)
    # unique_after_exact_dedup counts conflicting groups once per class folder; the global unique count is lower
    n_unique_global = len(rows) + len(m.get("excluded", []))
    assert total["final_rows"] == len(rows) and total["train"] + total["val"] + total["test"] == len(rows)
    assert total["raw_files"] == sum(raw.values())

    info = {"manifest": "data/v2/manifest.json", "fingerprint": fp, "seed": m["split_policy"]["seed"],
            "method": "cluster-grouped StratifiedGroupKFold (tools/build_dataset_v2.py); outer fold -> sealed test, "
                      "remaining dev rows -> 5 folds; this split: train = dev folds 1-4, val = dev fold 0, test = sealed",
            "group_key": "cluster (exact SHA-256 duplicates collapsed; near-duplicates linked by pHash<=10 & "
                         "CLIP-B/32 cos>=0.85, or cos>=0.92)",
            "sizes": {s: len(f) for s, f in frames.items()}, "n_rows": len(rows), "n_clusters": len(home),
            "n_classes": len(classes), "classes": classes,
            "raw_files_total": int(sum(raw.values())), "unique_images_global": n_unique_global,
            "excluded_conflicting_groups": len(m.get("excluded", [])),
            "raw_counts_source": "disk (data/raw/...) cross-checked with manifest src_paths" if disk is not None
            else "manifest src_paths + excluded records (raw folder not present)",
            "ood_suspect_definition": "CLIP zero-shot flag: softmax mass on distractor concepts >= 0.25 "
                                      "(data/v2/leakage_report.json, ood_rule); flagged rows are kept"}
    return frames, summary, info


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="compare with existing files instead of writing")
    a = ap.parse_args(argv)
    frames, summary, info = build(a.manifest)
    files = {f"{s}.csv": f.to_csv(index=False, lineterminator="\n") for s, f in frames.items()}
    files["split_summary.csv"] = summary.to_csv(index=False, lineterminator="\n")
    files["split_info.json"] = json.dumps(info, indent=1) + "\n"
    if a.check:
        def same(n: str, txt: str) -> bool:
            f = a.out / n
            if not f.exists():
                return False
            if n == "split_info.json":  # the provenance note depends on whether data/raw/ exists on this machine
                strip = lambda d: {k: v for k, v in d.items() if k != "raw_counts_source"}  # noqa: E731
                return strip(json.loads(f.read_text())) == strip(json.loads(txt))
            return f.read_text() == txt

        bad = [n for n, txt in files.items() if not same(n, txt)]
        if bad:
            raise SystemExit(f"split files differ from the manifest: {bad}")
        print("split files up to date:", sorted(files))
        return
    a.out.mkdir(parents=True, exist_ok=True)
    for n, txt in files.items():
        (a.out / n).write_text(txt)
    print(json.dumps({"out": str(a.out), **info["sizes"], "fingerprint": info["fingerprint"]}))


if __name__ == "__main__":
    main()
