"""Legacy audit evidence survives rebuilding without the untracked original split file."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_dataset_v2", ROOT / "tools" / "build_dataset_v2.py")
dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset)


def test_rebuild_preserves_all_recorded_legacy_assignments_and_leakage_flags():
    rows = json.loads((ROOT / "data/v2/manifest.json").read_text())["rows"]
    previous = {r["id"]: r for r in rows}
    for row in rows:
        split, leaked = dataset.legacy_row_metadata(row, {}, previous)
        assert split == row["legacy_split"]
        assert leaked == ("legacy_multi_split" in row["flags"])


def test_legacy_entries_identify_copies_across_splits():
    unit = {"id": "image", "src_paths": ["a.jpg", "b.jpg"]}
    assert dataset.legacy_row_metadata(unit, {"a.jpg": "test", "b.jpg": "train"}, {}) == ("train", True)
    assert dataset.legacy_row_metadata(unit, {}, {}) == (None, False)


def test_matching_historical_leakage_evidence_is_preserved():
    rows = json.loads((ROOT / "data/v2/manifest.json").read_text())["rows"]
    previous = json.loads((ROOT / "data/v2/leakage_report.json").read_text())
    audit = dataset.legacy_audit(rows, 0, 0, None, previous)
    assert audit["val_test_vs_train"] == previous["legacy_split"]["val_test_vs_train"]
    assert audit["matched_entries"] == previous["legacy_split"]["matched_entries"]
    assert audit["rows_with_legacy_split"] == previous["legacy_split"]["rows_with_legacy_split"]
    assert audit["evidence_source"] == "preserved_previous_report"


def test_missing_or_mismatched_evidence_is_unavailable_instead_of_zero():
    unavailable = dataset.legacy_audit([], 0, 0, None, {})
    for previous in ({}, {"fingerprint": "another_dataset", "legacy_split": {"matched_entries": 99}},
                     {"fingerprint": dataset.fingerprint([]), "legacy_split": unavailable}):
        audit = dataset.legacy_audit([], 0, 0, None, previous)
        assert audit["matched_entries"] is None
        assert audit["unmatched_entries"] is None
        assert audit["val_test_vs_train"] is None
        assert audit["evidence_source"] == "unavailable_without_legacy_split"


def test_recomputed_legacy_evidence_replaces_the_previous_report():
    audit = dataset.legacy_audit([], 3, 1, {"val": {"entries": 1}}, {})
    assert audit == {"matched_entries": 3, "unmatched_entries": 1, "rows_with_legacy_split": {},
                     "val_test_vs_train": {"val": {"entries": 1}}}
