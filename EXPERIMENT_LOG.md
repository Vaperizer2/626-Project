# Experiment log (v2)

Chronological record of decisions and results for the rebuilt pipeline. The pre-v2 log is archived at
[`legacy/docs/EXPERIMENT_LOG.md`](legacy/docs/EXPERIMENT_LOG.md); its numbers are **not comparable** (duplicate
leakage between train and val/test — see [`legacy/README.md`](legacy/README.md)).

## Evaluation contract

1. **Data:** `data/v2/manifest.json` (version `v2.0`). Every result records the manifest `fingerprint`; a
   result produced on a different fingerprint is not comparable.
2. **Model selection:** grouped 5-fold cross-validation on the `dev` split only. Groups are near-duplicate
   `cluster`s, so copies of one image never sit on both sides of a fold. Report the mean ± std of the five
   fold scores (primary metric: macro-F1; secondary: accuracy).
3. **Sealed test:** the `test` split is read only through `compclass.data.get_test(..., confirm=True)` /
   `evaluate --sealed-test`, **once**, for the final chosen model(s), after all choices are frozen.
   Any later look at test is logged here as such.
4. **Machine-readable record:** every run appends one JSON line to `results/runs.jsonl` (config hash, fold,
   params, train time, peak VRAM, throughput, metrics). This log holds the human interpretation.

## Entries

| Date | Experiment | CV macro-F1 (mean ± std) | CV accuracy | Notes |
|---|---|---:|---:|---|
| 2026-10-17 | Package scaffold (`src/compclass`), GPU witness on carlpc2 | — | — | Infrastructure only; no dataset results yet. |
