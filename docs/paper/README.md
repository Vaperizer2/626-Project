# IEEE conference article

`main.tex` is the project write-up as an IEEE conference paper (IEEEtran, `\documentclass[conference]{IEEEtran}`,
two columns, 8 pages including references). It condenses `docs/report/REPORT.md` and adds the original-Kaggle
evaluation (`results/kaggle_original/summary.json`).

| File | What it is |
|---|---|
| `main.tex` | article source; a `% SOURCE:` comment above every table and figure names the repository file(s) its numbers come from |
| `refs.bib` | 47 references, generated from arXiv-API / CrossRef metadata (plus an `IEEEtranBSTCTL` entry: "et al." after 6 authors) |
| `REFERENCES_VERIFIED.md` | per-reference verification record (id, title, how verified) |
| `figures/fig1–fig6_*.pdf` | column-width (3.45 in) figures, drawn by `scripts/make_paper_figures.py` |
| `figures/paper_figure_numbers.json` | every number drawn in the new figures, with its source file |
| `main.pdf` | compiled article (build output; regenerate after any edit) |

## Build

```bash
cd docs/paper
latexmk -pdf -interaction=nonstopmode main.tex      # pdflatex + bibtex, IEEEtran.bst
latexmk -c                                          # remove aux files (keeps main.pdf)
```

Needs a TeX Live with `IEEEtran` (Times via the class), `booktabs`, `multirow`, `xcolor`, `microtype`, `cite`,
`hyperref` and `cm-super`. The last build had no errors, no overfull boxes and no undefined references or citations.

Regenerate the figures (reads only repo files; no GPU or model needed):

```bash
uv run --no-project --with numpy --with pandas --with matplotlib python scripts/make_paper_figures.py
```

The script imports the loaders, colours and checks of `scripts/make_report_figures.py`. It re-asserts that
`results/benchmark.csv` equals the per-fold run records and that the sealed-test ECE recomputed from
`results/final/final_test_v2_preds.npz` equals `final_test_v2.json`. It also sizes every figure so its bounding box is
3.45 in wide, which keeps the 8/7/6-pt font ladder at 1:1 scale in the column.

## What you must fill in

Search `main.tex` for `\placeholder`:

1. Author block: department and university, city and country, email (`\IEEEauthorblockA`).
2. Acknowledgment: supervisor, funding and similar.
3. Repository URL (end of the Acknowledgment).

Also check:

* The Kaggle dataset URL in `refs.bib` (`kaggle_dataset`). It is the only reference that could not be machine-verified.
* Whether your venue wants the arXiv-preprint entries upgraded to their published venues (list in
  `REFERENCES_VERIFIED.md`).
* The sentence on the Rust/GPUI desktop front end (Section VIII), which says it is "under development". Update it
  once `apps/gpui-demo` is finished.
* Whether your venue allows the `hyperref` package. Some IEEE PDF eXpress checks dislike link annotations; delete
  the line if so.

## Honesty caveats kept in the text (do not drop them when editing)

* Frozen probe ≥ fine-tune holds **only for SigLIP**. ImageNet, DINOv2 and CLIP-B fine-tunes beat their probes on
  5/5 folds.
* Zero-shot fusion is null in distribution (sealed test 0.837 fused vs. 0.845 probe-only) and +4.2 points on the web
  holdout. No paired test is possible, because probe-only per-image predictions were not saved.
* The schedule ablation is confounded with BatchNorm freezing.
* Out-of-scope flags come from an automated CLIP screen. The web holdout was curated by one automated reviewer, so it
  is easier than a random web sample.
* Test standard error is about ±1.8 points; one seed; one source dataset.
* The 95.5% on all original Kaggle files includes images used to fit the head. The never-seen figure is 83.0%.
* The legacy-protocol replay (87.7% vs. 71.3%) is not fully independent: the backbone and fusion were chosen on v2 dev
  folds, which contain many legacy-test images.
* OpenCV DNN cannot load the ONNX graph; onnxruntime is used.
* The leakage simulation, schedule-ablation and cleaning-ablation summaries come from analysis code that is not in the
  repository.

## Number conventions

Three-decimal values are rounded from the four-decimal values in `results/benchmark.csv` and the JSON result files,
following `docs/report/REPORT.md`. Where `docs/report/tables/benchmark_main.csv` differs by 0.001 (e.g. MobileNetV3-L
macro-F1 0.547 vs. 0.548; the per-fold mean is 0.54754), the paper uses the correctly rounded per-fold value.
