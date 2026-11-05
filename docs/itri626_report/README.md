# Current course report

`main.tex` is the authoritative course-report source and `main.pdf` is its compiled export, revised 5 November
2026 on `docs/report-polish-excalidraw` from main `17aff8b`. The proposal and marked-up Markdown under
`../model_d_inserts/` are historical review material, not the current submission. `../report/REPORT.md` describes
the broader research study, whose protocols must not be confused with the course comparison.

The course models are A (scratch CNN), B (fine-tuned ConvNeXt-T), C (frozen SigLIP + linear head) and D (frozen
MobileCLIP2-S3 + linear head). D is supplementary: A–C satisfy the named-family requirement without assuming
lecturer approval for a fourth family. The title page's supplied team details are retained.

R1 uses an alternative head on C's training features. R2 (Deployed) and R3 (Small) refit on the development set and
fuse with zero-shot heads. Small is **not D**. Fast uses SigLIP at 308 px with INT8 TensorRT; the efficiency study's
308 px fp16 timing does not measure the Fast engine. C/D ratios use the same-machine C-tower anchor, not mixed
hardware rows. Timings exclude image decoding/resizing and are not end-to-end app latency.

## Build from committed results

From the repository root:

```sh
uv sync --locked
uv run --locked python docs/itri626_report/make_tables.py
cd docs/itri626_report
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Requires a LaTeX installation with the fonts/packages in `main.tex`, latexmk and BibTeX. No model downloads,
training, raw-image access or holdout inference are required. The generator validates cached predictions against
committed metrics, recomputes descriptive bootstrap intervals/paired tests from those predictions, and records
number provenance in `generated/key_numbers.json`. Existing A–C agreement/error summaries remain scoped to A–C.
The four-model statistical table includes all six pairs; reported McNemar p-values are unadjusted.

## Diagrams and review

`diagrams/` preserves the actual Excalidraw MCP input arrays, editable `.excalidraw` scenes, SVG and PDF exports.
The report uses the vector PDFs. Open a scene in Excalidraw to edit and export SVG again; convert with
`rsvg-convert -f pdf -o workflow.pdf workflow.svg` (and likewise for `protocols`). See its README for provenance.
The other plots are committed notebook exports from `results/itri626/figures_four_models/`; the three-model
augmentation, example, agreement and disagreement figures retain captions stating their scope.

Render the final PDF with Poppler and inspect **all pages**, especially the four-model tables, confusion matrices,
page transitions, references and appendix floats. The main body budget counts the title and sections 1–7;
references and supporting appendices follow separately. `RUBRIC_CHECKLIST.md` maps the revised labels.
