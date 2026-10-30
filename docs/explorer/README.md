# Interactive project explorer

`docs/explorer/index.html` is a single, self-contained page for going through what this project did and for
learning the model families it compares. Everything is inline (CSS, vanilla JavaScript, JSON data, base64 JPEG
thumbnails): no CDN, no external fonts, no network access. It works offline from `file://` and inside a sandboxed
iframe.

## Open it

```bash
xdg-open docs/explorer/index.html        # Linux; or double-click the file / drag it into any browser
```

Deep links work: `index.html#bench`, `index.html#families/vit`, `index.html#final&theme=dark`.
Keyboard: Tab to the tab bar, then ←/→/Home/End switch sections; buttons, chips, fold blocks and the confusion-matrix
row labels are focusable (Enter activates).

## What is in it

| Tab | Content |
|---|---|
| Overview | goal, the three headline numbers, the four research questions answered, caveats, clickable project timeline (what / why / result / files / commits from `git log`) |
| Data audit | cleaning funnel, per-class out-of-scope share, distractor concepts; galleries of out-of-scope images, exact copies across legacy splits, near-duplicate pairs, the largest near-duplicate cluster, conflicting labels, the 69 label issues, typical examples |
| Evaluation protocol | clickable split diagram (sealed test, 5 grouped folds, web holdout, original files), toy leakage simulation (file-level vs cluster-level split), measured legacy-split inflation, legacy-protocol replay vs the old notebooks |
| Benchmark explorer | every row of `results/benchmark.csv` with per-fold dots; filters (family, regime, schedule, metric); sortable table; accuracy-vs-cost scatter with selectable axis and Pareto front; probe-vs-fine-tune paired view; training curves; per-class F1 heatmap |
| Ablations | one card per ablation (probe vs fine-tune, partial fine-tune, LR-schedule fix, flip TTA, data cleaning, zero-shot fusion, leakage, compute) with question, result, honest interpretation and source files |
| Final model | data-flow diagram, model card, live abstention slider (coverage / accepted accuracy recomputed from the embedded per-image probabilities), interactive confusion matrix with image drill-down, reliability diagram with adjustable bins, most-confident mistakes, deployment numbers |
| Generalisation | the two sealed sets side by side, per-class recall test vs web, web-holdout error gallery with credits, the original-Kaggle evaluation with the seen / never-seen breakdown and legacy replay |
| Model families | one page per family (classical, scratch CNN, ResNet, EfficientNet, MobileNet, ConvNeXt, ViT, Swin, DINOv2, CLIP, SigLIP/SigLIP 2): core idea and papers, an architecture diagram, a step-through of tensor shapes for the exact model used here, characteristics, measured results and training curves, the effective config with repo keys and what each knob does, failure modes, commands to try, references, and the full `docs/models/*.md` explainer |
| Glossary / Reproduce | searchable glossary (69 terms); exact commands per phase |

## Rebuild

Opening the committed HTML requires no dataset. Rebuilding its embedded image galleries requires the ignored
`data/v2/images/` and `data/web_holdout_v1/images/` directories from the original datasets; a fresh clone does
not contain those images. Restore them using the project README before rebuilding. The tracked review CSVs and
result files provide the audit/benchmark numbers.

```bash
uv run --no-config scripts/build_explorer.py          # writes docs/explorer/index.html (CPU, ~10 s)
uv run --no-config scripts/build_explorer.py --check  # same, plus a size breakdown
```

The script has PEP 723 inline dependencies (numpy, pandas, pillow, pyyaml), so `uv run` builds its own small
environment and does not touch the project's torch environment. It reads only repository files:
`results/` (benchmark, per-fold run records, final test JSON + `*_preds.npz`, Kaggle-original summary), `exports/final/`,
`data/v2/` (manifest, leakage report, review CSVs, images), `data/web_holdout_v1/` (manifest, images),
`data/kaggle_eval_extra/images`, `docs/report/tables/`, `docs/models/*.md`, `configs/*.yaml` and `git log`.

Sources of the page itself:

| File | Role |
|---|---|
| `scripts/build_explorer.py` | data extraction, thumbnails, Markdown rendering of `docs/models/*.md`, assertions, assembly |
| `docs/explorer/src/content.py` | hand-written prose: timeline, family explainers (idea, walk-through shapes, characteristics), config-key explanations, glossary, reproduce commands |
| `docs/explorer/src/app.js` | the application (tabs, charts and diagrams drawn as inline SVG) |
| `docs/explorer/src/style.css` | light/dark theme and layout |

## Number provenance and checks

No result is typed into the page by hand except a few headline values in the timeline prose, and the build asserts
those against their source files. The build fails if any of these disagree:

* sealed-test and web-holdout accuracy, macro-F1, confusion matrix, coverage, accepted accuracy and ECE recomputed
  from the embedded (5-decimal) probabilities vs `results/final/final_test_*.json`; Wilson CI vs `key_numbers.json`;
* `results/benchmark.csv` vs the mean of the per-fold records (max difference < 1e-4);
* probe-vs-fine-tune deltas and win counts vs `docs/report/tables/probe_vs_finetune.csv`; schedule-ablation deltas,
  improved-fold count and SD vs `results/schedule_ablation_folds012.csv` / `key_numbers.json`; TTA deltas;
* audit counts (3,530 / 2,943 / 2,930 / 2,787 / 816 / 418 / 2,512), per-class OOD counts, distractor counts;
* original-Kaggle numbers (95.5 % all, 83.0 % on 493 never-seen files, 37.0 % conflicting, replay 87.7 % vs 71.3 %);
* every commit hash in the timeline exists in `git log`.

Architecture shapes come from `docs/models/*.md` (checked against timm 1.0.30). SigLIP So400m/14@384 values (patch
14, width 1152, depth 27, 16 heads, mlp_ratio 3.7362, MAP pooling, no [CLS]) were read from timm 1.0.30
`vision_transformer.py`. FLOPs appear only where the original paper reports them, labelled as such.

## QA performed

`node --check` on the script; a jsdom run that opens every tab and drives the main interactions (filters, sort,
scatter axes, slider, confusion-matrix cells, galleries, glossary search, theme) with zero JavaScript errors; headless
screenshots of every tab and family page (Zen/Firefox `--headless --screenshot`) in light and dark mode, at 1400 px
and 1024 px width; and a load inside `<iframe sandbox="allow-scripts">` (no same-origin, so no localStorage or
history access) with no errors. Screenshots: `docs/explorer/screenshots/`.

Thumbnails of the web holdout are Wikimedia Commons images under CC/PD licences; each carries its author, licence
and source URL in the page (hover), and full credits are in `data/web_holdout_v1/ATTRIBUTION.md`.
