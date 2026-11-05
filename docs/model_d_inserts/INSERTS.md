# Model D: proposed additions to the ITRI626 report

From Meiring van Niekerk, 2026-11-05. For the team lead to place in `docs/itri626_report/main.tex`.

Nothing in the report has been changed on branch `Meiring`: `main.tex`, `make_tables.py`, the bibliography, the
generated tables, `main.pdf` and the report's figures in `results/itri626/figures/` are as you left them. The two
notebooks do include model D, but the three-models notebook now writes its figures to
`results/itri626/figures_four_models/`; with `WITH_D = False` in its first cell it runs on A, B and C only and writes
the report's figures as before. This file lists where I would like text added to the report and gives the text. Section numbers are those of the
current PDF. Numbers are written out (the report's macros for D do not exist yet); every one comes from
`results/itri626/metrics.json` and is listed with its source in `key_numbers_with_D.json` next to this file.

## The whole report with the additions in place

`FULL_REPORT_WITH_MODEL_D.md`, next to this file, is the complete report as Markdown with model D written in. Added
text is underlined, removed text is struck through, and a quoted note marks each changed or new table and figure.
Its figures are in `figures/` (the report's own for the unchanged ones, the four-model versions for the changed
ones, and the workflow diagram rendered from the TikZ source). It was generated from the trial build by
`build_full_report.py`; nothing under `docs/itri626_report/` was edited. The numbered list below gives the same
additions one at a time, with a core or optional mark.

## What model D is

A fourth main model trained and scored by the course pipeline exactly as model C, with a smaller backbone.

| | C (in the report) | D (proposed) |
|---|---|---|
| Backbone | SigLIP So400m/14 ViT, 384 px, frozen | MobileCLIP2-S3 (FastViT-MCi3 convolution–attention hybrid), 256 px, frozen |
| Head | `nn.Linear(1152, 20)`, 23,060 parameters | `nn.Linear(3072, 20)`, 61,460 parameters |
| Total parameters | 428,248,660 | 122,775,668 (3.5 times fewer) |
| Training | `step_head`, weight decay from the four-value grid on validation | the same step, grid and rule (`--steps head_D`) |
| Test accuracy / macro-F1 | 0.823 / 0.817 | **0.773 / 0.770** |
| 95 % bootstrap interval, accuracy | 0.787–0.859 | 0.732–0.811 |
| Validation at the selected epoch | 0.803 | 0.775 / 0.772, epoch 11 of 31 run |

Already on the branch, outside the report: `scripts/itri626_experiments.py` (steps `head_D`, `reference_D`,
`--profile-only`), `results/itri626/{metrics.json, per_class.csv, histories.json, hyperparameters.csv, efficiency.csv,
head_grid_D.csv, preds/D_*.npz, preds/R3_small_fused_test.npz, runs/D_mobileclip2_s3_linear.json}`. The entries of A,
B, C, R1 and R2 in those files were checked to be unchanged. D's head was scored on the 418 test images once.

## Three things you need to decide first

1. **Is the hybrid an acceptable fourth family?** The brief's list ends with "Swin Transformer or another approved
   image classification architecture". FastViT is a convolution–attention hybrid, not on the named list, so it needs
   the lecturer's approval. A, B and C satisfy the brief without it.
2. **Page budget.** The body is at the recommended 15 pages now. In a trial build with every insert below it grew to
   17; with the hyperparameter table moved to Appendix B it was 15.4. Inserts are marked **core** or **optional** so
   you can choose. The core set alone is about 0.6 page.
3. **Two sentences in the current text stop being true** once D is in the tables (inserts 6 and 13).

## Inserts, in report order

### 1. Abstract (core)

After "…(C) a frozen SigLIP So400m Vision Transformer with a trained linear head." add:

> A fourth model (D) repeats C's approach with a frozen MobileCLIP2-S3 hybrid that is 3.5 times smaller; it reaches
> 0.773 accuracy (macro-F1 0.770).

In the last sentence, "We recommend model C, and ConvNeXt-T where inference must be fast or run on a CPU" could become
"We recommend model C, model D where C is too heavy, and ConvNeXt-T for real-time use on a CPU."

### 2. Section 1, "Main findings" (optional)

New bullet after the second one:

> The frozen-feature strategy survives a 3.5-fold smaller backbone: model D is 5.0 points below C and 13.6 above B,
> and on the same PC 2.5 times faster than C on a GPU and 14 times on a CPU (Section 5).

### 3. Section 3, new subsection after 3.3 "Model C" (core)

> **3.4 Model D: MobileCLIP2-S3 hybrid, frozen, with a linear head (feature extraction, small backbone).**
> MobileCLIP 2 [new ref] trains image–text encoders for on-device use. Its S3 image encoder is a FastViT hybrid
> [new ref]: three convolutional stages followed by two self-attention stages, at 256×256 pixels (122,775,668
> parameters, 3.5 times fewer than C; timm weights `fastvit_mci3.apple_mclip2_dfndr2b`). It is used exactly as C:
> the backbone is frozen, the 3072-dimensional pooled feature is standardised, and one `nn.Linear(3072, 20)` layer
> (61,460 parameters) is trained with the same code, grid and selection rule. D was added after A–C to answer the
> cost half of the research question, how much of C's accuracy survives a backbone small enough for a laptop CPU.
> It was chosen on the development folds only, as the best of twelve small image–text encoders
> (`docs/EFFICIENCY_STUDY.md`).

### 4. Section 3.4 "Rationale" (core, one sentence)

After "…and frozen features from web-scale image–text pretraining (C)." add:

> D repeats C's strategy with a fourth family, a convolution–attention hybrid, offered under the brief's clause
> "another approved image classification architecture"; A–C satisfy the brief without it.

### 5. Section 4, setup details (core, short)

- 4.1 Preprocessing, in the list of crop fractions and sizes: "1.0 for D: 256 px bicubic with no border removed;
  CLIP mean and standard deviation."
- 4.2 Augmentation, last sentence: "Model C uses no augmentation" → "Models C and D use no augmentation".
- 4.3 Hyperparameters, after the tie-break sentence:
  > Model D's head went through the same grid under the same rule; there all four values tied.
- 4.4, "Fairness" paragraph, at the end:
  > D's crop fraction of 1.0 is the view it was registered with before any test look. On the development folds the
  > same view is worth about 1.5 points to C's features as well (`docs/EFFICIENCY_STUDY.md`), so, if anything, the
  > gap from D to C is understated.

### 6. Section 5.1 "Overall test performance" (core; replaces one claim)

The sentence "…and the 95 % bootstrap intervals do not overlap" is no longer true for the pair C, D (0.787–0.859
and 0.732–0.811). Proposed replacement for the first three sentences:

> Table 3 gives the test metrics. The ranking is the same for every metric: model C (accuracy 0.823, macro-F1
> 0.817) > model D (0.773, 0.770) > model B (0.636, 0.628) > model A (0.400, 0.392). The 95 % bootstrap intervals of
> A, B and the two frozen models do not overlap; those of C and D do, and the paired comparison separates them. The
> paired comparisons (Table 4) are decisive: B is right where A is wrong on 119 images against 20 the other way, C
> beats B on 92 against 14, D beats B on 80 against 23 (p = 1.5×10⁻⁸), and C beats D on 39 against 18 (p = 7.5×10⁻³;
> 5.0 points, interval 1.4 to 8.6).

"Test accuracy is slightly above validation accuracy … for every model" does not hold for D (validation 0.775, test
0.773); "within about two points of" covers all four.

At the end of the paragraph on reference rows:

> R3 is R2's recipe on D's features (accuracy 0.806) and is likewise a reference row, not a further model.

### 7. Section 5.2 "Learning curves" (optional, one bullet)

> **D** follows C one step lower: validation accuracy passes 0.75 in epoch 4 and reaches 0.775, training accuracy
> reaches 1.00 on 3,072 standardised features, and early stopping selected epoch 11.

### 8. Section 5.3 "Per-class results" (optional, one sentence)

> D is within a few points of C in most classes, ahead in three (resistor, capacitor, battery holder) and weakest on
> led (F1 0.61).

### 9. Section 5.5 "Agreement and computational cost" (core for the cost sentence)

After the three-way agreement sentences (which stay about A, B and C):

> With D counted too, all four models are right on 140 images and all four wrong on 43; D alone is right on 14.

After "…its frozen backbone weighs 1,633.6 MB in fp32.":

> D recovers most of that cost: 468.4 MB of weights and, on the PC where both were timed with the same procedure,
> 2.5 times faster than C's tower on the GPU at batch size 1, 9 times at batch size 32 and 14 times on the CPU
> (111.5 against 1,563 ms per image).

### 10. Section 6.1 "Why the models differ" (core)

At the end of the subsection:

> **D** applies the same strategy with a 122.8-million-parameter hybrid pretrained on two billion image–text pairs
> for on-device use. Its features separate the classes less well than C's (5.0 points) but far better than B's
> fine-tuned ImageNet features (13.6 points), although B's network is fully adapted to the task and D's is not
> touched: what the backbone has seen matters more than its family or whether it is fine-tuned.

### 11. Section 6.2 "Error analysis" (optional)

- After the look-alike paragraph: "D's most frequent confusions are ic chip → microcontroller board (4), inductor
  coil → transformer (3), jumper wire → connector (3) and led → diode (3), the same look-alike families."
- In the clean/flagged sentence: "D: 0.807 on clean and 0.679 on flagged images, a drop of 13 points."
- In the confidence sentence: "D: 0.90 on correct and 0.56 on wrong predictions."

### 12. Section 6.5 "Limitations", new item (core)

> **Model D's asymmetries.** D was added after A–C and timed on a different PC (the efficiency table gives C's tower
> timed there as the anchor). Its evaluation view removes no border while C's removes 10 %. Its weights carry
> Apple's research-only model licence, unlike the Apache-2.0 weights of B and C, so it can be used here but not
> shipped commercially.

### 13. Section 6.3 "Can the scratch CNN be improved?" (wording only)

"…so it is a fourth model rather than a better A" would now collide with D. Suggest "a different model".
Likewise in 5.1, "R2 is not a fourth model" → "R2 is not a further model".

### 14. Section 7 "Conclusion" (core)

In the ranking sentence, add D between C and B: "…a frozen MobileCLIP2-S3 hybrid with the same head (0.773,
0.770)…". In the recommendation paragraph, before the ConvNeXt-T sentence:

> Where C is too heavy, model D is the second choice: 5.0 points lower with 3.5 times fewer parameters, 2.5 times
> faster on a GPU and 14 times on a CPU (about 112 ms per image), for non-commercial use under its weights' licence.

## Table rows (ready-made LaTeX in `tables/`)

Each file is the complete table as `make_tables.py` would write it with D included; the D lines are the ones to take.

| Report table | File | What is new |
|---|---|---|
| Table 3, test metrics | `tables/tab_test_metrics.tex` | D row with its interval; R3 reference row |
| Table 4, paired comparisons | `tables/tab_pairwise.tex` | C − D, D − B, D − A |
| Hyperparameters | `tables/tab_hparams.tex` | a fourth column (this made the table a full page) |
| Table B, state at selected epoch | `tables/tab_curves.tex` | D row |
| Table B, efficiency | `tables/tab_efficiency.tex` | D row and "C's tower on D's PC" anchor row |
| Clean vs flagged | `tables/tab_ood.tex` | D row |
| new, Appendix A | `tables/tab_per_class_D.tex` | per-class P/R/F1 of D and R3 |
| new, Appendix B | `tables/tab_wdgrid_D.tex` | D's weight-decay grid |

Paired comparisons, for reference:

| Pair | Δ accuracy | 95 % CI | Only 1st right | Only 2nd right | McNemar p |
|---|---|---|---|---|---|
| C − D | +0.050 | +0.014 to +0.086 | 39 | 18 | 7.5×10⁻³ |
| D − B | +0.136 | +0.091 to +0.179 | 80 | 23 | 1.5×10⁻⁸ |
| D − A | +0.373 | +0.325 to +0.423 | 166 | 10 | 1.3×10⁻³⁷ |

Efficiency row for D (timed on an RTX 4070 Laptop PC, not the report's RTX 4060 Ti PC):
122.78 M parameters, 61,460 trainable, 256 px, checkpoint 0.26 MB, GPU 15.16 ms at batch 1 and 3.098 ms per image
at batch 32, CPU 111.5 ms. C's tower on that same PC: 37.94, 27.877 and 1,563.3.

## Figures with D (in `results/itri626/figures_four_models/`, PDF and PNG)

The report's own figures in `results/itri626/figures/` are untouched. These are the four-model versions, written by
`notebooks/ITRI626_three_models.ipynb`, should you want them: `fig03_augmentation`, `fig04_learning_curves`, `fig05_test_metrics`, `fig06_confusion_grid` (four count
matrices on one colour scale; it replaces the panel with C row-normalised), `fig06_confusion_D_…`,
`fig07_per_class_f1`, `fig08_gallery_D_…` (D's correct and incorrect predictions), `fig11_ood_accuracy`,
`fig12_efficiency`. D's colour is wine `#882255`, checked against the existing three for colour-vision deficiency.

## References to add (verified on arXiv, 2026-11-05)

```bibtex
@inproceedings{vasu2023fastvit,
  title = {{FastViT}: A fast hybrid vision transformer using structural reparameterization},
  author = {Pavan Kumar Anasosalu Vasu and James Gabriel and Jeff Zhu and Oncel Tuzel and Anurag Ranjan},
  booktitle = {Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)},
  year = {2023}
}
@article{faghri2025mobileclip2,
  title = {{MobileCLIP2}: Improving multi-modal reinforced training},
  author = {Fartash Faghri and Pavan Kumar Anasosalu Vasu and Cem Koc and Vaishaal Shankar and Alexander Toshev and Oncel Tuzel and Hadi Pouransari},
  journal = {Trans. Mach. Learn. Res.},
  year = {2025},
  note = {arXiv:2508.20691}
}
```

## Rubric checklist lines affected (Appendix E)

- "At least three models from three families": add "plus D, a convolution–attention hybrid".
- "At least one model with transfer learning": "B (fine-tuning), C and D (feature extraction)".
- "Confusion matrix", "Comparison of all three models": "four" if the four-model figures are used.

## The notebook's file name

`notebooks/ITRI626_three_models.ipynb` now covers four models but keeps its name, because the report cites that
path in three places (the reproduction commands and the rubric table) and so do `RUBRIC_CHECKLIST.md`,
`REVISION_NOTES.md`, `README.md`, `AGENTS.md`, `CODEBASE.md`, `STATUS.md` and the training notebook: 14 references in
9 files. If model D goes into the report, renaming it to `ITRI626_four_models.ipynb` belongs in that same change,
with those references updated together.

## If you would rather see it assembled

A complete trial build exists on the local branch `Meiring-four-model-draft` (commit 2bd71b3, not pushed, not for
merge): `main.tex`, `make_tables.py` and `extra.bib` with every insert above as macros, and the rebuilt PDF. It is
there only as a reference for the text and layout.
