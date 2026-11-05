# ITRI626 rubric compliance checklist

Maps every requirement of the ITRI626 project brief (sections 2–12) to where it is addressed in the report
`docs/itri626_report/main.pdf` and in the repository. This standalone checklist is retained as submission-planning documentation; it is not included in the paper.
Section/figure/table numbers refer to the 5 November 2026 four-model revision. Appendix floats are numbered
per appendix. The authoritative source is `main.tex`; the older model-D inserts are historical proposals.
A–C meet the three named families; D is supplementary pending any lecturer approval as a fourth family.


## Problem definition and motivation (2 marks)

| Brief requirement | Report location | Repository files |
|---|---|---|
| Classification task, aim, scope, motivation (§2, §4) | Section 1 (incl. research question) |  |
| Target classes and reasons for them (§2, §5) | Section 1, Table 1 |  |

## Dataset selection and preparation (4 marks)

| Brief requirement | Report location | Repository files |
|---|---|---|
| Source, access link, licence, usage conditions (§5) | Section 2.1; README "ITRI626 submission" | `README.md` § ITRI626 submission |
| Class definitions (§5) | Table 1 | `data/v2/splits/split_summary.csv` |
| Images per class before and after cleaning (§5) | Table 1, Table 2, Figure B.1 | `data/v2/splits/split_summary.csv`, `data/v2/leakage_report.json` |
| Removed duplicates, corrupt files, irrelevant samples, uncertain labels (§5) | Section 2.3, Table 2 | `tools/build_dataset_v2.py`, `data/v2/leakage_report.json`, `data/v2/review/` |
| How labels were obtained, checked and corrected (§5) | Section 2.3 (last paragraph) | `results/phase1/label_issues_v2.csv`, `data/v2/review/` |
| Class imbalance and how it was handled (§5) | Section 2.4; Section 6.2 |  |
| Representative examples of every class (§5) | Figure 1 | `results/itri626/figures/fig02_examples_per_class.pdf` |
| Ethics, privacy, lawful use, acknowledgement of internet data (§5) | Section 2.6, Section 2.1 | `data/web_holdout_v1/ATTRIBUTION.md` |
| Train/validation/test split; same test set for all models (§7.1) | Section 2.5; `data/v2/splits/` | `data/v2/splits/{train,val,test}.csv`, `split_info.json` |
| Leakage prevention by group identity (§7.1) | Section 2.5, Section 2.3 item 3, Section 6.3 | `tools/export_splits.py --check`, `data/v2/leakage_report.json` |
| Random seed, images per split, class distribution per split (§7.1) | Section 2.5, Table 1 | `data/v2/splits/split_info.json`, `split_summary.csv` |

## Model selection and methodology (4 marks)

| Brief requirement | Report location | Repository files |
|---|---|---|
| At least three models from three families (§6) | Section 3: plain CNN, ConvNeXt, ViT; D is a supplementary hybrid |  |
| At least one model with transfer learning (§6) | B (fine-tuning), C and D (feature extraction), Section 3 |  |
| Methods, transfer-learning approach, complete workflow (§6) | Section 3, Figure 2 |  |
| Same split for all models; fair, justified hyperparameters (§6) | Section 4, Table B.1, Table B.5, Table B.6 | `results/itri626/hyperparameters.csv`, `head_grid.csv` |

## Experimental setup and reproducibility (3 marks)

| Brief requirement | Report location | Repository files |
|---|---|---|
| Preprocessing steps (§7.2) | Section 4 (Preprocessing) |  |
| Augmentation, training data only, realism argued (§7.2) | Section 4 (Augmentation), Figure 3 |  |
| Model name, variant, pretrained weights; input size (§7.3) | Table B.1 | `results/itri626/hyperparameters.csv` |
| Batch size, gradient accumulation; epochs, early stopping (§7.3) | Table B.1 | `results/itri626/hyperparameters.csv`, `runs/*.json` |
| Optimiser, learning rate, schedule; weight decay, loss (§7.3) | Table B.1, Table B.5, Table B.6 | `results/itri626/hyperparameters.csv` |
| Class weights or sampling; random seed (§7.3) | Table B.1, Section 2.4 | `results/itri626/hyperparameters.csv` |
| Software framework and library versions; hardware, GPU memory (§7.3) | Section 4.4, Table B.2 | `results/itri626/environment.json`, `uv.lock` |
| Fairness of the comparison (rubric) | Section 4.4 (paragraph "Fairness"), Table B.1 |  |

## Results and evaluation (4 marks)

| Brief requirement | Report location | Repository files |
|---|---|---|
| Test accuracy; precision, recall, F1; macro-F1 (§8) | Table 3 | `results/itri626/metrics.json` |
| Per-class precision, recall and F1 (§8) | Figure 5; Table A.1, Table A.2, Table A.3 (Appendix A) | `results/itri626/per_class.csv` |
| Confusion matrix (§8) | Figure A.1/A.2 (all four models, shared colour scale) | `results/itri626/figures_four_models/fig06_confusion_grid.pdf` (+ per-model `fig06_confusion_*.pdf`) |
| Training and validation loss curves (§8) | Figure 4 (top row), Table B.3 | `results/itri626/histories.json`, `figures_four_models/fig04_learning_curves.pdf` |
| Validation accuracy or F1 curve (§8) | Figure 4 (bottom row) | `results/itri626/figures_four_models/fig04_learning_curves.pdf` |
| Several correct and incorrect predictions (§8) | Figure 6 (C) and A.3 (A/B), Figure 7, Figure B.2 (D) | `results/itri626/figures/fig08_gallery_*.pdf`, `fig10_disagreements.pdf` |
| Comparison of all four models (rubric) | Table 3, Table 4, Figure A.4, Table B.4 |  |

## Discussion, error analysis and conclusion (2 marks)

| Brief requirement | Report location | Repository files |
|---|---|---|
| Important classification errors (§8) | Section 6.2, Table 5 |  |
| Effects of class imbalance and data quality (§8) | Section 6.2, Section 2.4 | `results/itri626/figures/fig11_ood_accuracy.pdf` |
| Plausible reasons for performance differences (§8) | Section 6.1 |  |
| Limitations (§9) | Section 6.5 |  |
| Justified conclusion and recommended model (rubric) | Section 7 (incl. lessons learned) |  |

## Report quality and references (1 mark)

| Brief requirement | Report location | Repository files |
|---|---|---|
| Nine-part structure, 10–15 pages, PDF (§9) | 15-page body (title + sections 1–7); references and appendices follow |  |
| Sources, pretrained models, software acknowledged (§9) | Section 2.1, Section 3, Table B.2, references |  |

## Submission items (§11)

| Brief requirement | Report location | Repository files |
|---|---|---|
| Report as PDF | `docs/itri626_report/main.pdf` |  |
| Presentation slides | to be prepared by the group (not part of this repository) |  |
| Notebook with visible outputs, figures, metric tables | `notebooks/ITRI626_four_models.ipynb` |  |
| Python scripts for data preparation and evaluation | `tools/build_dataset_v2.py`, `tools/export_splits.py`, `scripts/itri626_experiments.py`, `src/compclass/` |  |
| README with dataset source, setup, reproduction steps | `README.md`, section "ITRI626 submission" |  |
| Split files or a script that recreates them | `data/v2/splits/`, `tools/export_splits.py` |  |
| Repository link or archive; data access instructions | README "ITRI626 submission"; Appendix D |  |

## Items the group must supply

- Verify the supplied group names/numbers and institutional details on the title page before submission; these are already filled in.
- Presentation slides and the oral presentation (brief §10; 10 marks) — not part of this repository.
- Repository link or submission archive for eFundi (brief §11).
