# Paraphrase worksheet: the text added for model D

Everything here is taken from `FULL_REPORT_WITH_MODEL_D.md`, in the order of the report. Write your own wording
under **Your version** and leave the labels (P01, C01, ...) as they are: a label is how each piece finds its way
back into the hand-over files. An empty **Your version** keeps the original.

Keep unchanged: every number, the names in `code font`, the citation markers such as \[22\], and the references
to tables, figures and sections. The tables, the figures and the two new reference entries are not in this file;
they hold no prose to paraphrase.

- **Part A** (15 pieces): whole sentences and paragraphs that are new. This is the main work.
- **Part B** (23 pieces): a clause added inside a sentence that is already in the report. Only the bold
  words are new; the rest of the sentence is the report's own text and stays.
- **Part C** (16 lines): single-word and number changes such as *three* to *four*. Nothing to paraphrase;
  listed so that the file is complete.

## Part A: new sentences and paragraphs

### P01 — 3.4 Model D: MobileCLIP2-S3 hybrid, frozen, with a linear head (feature extraction, small backbone) (heading)

**Original:**

> Model D: MobileCLIP2-S3 hybrid, frozen, with a linear head (feature extraction, small backbone)

**Your version:**


### P02 — 3.4 Model D: MobileCLIP2-S3 hybrid, frozen, with a linear head (feature extraction, small backbone)

**Original:**

> MobileCLIP 2 \[22\] trains image–text encoders for on-device use; its S3 image encoder is a FastViT convolution–attention hybrid \[23\] at 256×256 pixels with 122,775,668 parameters, 3.5 times fewer than C (`timm` weights `fastvit_mci3.apple_mclip2_dfndr2b`). It is used exactly as C, with one `nn.Linear(3072, 20)` head trained by the same code, grid and selection rule. It was added to answer the cost half of the research question and was chosen on the development folds only (`docs/EFFICIENCY_STUDY.md`).

**Your version:**



### P03 — 3.5 Rationale for the models and evaluation protocol

**Original:**

> D repeats C’s strategy with a fourth family, a convolution–attention hybrid (the brief’s “another approved image classification architecture”; A–C satisfy the brief without it).

**Your version:**



### P04 — 4.3 Hyperparameters

**Original:**

> Model D’s head went through the same grid under the same rule (Table B.5); there all four values tied.

**Your version:**



### P05 — 4.4 Software, hardware and reproducibility

**Original:**

> D’s crop fraction of 1.0 was fixed before any test look; on the development folds that view is worth about 1.5 points to C as well, so the gap from D to C is, if anything, understated.

**Your version:**



### P06 — 5.1 Overall test performance

**Original:**

> The 95% bootstrap intervals of A, B and the two frozen models do not overlap; those of C and D do, and the paired comparison separates them.

**Your version:**



### P07 — 5.1 Overall test performance

**Original:**

> R3 is the same recipe on D’s features (0.806) and is likewise a reference row.

**Your version:**



### P08 — 5.5 Agreement between the models and computational cost

**Original:**

> D recovers most of that cost: 468.4 MB of weights and, timed on one PC with the same procedure, 2.5 times faster than C’s tower on the GPU and 14 times on the CPU.

**Your version:**



### P09 — 6.1 Why the models differ

**Original:**

> **D** applies the same strategy with a 122.8-million-parameter hybrid. Its features separate the classes less well than C’s (5.0 points) but far better than B’s fine-tuned ImageNet features (13.6 points): what the backbone has seen matters more than its family or whether it is fine-tuned.

**Your version:**



### P10 — 6.5 Limitations

**Original:**

> **Model D’s asymmetries.** D was timed on a different PC (Table B.3 gives C’s tower there as the anchor), its evaluation view removes no border while C’s removes 10%, and its weights carry Apple’s research-only licence, unlike the Apache-2.0 weights of B and C.

**Your version:**



### P11 — 7 Conclusion

**Original:**

> Where C is too heavy, model D is the second choice: 5.0 points lower with 3.5 times fewer parameters, 2.5 times faster on a GPU and 14 times on a CPU, for non-commercial use.

**Your version:**



### P12 — A Full per-class results (caption of new Table A.3)

**Original:**

> Per-class test results of model D and of R3, the packaged Small model: D’s frozen features with R2’s recipe (probe fitted on train + validation, fused with MobileCLIP 2’s zero-shot text head; `exports/mobileclip2_s3/`).

**Your version:**



### P13 — B Additional dataset and training details

**Original:**

> ‡D was timed on a second PC (RTX 4070 Laptop GPU); the last row is C’s tower on that PC under the same procedure, the anchor for the C-to-D ratios in the text.

**Your version:**



### P14 — B Additional dataset and training details

**Original:**

> **Table B.5.** The same weight-decay selection for model D’s head (`results/itri626/head_grid_D.csv`); all 4 values tie and the smallest was kept.

**Your version:**



### P15 — B Additional dataset and training details (caption of new Figure B.2)

**Original:**

> Model D (MobileCLIP2-S3 hybrid, frozen + linear): six random correct predictions (top) and the six most confident errors (bottom), as in Figure 7.

**Your version:**



## Part B: clauses added inside existing sentences

Rewrite only the bold words. Write the new wording of the bold part, or the whole sentence with your wording in it.

### C01 — Title page (subtitle)

> A leakage-free comparison of a scratch CNN, a fine-tuned **ConvNeXt,** a frozen SigLIP Vision Transformer **and a frozen MobileCLIP 2 hybrid**

**Your version:**



### C02 — Abstract

> **Four** model families are trained on this split and scored once on the same test set: (A) a VGG-style CNN trained from scratch, (B) ConvNeXt-Tiny fine-tuned from ImageNet-22k **weights,** (C) a frozen SigLIP So400m Vision Transformer with a trained linear **head and (D) a frozen MobileCLIP2-S3 convolution–attention hybrid, 3.5 times smaller, with the same** head.

**Your version:**



### C03 — Abstract

> Pretraining accounts for the ranking: ImageNet fine-tuning adds 23.7 percentage points over the scratch baseline, and **frozen** web-scale image–text **features** a further **18.7 with the large encoder or 13.6 with the small one.**

**Your version:**



### C04 — Abstract

> We recommend model C, **model D where C is too heavy,** and ConvNeXt-T **for real-time use** on a CPU.

**Your version:**



### C05 — 1 Introduction and problem motivation

> We compare **four** model families, one trained from scratch and **three that use transfer learning (fine-tuning, and frozen features from a large and from a small web-pretrained encoder),** on one leakage-free train/validation/test split, evaluate all of them on the same held-out test set, explain their differences and recommend one.

**Your version:**



### C06 — 1 Introduction and problem motivation

> We recommend model C, **model D where C is too heavy,** or ConvNeXt-T **for real-time CPU** inference (Section 7).

**Your version:**



### C07 — 4.1 Preprocessing

> For evaluation (validation, test, and the feature extraction of model **C and model D)** the short side is resized to the input size divided by the crop fraction (0.875 for A and B, 0.9 for **C, 1.0 for D)** and the centre square is cut out: 183→160 px bilinear for A, 256→224 px bicubic for B, 427→384 px bicubic for **C, 256 px bicubic with no border removed for D.**

**Your version:**



### C08 — 4.1 Preprocessing

> Pixels are scaled to [0,1] and normalised per channel (mean and standard deviation 0.5 for A and C; ImageNet statistics for **B; CLIP statistics for D).**

**Your version:**



### C09 — 4.2 Augmentation (training data only) (caption)

> Figure 3. One training image, its evaluation crop and five random training augmentations **for A, B and C.**

**Your version:**



### C10 — 4.4 Software, hardware and reproducibility

> Peak GPU memory was 435 MB for A and 2,312 MB for B, and **the heads of** C **and D were** trained on the CPU from cached **features (D’s features were extracted, and D was timed, on the second machine).**

**Your version:**



### C11 — 4.4 Software, hardware and reproducibility

> Tuning effort is comparable: the recipes of A and B come from the earlier development-set benchmark, and **the** only tuned setting **of C and of D** is a four-value weight-decay grid on the validation set **(Tables B.4 and B.5).**

**Your version:**



### C12 — 5.1 Overall test performance

> The ranking is the same for every metric: model C (accuracy 0.823, macro-F1 0.817) > model **D (0.773, 0.770) > model** B (0.636, 0.628) > model A (0.400, **0.392).**

**Your version:**



### C13 — 5.1 Overall test performance

> The paired comparisons (Table 5) are decisive: B is right where A is wrong on 119 images against 20 the other way (p= 2.4×10⁻¹⁸), C beats B on 92 images against 14 (p= **3.1×10⁻¹⁵), D beats B on 80 against 23 (p= 1.5×10⁻⁸), and C beats D on 39 against 18 (p= 7.5×10⁻³).**

**Your version:**



### C14 — 5.1 Overall test performance

> Test accuracy is **within about two points of** validation accuracy at the selected epoch for every model (A 0.378, B 0.618, C **0.803, D 0.775),** within the sampling variation of two sets of about 450 images.

**Your version:**



### C15 — 5.3 Per-class results and confusion matrices

> The confusion matrices (Figure 6) show A’s errors spread over the whole matrix, B’s concentrated in a few blocks, and **those of C and D** almost diagonal; Section 6.2 analyses the largest off-diagonal entries.

**Your version:**



### C16 — 5.4 Correct and incorrect predictions

> Figure 7 shows, for **A, B and C,** six random correct predictions and **the** six most confident errors with true class, prediction and softmax **confidence; Figure B.2 (Appendix B) does the same for D.**

**Your version:**



### C17 — 5.5 Agreement between the models and computational cost (caption)

> Figure 8. Test images by the set of models **among A, B and C** that classify them correctly.

**Your version:**



### C18 — 6.2 Error analysis

> **A, B and C all** fail on 57 test images, many of them infographics, diagrams or scenes without the labelled component, where the label rather than the prediction is often questionable (Figure 9).

**Your version:**



### C19 — 7 Conclusion

> On one leakage-free split and the same 418 test images the **four** model families rank: a frozen SigLIP So400m ViT with a linear head (accuracy 0.823, macro-F1 0.817), **a frozen MobileCLIP2-S3 hybrid with the same head (0.773, 0.770),** a fine-tuned ConvNeXt-T (0.636, 0.628) and a VGG-style CNN trained from scratch (0.400, 0.392).

**Your version:**



### C20 — A Full per-class results

> Table A.1 gives the per-class test precision, recall and F1 of **model A, model B and model C, Table A.3 those of model D and its reference row R3,** and Table A.2 those of the reference rows **R1 and R2, all** on the same test images.

**Your version:**



### C21 — B Additional dataset and training details

> Software and hardware of the training run of B and C (`results/itri626/environment.json`); the final run of A **and the feature extraction and timing of D** used a second machine (Section 4.4).

**Your version:**



### C22 — B Additional dataset and training details

> Table B.2 gives the state of each model at its selected epoch, Table B.3 model size and inference time, Figure B.1 the class distribution before and after cleaning and per split (the numbers are in Table 1), **Tables B.4** and **B.5** the weight-decay **grids of the heads of C and D, and Figure B.2 the correct and incorrect predictions** of model **D.**

**Your version:**



### C23 — B Additional dataset and training details

> **The checkpoints of C and D hold** only the head; the backbone is the public weight file.

**Your version:**



## Part C: word and number changes (no paraphrase needed)

The changed words are in bold.

- *Abstract:* Test accuracy is 0.400, **0.636, 0.823** and **0.773** (macro-F1 0.392, 0.628, **0.817, 0.770),** and McNemar tests reject equal accuracy for every pair.
- *1 Introduction and problem motivation:* Model C is the most accurate (accuracy 0.823, macro-F1 0.817), ahead of **model D (0.773, 0.770),** model B (0.636, 0.628) and model A (0.400, 0.392); every pairwise difference is significant (Section 5).
- *3 Methodology, selected models and transfer learning:* The same split files feed all **four** models; each is selected on the validation set only and scored once on the test set.
- *3 Methodology, selected models and transfer learning (caption):* Data preparation (top) is shared; the **four** models (middle, in the colours used in every figure) use the same split files and selection rule and are evaluated on the same 418 test images.
- *3.5 Rationale for the models and evaluation protocol:* The brief lists a small custom CNN, ConvNeXt and the Vision Transformer as separate suitable families; our **first three** models take one from each (plain CNN, modernised CNN, Vision Transformer) and span the transfer-learning spectrum: no transfer (A), full fine-tuning of supervised ImageNet features (B) and frozen features from web-scale image–text pretraining (C).
- *4.2 Augmentation (training data only):* Model C **and model D use** no augmentation: **their** features are computed once from the evaluation **view,** which keeps the frozen-feature protocol simple and cheap.
- *4.2 Augmentation (training data only) (caption):* Model C **and model D use** frozen features of the evaluation **view** and therefore no augmentation.
- *4.4 Software, hardware and reproducibility:* All **four** models use the same 2,010 training, 502 validation and 418 test images, the same selection rule (best validation macro-F1) and a single test evaluation each, with no test-time augmentation.
- *5.1 Overall test performance:* R2 is not a **further** model: it is C refitted on more data and is reported only to show what the recommended approach reaches in deployment.
- *5.3 Per-class results and confusion matrices (caption):* Figure 5. Per-class test F1 of the **four** models, sorted by model C; the number after each class is its training-set size.
- *5.3 Per-class results and confusion matrices (caption):* Figure 6. Test confusion matrices **of the four models** (rows: true class; columns: predicted class; n = **418): counts** on one shared colour scale, so equal shades mean equal counts; empty cells are zero.
- *5.3 Per-class results and confusion matrices (caption):* **The** row-normalised **matrices (per-class recall on** the **diagonal)** are in the notebook.
- *5.5 Agreement between the models and computational cost:* **A, B and C** are **all** right on 143 test images and all wrong on 57 (Figure 8).
- *6.2 Error analysis:* Summed over **A, B and C,** the most frequent confusions are resistor → crystal oscillator (8); jumper wire → connector (7); transistor → diode (7); crystal oscillator → relay (6).
- *7 Conclusion:* **Where** inference **must be faster still,** the fine-tuned ConvNeXt-T **remains:** about 11 times faster **than C** on a GPU, at about 18.7 percentage points lower.
- *D Reproduction:* `scripts/itri626_experiments.py` `--steps head reference **head_D reference_D** collect` re-runs only the CPU parts **(the heads of** C **and D** from cached features, the reference rows and the result collection).
