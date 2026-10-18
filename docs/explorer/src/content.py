"""Hand-written text for the interactive explorer (docs/explorer/index.html).

Loaded by scripts/build_explorer.py with runpy. Everything here is prose, structure and *pointers* to numbers:
measured results are NOT typed here -- the build script reads them from results/*.csv|json and the report tables,
and asserts the headline values. A few headline numbers appear in the timeline prose; the build script checks each
of them against its source file. Strings use a tiny Markdown subset: **bold**, *italic*, `code`.

Architecture facts (tensor shapes, block counts) are copied from docs/models/*.md, which were checked against
timm 1.0.30. The SigLIP So400m/14@384 numbers (patch 14, width 1152, depth 27, 16 heads, mlp_ratio 3.7362 -> 4304,
no [CLS], MAP pooling) were read from timm 1.0.30 `vision_transformer.py` (`vit_so400m_patch14_siglip_384`).
MobileNetV3-L channel widths are from Howard et al. 2019 (Table 1); EfficientNetV2-S spatial sizes follow from the
documented strides with TF 'same' padding. FLOPs are only given where the original paper reports them, at the
paper's resolution, and are labelled as such.
"""

# --------------------------------------------------------------------------------------------------------------
# Timeline: phases of the project. `commits` are short hashes that must exist in `git log` (asserted by the build).
# `checks` lists (text-fragment, key) pairs the build script verifies against source files.
# --------------------------------------------------------------------------------------------------------------
TIMELINE = [
    {
        "id": "legacy", "title": "Legacy notebooks", "date": "2026-08",
        "what": "Self-contained notebooks trained ImageNet CNNs/transformers (ResNet, EfficientNet, ConvNeXt, Swin, "
                "DenseNet, MobileNet) and a scratch CNN on a file-level 70/15/15 split of the Kaggle download, "
                "with Optuna tuning and ensembles.",
        "why": "First attempt at the comparison and at an 80-85 % demo model.",
        "result": "Best validation-selected test result: ConvNeXt-Base, accuracy 0.7132 / macro-F1 0.6969 "
                  "(`legacy/docs/EXPERIMENT_LOG.md`). Later shown to be measured on a split that leaked "
                  "duplicate images between train and test, so the numbers are not comparable with v2.",
        "files": ["legacy/", "legacy/docs/EXPERIMENT_LOG.md"],
        "commits": ["12d43de", "0f81bd5"],
    },
    {
        "id": "rebuild", "title": "Rebuild + GPU witness", "date": "2026-10-17",
        "what": "Everything moved into one importable package `src/compclass` with a single model registry, YAML "
                "configs (`configs/base.yaml` + one file per family), a training loop, probe, zero-shot, classical "
                "and export modules. A *GPU witness* script built and trained every registry model on the 8 GB "
                "RTX 4060 Ti.",
        "why": "Notebooks duplicated code and hid bugs; a package makes every family run through identical data, "
               "folds, metrics and logging.",
        "result": "13/13 registry models build and train (fp16 AMP); CLIP ViT-L/14 needs gradient checkpointing "
                  "(`results/gpu_witness_carlpc2.json`).",
        "files": ["src/compclass/", "configs/", "CODEBASE.md", "results/gpu_witness_carlpc2.json"],
        "commits": ["ebb0e80", "2cc6954", "d643c41", "5dd9ca3", "9545682"],
    },
    {
        "id": "data", "title": "Data audit (v2)", "date": "2026-10-17",
        "what": "`tools/build_dataset_v2.py` decodes all 3,530 files, collapses exact duplicates by SHA-256, links "
                "near-duplicates (pHash + CLIP cosine), excludes conflicting-label duplicates, flags out-of-scope "
                "images with a CLIP distractor screen and assigns whole duplicate *clusters* to a sealed test split "
                "and five grouped folds.",
        "why": "The legacy split put copies of the same picture on both sides of train/test; any comparison built on "
               "it measures memorisation.",
        "result": "3,530 files -> 2,943 unique -> 2,930 rows -> 2,787 clusters; 816 rows flagged out-of-scope; "
                  "sealed test 418, dev 2,512 (see the Data audit tab).",
        "files": ["data/v2/manifest.json", "data/v2/leakage_report.json", "data/v2/review/"],
        "commits": ["de30c99"],
    },
    {
        "id": "phase1", "title": "Phase 1: probes + zero-shot", "date": "2026-10-17",
        "what": "Frozen-feature logistic-regression probes for 12 backbones, zero-shot for 7 vision-language "
                "models, the HOG + SVM baseline; a confident-learning label audit and a training-data cleaning "
                "ablation; later flip-TTA probes (Phase 1b).",
        "why": "Probes are cheap (minutes per backbone) and show how good each pretrained feature space is before "
               "spending GPU hours on fine-tuning.",
        "result": "SigLIP So400m/14@384 probe: macro-F1 0.824 +/- 0.012 -- the best method in the whole project. "
                  "69 label issues found; removing them or the OOD-flagged rows from training did not help.",
        "files": ["results/phase1/", "results/phase1b/runs.jsonl", "results/phase1/label_issues_v2.csv"],
        "commits": ["8a06aad", "cbf1877"],
    },
    {
        "id": "docs", "title": "Learning docs", "date": "2026-10-17",
        "what": "One explainer per model family (`docs/models/*.md`), three concept notes (`docs/concepts/`) and a "
                "reading order (`docs/LEARNING_PATH.md`). This explorer embeds and extends them.",
        "why": "The repository doubles as a learning resource on model families and honest evaluation.",
        "result": "11 family explainers, checked against timm 1.0.30.",
        "files": ["docs/models/", "docs/concepts/", "docs/LEARNING_PATH.md"],
        "commits": ["3564493", "da2fb4c"],
    },
    {
        "id": "phase2", "title": "Phase 2: fine-tune benchmark", "date": "2026-10-17/18",
        "what": "Fine-tuning of 10 families on all five folds with one hand-set recipe each, plus a partial "
                "fine-tune of SigLIP So400m (last 4 of 27 blocks). A first run (jobs 2A/2B) had a schedule bug "
                "(LR warm-up not restarted when the backbone unfroze); commit `d6d4e09` fixed it (and also froze "
                "BatchNorm statistics during the head-only phase) and the main table was re-run overnight.",
        "why": "Answers the second research question: at ~2.5k images, should a pretrained backbone be frozen or "
               "fine-tuned?",
        "result": "Best full fine-tune: SigLIP 2 B/16, 0.733 macro-F1 -- below its own probe (0.758). ImageNet "
                  "fine-tunes 0.548-0.654. The schedule fix had no consistent effect (null result).",
        "files": ["results/phase2/runs_fixed.jsonl", "results/phase2/raw_oldschedule/", "results/benchmark.csv",
                  "results/schedule_ablation_folds012.csv", "results/leakage_simulation.csv"],
        "commits": ["fe777b7", "d6d4e09", "6592817", "2dd3e8a", "280d6e1", "eaa009f"],
    },
    {
        "id": "final", "title": "Phase 3: final model + sealed test", "date": "2026-10-18",
        "what": "`compclass.final`: frozen SigLIP So400m image tower -> probe head fused with the backbone's own "
                "zero-shot text head -> temperature scaling -> abstain below confidence 0.5. Fitted on the 2,512 "
                "dev rows, exported to ONNX (one graph, 1.6 GB), parity-checked in onnxruntime, then the sealed "
                "test was read exactly once.",
        "why": "The best CV method, made deployable and honest about uncertainty.",
        "result": "Sealed test accuracy 0.837 (95 % CI 0.799-0.870), macro-F1 0.834, ECE 0.034; answers 87.1 % "
                  "of images at 90.9 % accuracy. OpenCV DNN cannot load the graph (segfault), so inference uses "
                  "onnxruntime.",
        "files": ["src/compclass/final.py", "exports/final/", "results/final/final_test_v2.json"],
        "commits": ["8bf4c35", "7e76561", "17b5cf8", "6cee81f", "ee64f58", "3c88283"],
    },
    {
        "id": "report", "title": "Report + demo", "date": "2026-10-18",
        "what": "Comparative report with 8 figures and LaTeX tables, all regenerated from repository files "
                "(`scripts/make_report_figures.py`, `scripts/analysis_tables.py`); OpenCV + onnxruntime demo for "
                "files, folders, URLs and webcams with an UNSURE state.",
        "why": "Deliverables: the written comparison and a working demo.",
        "result": "Demo parity 64/64 images vs PyTorch; ~0.9 s/image on CPU (`results/demo/demo_parity.json`).",
        "files": ["docs/report/REPORT.md", "docs/DEMO.md", "results/demo/"],
        "commits": ["38d5a50", "224ae28", "46adce7"],
    },
    {
        "id": "web", "title": "Web holdout", "date": "2026-10-18",
        "what": "A second sealed set of 407 Wikimedia Commons photos (CC/PD), built after the model was frozen, "
                "with a CLIP leakage gate against every v2 image and a visual review; evaluated once.",
        "why": "The Kaggle test split shares the scrape's biases; this tests other internet photos.",
        "result": "Accuracy 0.821 (95 % CI 0.780-0.855), macro-F1 0.816; here the zero-shot head adds +4.2 points "
                  "over the probe alone.",
        "files": ["data/web_holdout_v1/", "results/final/final_test_web_holdout_v1.json"],
        "commits": ["5d8ddf5", "d071191", "8718a97", "bd68f7e"],
    },
    {
        "id": "kaggle", "title": "Original-Kaggle evaluation", "date": "2026-10-18",
        "what": "The final model scored on every one of the 3,530 original files, broken down by whether the file's "
                "image was used to fit the head, plus a replay of the legacy 70/15/15 protocol.",
        "why": "Answers 'how good is it on the original dataset?' without hiding that most files were training "
               "data, and makes the old notebooks directly comparable.",
        "result": "95.5 % on all files, but 83.0 % on the 493 never-seen files; legacy-protocol replay 87.7 % vs "
                  "the old notebooks' best 71.3 % on the same files.",
        "files": ["scripts/eval_original_kaggle.py", "results/kaggle_original/"],
        "commits": ["f2af0bc"],
    },
]

# --------------------------------------------------------------------------------------------------------------
# Model families. `bench` = benchmark.csv names belonging to the family. `doc` = docs/models/<doc>.md.
# `config` = configs/<config>.yaml whose effective recipe is shown (None for non-trained families).
# --------------------------------------------------------------------------------------------------------------
FAMILIES = [
    {
        "id": "classical", "name": "HOG + colour + SVM", "short": "Classical", "group": "Classical",
        "doc": "classical", "config": None, "diagram": "classical", "year": "1991-2005",
        "papers": ["N. Dalal, B. Triggs, *Histograms of Oriented Gradients for Human Detection*, CVPR 2005",
                   "M. Swain, D. Ballard, *Color Indexing*, IJCV 1991",
                   "C. Cortes, V. Vapnik, *Support-Vector Networks*, Machine Learning 1995"],
        "bench": ["hog_hsv_svm"],
        "tagline": "A human designs the features; a kernel SVM draws the boundary.",
        "idea": [
            "Before deep learning, recognition was a two-stage pipeline: a fixed, hand-designed feature extractor "
            "followed by a generic classifier. **HOG** describes local shape -- which way edges point in each small "
            "cell, normalised over overlapping blocks so it tolerates lighting changes. A **colour histogram** adds "
            "what HOG ignores (green PCBs, blue trimmers, colour bands).",
            "An **RBF-kernel SVM** then finds the maximum-margin boundary in an implicit high-dimensional space. "
            "Nothing in the features is learned from data, so this baseline measures *what learned features buy*.",
        ],
        "flow_title": "Data flow (`src/compclass/classical.py`, docs/models/classical.md)",
        "flow": [
            ["input", "any size, RGB", "image"],
            ["resize + crop", "short side -> 128 px (bilinear), centre crop", "3x128x128"],
            ["HOG (grey)", "9 orientations, 16x16-px cells (8x8 grid), 2x2-cell blocks (7x7 blocks)", "1764"],
            ["HSV histogram", "8 (H) x 4 (S) x 4 (V) bins, divided by #pixels", "128"],
            ["concatenate", "HOG ++ colour", "1892"],
            ["StandardScaler", "z-score per feature, fitted on the training fold", "1892"],
            ["SVC", "kernel='rbf', C=10, gamma='scale', one-vs-one", "20 classes"],
        ],
        "chars": {
            "Learned parameters": "scaler statistics + support vectors (no network)",
            "FLOPs": "negligible (no network)",
            "Input here": "128 x 128",
            "Inductive bias": "fixed by hand: gradient orientations + global colour; no feature learning",
            "Data appetite": "low -- but it cannot turn more data into better features",
            "Pretraining": "none",
            "Strengths": "fast on CPU, interpretable features, no GPU",
            "Weaknesses": "background and photo style dominate; no scale/viewpoint tolerance; no augmentation",
        },
        "hp": [
            ["--C", "SVM regularisation (default 10). Smaller C = wider margin, more training errors tolerated; "
                    "larger C fits the training fold harder. Not tuned here; the docs note a different C may move "
                    "the score by several points."],
        ],
        "regimes": "One regime: hand-crafted features + SVM, refit per fold.",
        "lesson": "The lower bound. The gap to the scratch CNN is what learning features from ~2.5k images buys; every "
                  "pretrained model is far above both.",
    },
    {
        "id": "scratch_cnn", "name": "Scratch CNN (VGG-style)", "short": "Scratch CNN", "group": "From scratch",
        "doc": "scratch_cnn", "config": "scratch_cnn", "diagram": "scratch", "year": "1998-2015",
        "papers": ["Y. LeCun et al., *Gradient-Based Learning Applied to Document Recognition*, Proc. IEEE 1998",
                   "K. Simonyan, A. Zisserman, *Very Deep Convolutional Networks*, arXiv:1409.1556, 2014",
                   "S. Ioffe, C. Szegedy, *Batch Normalization*, arXiv:1502.03167, 2015"],
        "bench": ["scratch_cnn"],
        "tagline": "Stacked 3x3 convolutions learn edges -> textures -> parts, from random initialisation.",
        "idea": [
            "A convolutional network shares one small filter across every image position (translation "
            "equivariance) and looks only at a local neighbourhood (locality). Stacking 3x3 convolutions and "
            "pooling grows the receptive field layer by layer: early layers learn edges, later ones textures and "
            "parts. **BatchNorm** keeps activations well-scaled so high learning rates are stable; **global "
            "average pooling** replaces large fully-connected layers.",
            "Here every weight is learned from ~2,000 training images per fold, with no pretraining. It is the "
            "reference point that shows how much *pretraining* is worth.",
        ],
        "flow_title": "How a picture flows through it (input 3x160x160, width w = 32; docs/models/scratch_cnn.md)",
        "flow": [
            ["input", "RGB, mean = std = 0.5", "3x160x160"],
            ["stem", "conv3x3 (3->w, stride 2) -> BN -> ReLU", "32x80x80"],
            ["stage 1", "[conv3x3 -> BN -> ReLU] x 2 at w, MaxPool 2", "32x40x40"],
            ["stage 2", "[conv3x3 -> BN -> ReLU] x 2 at 2w, MaxPool 2", "64x20x20"],
            ["stage 3", "[conv3x3 -> BN -> ReLU] x 2 at 4w, MaxPool 2", "128x10x10"],
            ["stage 4", "[conv3x3 -> BN -> ReLU] x 2 at 8w (no pool)", "256x10x10"],
            ["pool", "AdaptiveAvgPool(1) -> Flatten -> Dropout(0.3)", "256"],
            ["head", "Linear(8w -> 20)", "20 logits"],
        ],
        "chars": {
            "Inductive bias": "strong: locality + weight sharing (translation equivariance), pooling hierarchy",
            "Data appetite": "high relative to its size -- all features must come from the training fold",
            "Pretraining": "none (random He/Kaiming initialisation)",
            "Input here": "160 x 160",
            "FLOPs": "not reported in a source",
            "Strengths": "tiny (about 1.2M parameters), fast, fully understandable end to end",
            "Weaknesses": "learns shortcuts (background, watermarks, photo style); large seed/fold variance",
        },
        "hp_keys": ["model.kwargs", "train.epochs", "train.patience", "train.lr", "train.warmup_epochs",
                    "train.batch_size", "aug.rrc_scale", "aug.color_jitter", "train.weight_decay"],
        "regimes": "Trained from scratch only (there are no pretrained weights to probe or fine-tune).",
        "lesson": "+19 to +30 points separate it from the ImageNet fine-tunes: that is what pretraining buys at this "
                  "data size.",
    },
    {
        "id": "resnet", "name": "ResNet-50", "short": "ResNet", "group": "ImageNet CNN",
        "doc": "resnet", "config": "resnet50", "diagram": "resnet", "year": "2015",
        "papers": ["K. He, X. Zhang, S. Ren, J. Sun, *Deep Residual Learning for Image Recognition*, "
                   "arXiv:1512.03385, 2015",
                   "R. Wightman, H. Touvron, H. Jegou, *ResNet strikes back*, arXiv:2110.00476, 2021 (the a1 weights)"],
        "bench": ["resnet50", "probe_resnet50"],
        "tagline": "Learn a residual F(x) on top of an identity shortcut: y = x + F(x).",
        "idea": [
            "Around 2015 deeper plain networks trained *worse*, even on training data: an optimisation problem, not "
            "over-fitting. ResNet lets each block learn only a **residual** that is added to an identity shortcut. "
            "Doing nothing is easy (drive F to zero), and gradients flow straight back through the shortcuts, so "
            "50-152-layer networks became trainable.",
            "ResNet-50 uses **bottleneck** blocks (1x1 reduce -> 3x3 -> 1x1 expand x4) so depth stays cheap. The "
            "`a1_in1k` weights are the same architecture retrained with a modern recipe (*ResNet strikes back*).",
        ],
        "flow_title": "How a picture flows through it (224 px; docs/models/resnet.md, timm 1.0.30)",
        "flow": [
            ["input", "ImageNet mean/std; eval: resize 236, centre crop 224", "3x224x224"],
            ["stem", "conv7x7 (3->64, stride 2) -> BN -> ReLU -> maxpool 3x3 s2", "64x56x56"],
            ["layer1", "3 bottlenecks (out 256)", "256x56x56"],
            ["layer2", "4 bottlenecks (out 512, first stride 2)", "512x28x28"],
            ["layer3", "6 bottlenecks (out 1024, first stride 2)", "1024x14x14"],
            ["layer4", "3 bottlenecks (out 2048, first stride 2)", "2048x7x7"],
            ["pool", "global average pool (= linear-probe feature)", "2048"],
            ["head", "fc(2048 -> 20), 40,980 params", "20 logits"],
        ],
        "chars": {
            "FLOPs (paper)": "3.8 x 10^9 multiply-adds at 224 px (He et al. 2015, Table 1)",
            "Input here": "224 x 224",
            "Inductive bias": "convolutional locality + translation equivariance; BatchNorm",
            "Data appetite": "moderate; transfers from ImageNet-1k",
            "Pretraining": "supervised ImageNet-1k (1.28M images), RSB A1 recipe",
            "Strengths": "robust, well understood, fast, tolerates a high fine-tuning LR",
            "Weaknesses": "texture bias; supervised-1k features are weak as frozen features here",
        },
        "hp_keys": ["train.lr", "train.weight_decay", "train.freeze_epochs", "train.layer_decay", "train.batch_size",
                    "aug.randaugment"],
        "regimes": "Linear probe on the 2048-d pooled feature, and full fine-tune (default config).",
        "lesson": "Fine-tuning beats its frozen probe by the largest margin of any backbone (+6.5 points, 5/5 folds): "
                  "ImageNet-1k features need adapting to this domain.",
    },
    {
        "id": "efficientnet", "name": "EfficientNetV2-S", "short": "EfficientNet", "group": "ImageNet CNN",
        "doc": "efficientnet", "config": "efficientnet_v2_s", "diagram": "efficientnet", "year": "2019 / 2021",
        "papers": ["M. Tan, Q. V. Le, *EfficientNet: Rethinking Model Scaling for CNNs*, arXiv:1905.11946, 2019",
                   "M. Tan, Q. V. Le, *EfficientNetV2: Smaller Models and Faster Training*, arXiv:2104.00298, 2021",
                   "J. Hu et al., *Squeeze-and-Excitation Networks*, arXiv:1709.01507, 2017"],
        "bench": ["efficientnet_v2_s"],
        "tagline": "Search a good small block, then scale depth, width and resolution together.",
        "idea": [
            "EfficientNet found a small base network by architecture search and grew it by **compound scaling**: "
            "depth, width and input resolution increase together by fixed ratios, so resolution is part of the "
            "model. Its building block is **MBConv** (inverted residual: 1x1 expand -> depthwise 3x3 -> "
            "squeeze-and-excitation -> 1x1 project).",
            "**EfficientNetV2** replaces the slow-to-train depthwise blocks in the early, high-resolution stages "
            "with **Fused-MBConv** (one regular 3x3 conv instead of expand + depthwise), and proposes progressive "
            "learning. **Squeeze-and-excitation** re-weights channels with a tiny gating network computed from the "
            "globally pooled feature.",
        ],
        "flow_title": "How a picture flows through it (300 px; block counts from docs/models/efficientnet.md)",
        "flow": [
            ["input", "mean = std = 0.5 (TF port), 'same' padding", "3x300x300"],
            ["stem", "conv3x3 (->24, stride 2)", "24x150x150"],
            ["stage 0", "2 x Fused-MBConv1, 24 ch", "24x150x150"],
            ["stage 1", "4 x Fused-MBConv, 48 ch, stride 2", "48x75x75"],
            ["stage 2", "4 x Fused-MBConv, 64 ch, stride 2", "64x38x38"],
            ["stage 3", "6 x MBConv + SE, 128 ch, stride 2", "128x19x19"],
            ["stage 4", "9 x MBConv + SE, 160 ch", "160x19x19"],
            ["stage 5", "15 x MBConv + SE, 256 ch, stride 2", "256x10x10"],
            ["head conv", "conv1x1 (256->1280) -> BN -> act -> global avg pool", "1280"],
            ["classifier", "Linear(1280 -> 20)", "20 logits"],
        ],
        "chars": {
            "FLOPs (paper)": "8.8 x 10^9 at the paper's evaluation size (Tan & Le 2021)",
            "Input here": "300 x 300 (native)",
            "Inductive bias": "convolutional; channel attention (SE); search-chosen block mix",
            "Data appetite": "moderate",
            "Pretraining": "supervised ImageNet-21k -> fine-tuned ImageNet-1k (TF release)",
            "Strengths": "good accuracy per FLOP; exports cleanly",
            "Weaknesses": "BatchNorm + depthwise stack fragile at small batches; high VRAM per image at 300 px",
        },
        "hp_keys": ["model.kwargs", "train.batch_size", "train.lr", "train.freeze_epochs", "train.layer_decay"],
        "regimes": "Full fine-tune only (no probe was run for this family).",
        "lesson": "Highest peak VRAM of the CNNs (300 px input) for a mid-table score; the schedule/BatchNorm fix "
                  "moved it up on 2 of 3 folds.",
    },
    {
        "id": "mobilenet", "name": "MobileNetV3-Large", "short": "MobileNet", "group": "ImageNet CNN",
        "doc": "mobilenet", "config": "mobilenetv3_large", "diagram": "mobilenet", "year": "2017-2019",
        "papers": ["A. G. Howard et al., *MobileNets*, arXiv:1704.04861, 2017",
                   "M. Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, arXiv:1801.04381, 2018",
                   "A. Howard et al., *Searching for MobileNetV3*, arXiv:1905.02244, 2019"],
        "bench": ["mobilenetv3_large"],
        "tagline": "Depthwise-separable convolutions: accuracy per millisecond on a phone CPU.",
        "idea": [
            "A standard k x k convolution mixes space and channels at once (k^2 * C_in * C_out multiply-adds per "
            "pixel). **Depthwise-separable** convolution splits it into a per-channel k x k filter plus a 1x1 "
            "channel mixer -- about 8-9x cheaper for k = 3.",
            "V2 added the **inverted residual** (thin -> expand -> depthwise -> *linear* project -> thin, shortcut "
            "between thin ends). V3 found its block choices by platform-aware search, added squeeze-and-excitation "
            "in some blocks and the cheap **hard-swish** activation x * ReLU6(x+3)/6.",
        ],
        "flow_title": "How a picture flows through it (224 px; stage layout from docs/models/mobilenet.md, "
                      "channel widths from Howard et al. 2019, Table 1)",
        "flow": [
            ["input", "ImageNet mean/std", "3x224x224"],
            ["stem", "conv3x3 (3->16) stride 2 -> BN -> hard-swish", "16x112x112"],
            ["stage 0", "1 x depthwise-separable conv, ReLU", "16x112x112"],
            ["stage 1", "2 x inverted residual, ReLU", "24x56x56"],
            ["stage 2", "3 x inverted residual + SE, ReLU", "40x28x28"],
            ["stage 3", "4 x inverted residual, hard-swish", "80x14x14"],
            ["stage 4", "2 x inverted residual + SE, hard-swish", "112x14x14"],
            ["stage 5", "3 x inverted residual + SE, hard-swish", "160x7x7"],
            ["stage 6", "conv1x1 -> 960 ch, hard-swish", "960x7x7"],
            ["pool + conv_head", "global avg pool -> 1x1 conv (960->1280) -> hard-swish (= probe feature)", "1280"],
            ["classifier", "Linear(1280 -> 20), 25,620 params", "20 logits"],
        ],
        "chars": {
            "FLOPs (paper)": "219 M multiply-adds at 224 px (Howard et al. 2019)",
            "Input here": "224 x 224",
            "Inductive bias": "convolutional; depthwise factorisation; SE in some blocks",
            "Data appetite": "moderate; limited capacity for fine-grained cues",
            "Pretraining": "supervised ImageNet-1k (timm `ra_in1k` RandAugment recipe)",
            "Strengths": "fastest family here (under a minute of fine-tuning per fold); candidate for CPU/OpenCV-DNN",
            "Weaknesses": "lowest pretrained-model score; sensitive to a high LR after unfreezing",
        },
        "hp_keys": ["train.batch_size", "train.lr", "train.weight_decay", "train.freeze_epochs"],
        "regimes": "Full fine-tune only.",
        "lesson": "The latency end of the trade-off. The report's future work proposes distilling the SigLIP model "
                  "into a network of this size for real-time OpenCV-DNN use.",
    },
    {
        "id": "convnext", "name": "ConvNeXt-T", "short": "ConvNeXt", "group": "ImageNet CNN",
        "doc": "convnext", "config": "convnext_tiny", "diagram": "convnext", "year": "2022",
        "papers": ["Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, S. Xie, *A ConvNet for the 2020s*, "
                   "arXiv:2201.03545, 2022"],
        "bench": ["convnext_tiny", "probe_convnext_tiny"],
        "tagline": "A ResNet rebuilt step by step with Transformer-era design choices -- and no attention.",
        "idea": [
            "By 2021 Vision Transformers had overtaken CNNs. ConvNeXt asked how much of that came from attention "
            "and how much from everything else. Starting from ResNet-50 it applies, one at a time: a modern "
            "training recipe, Swin-like stage ratios (3:3:9:3), a 4x4 'patchify' stem, depthwise convolution, an "
            "inverted-bottleneck MLP, large 7x7 kernels, GELU, LayerNorm instead of BatchNorm, fewer "
            "normalisations, and separate downsampling layers.",
            "The result matches Swin at similar cost with **no attention at all**: macro design and training "
            "recipe explain much of the ViT advantage.",
        ],
        "flow_title": "How a picture flows through it (ConvNeXt-T, 224 px; docs/models/convnext.md)",
        "flow": [
            ["input", "ImageNet mean/std", "3x224x224"],
            ["stem", "conv4x4 stride 4 (3->96) -> LayerNorm", "96x56x56"],
            ["stage 1", "3 x ConvNeXt block, 96 ch", "96x56x56"],
            ["down + stage 2", "LayerNorm -> conv2x2 s2; 3 x block, 192 ch", "192x28x28"],
            ["down + stage 3", "9 x block, 384 ch", "384x14x14"],
            ["down + stage 4", "3 x block, 768 ch", "768x7x7"],
            ["pool", "global avg pool -> LayerNorm", "768"],
            ["head", "fc(768 -> 20)", "20 logits"],
        ],
        "chars": {
            "FLOPs (paper)": "4.5 x 10^9 at 224 px (Liu et al. 2022)",
            "Input here": "224 x 224",
            "Inductive bias": "convolutional locality with large (7x7) depthwise kernels; LayerNorm, layer scale",
            "Data appetite": "moderate",
            "Pretraining": "supervised ImageNet-22k (~14M images) -> ImageNet-1k (`fb_in22k_ft_in1k`)",
            "Strengths": "strongest ImageNet CNN here; under 2 min/fold; a clean CNN-vs-transformer control",
            "Weaknesses": "large kernels see context -- can learn the board instead of the part",
        },
        "hp_keys": ["model.kwargs", "train.lr", "train.layer_decay", "train.freeze_epochs"],
        "regimes": "Linear probe on the 768-d pooled feature, and full fine-tune with LLRD 0.8.",
        "lesson": "Ties Swin-T and sits just below ViT-B: at matched pretraining source, CNN vs transformer matters "
                  "much less than pretraining data.",
    },
    {
        "id": "vit", "name": "ViT-B/16", "short": "ViT", "group": "ImageNet transformer",
        "doc": "vit", "config": "vit_b16", "diagram": "vit", "year": "2020",
        "papers": ["A. Dosovitskiy et al., *An Image is Worth 16x16 Words*, arXiv:2010.11929, 2020",
                   "A. Steiner et al., *How to train your ViT?*, arXiv:2106.10270, 2021 (the AugReg weights)",
                   "A. Vaswani et al., *Attention Is All You Need*, arXiv:1706.03762, 2017"],
        "bench": ["vit_b16"],
        "tagline": "Cut the image into 16x16 patches, treat them as words, run a Transformer encoder.",
        "idea": [
            "ViT applies a standard Transformer encoder to images with minimal changes: each 16x16 patch is "
            "linearly embedded as a token, a learned position embedding is added, a learnable `[CLS]` token is "
            "prepended, and **multi-head self-attention** lets every token look at every other token from the "
            "first layer -- a global receptive field from the start.",
            "The price is **weak inductive bias** (no built-in locality or translation equivariance). Trained on "
            "ImageNet-1k alone ViT underperforms ResNets; it wins only with large pretraining sets. The same "
            "backbone design underlies DINOv2, CLIP and SigLIP in this repo.",
        ],
        "flow_title": "How a picture flows through it (ViT-B/16, 224 px; docs/models/vit.md)",
        "flow": [
            ["input", "mean = std = 0.5", "3x224x224"],
            ["patch embed", "conv16x16 stride 16 = linear map of each flattened patch", "196x768"],
            ["+ [CLS], + pos", "prepend [CLS]; add learned position embedding (1x197x768)", "197x768"],
            ["12 x block", "t = t + MHSA(LN(t)) [12 heads x 64]; t = t + MLP(LN(t)) [768->3072->768, GELU]",
             "197x768"],
            ["norm + pool", "final LayerNorm, take [CLS] (global_pool='token')", "768"],
            ["head", "Linear(768 -> 20)", "20 logits"],
        ],
        "chars": {
            "FLOPs (paper)": "about 17.6 x 10^9 at 224 px (reported for the identical DeiT-B architecture, "
                             "Touvron et al. 2021)",
            "Input here": "224 x 224 (196 patch tokens)",
            "Inductive bias": "weak: only the patch grid; attention is global and content-dependent",
            "Data appetite": "very high -- relies on pretraining scale",
            "Pretraining": "supervised ImageNet-21k (AugReg recipe) -> ImageNet-1k",
            "Strengths": "best ImageNet-pretrained fine-tune here; resolution change by interpolating position "
                         "embeddings",
            "Weaknesses": "unstable at high LR without LLRD; small parts cover few patches; attention cost grows "
                          "with the square of the token count",
        },
        "hp_keys": ["train.lr", "train.layer_decay", "train.batch_size", "train.accum_steps", "train.freeze_epochs",
                    "model.kwargs"],
        "regimes": "Full fine-tune with LLRD 0.75 (no probe was run for the supervised ViT).",
        "lesson": "Same ViT-B shape (86-93M parameters) as DINOv2-B, CLIP-B and SigLIP 2-B, yet the lowest of the four: "
                  "pretraining data and objective, not architecture, set the ceiling.",
    },
    {
        "id": "swin", "name": "Swin-T", "short": "Swin", "group": "ImageNet transformer",
        "doc": "swin", "config": "swin_t", "diagram": "swin", "year": "2021",
        "papers": ["Z. Liu et al., *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows*, "
                   "arXiv:2103.14030, 2021"],
        "bench": ["swin_t"],
        "tagline": "Attention inside local 7x7 windows, shifted every other block, in a CNN-like pyramid.",
        "idea": [
            "Global attention is quadratic in token count and ViT has one coarse resolution, which is awkward for "
            "detection and segmentation. Swin restricts self-attention to non-overlapping **local windows** of 7x7 "
            "tokens (cost linear in image area) and **shifts** the window grid by half a window in every other "
            "block so information crosses window borders.",
            "**Patch merging** halves the resolution and doubles the channels between stages (strides 4, 8, 16, "
            "32), giving a CNN-like feature pyramid. Locality and hierarchy -- the inductive biases ViT dropped -- "
            "come back.",
        ],
        "flow_title": "How a picture flows through it (Swin-T, 224 px; docs/models/swin.md)",
        "flow": [
            ["input", "ImageNet mean/std", "3x224x224"],
            ["patch embed", "4x4 patches -> linear", "56x56 tokens x 96"],
            ["stage 1", "2 blocks (W-MSA, SW-MSA), C=96, 3 heads", "56x56x96"],
            ["merge + stage 2", "concat 2x2 neighbours -> 4C -> linear -> 2C; 2 blocks, 6 heads", "28x28x192"],
            ["merge + stage 3", "6 blocks, 12 heads", "14x14x384"],
            ["merge + stage 4", "2 blocks, 24 heads", "7x7x768"],
            ["pool", "LayerNorm -> global average pool", "768"],
            ["head", "fc(768 -> 20)", "20 logits"],
        ],
        "chars": {
            "FLOPs (paper)": "4.5 x 10^9 at 224 px (Liu et al. 2021)",
            "Input here": "224 x 224",
            "Inductive bias": "locality (windows) + hierarchy (patch merging); relative position bias",
            "Data appetite": "lower than ViT at the same size",
            "Pretraining": "supervised ImageNet-22k -> ImageNet-1k (`ms_in22k_ft_in1k`)",
            "Strengths": "multi-scale features; linear cost in image area",
            "Weaknesses": "window layout tied to resolution; only 2 blocks in the last stage",
        },
        "hp_keys": ["model.kwargs", "train.lr", "train.layer_decay", "train.freeze_epochs"],
        "regimes": "Full fine-tune with LLRD 0.8 (recipe matched to ConvNeXt-T).",
        "lesson": "Swin-T and ConvNeXt-T share the recipe and nearly the same size and FLOPs; their CV scores are "
                  "within fold noise of each other -- a controlled 'attention vs convolution' comparison that comes "
                  "out a tie.",
    },
    {
        "id": "dinov2", "name": "DINOv2 (ViT-S/B/L-14)", "short": "DINOv2", "group": "Self-supervised",
        "doc": "dinov2", "config": "dinov2_vitb14", "diagram": "dinov2", "year": "2023",
        "papers": ["M. Oquab et al., *DINOv2: Learning Robust Visual Features without Supervision*, "
                   "arXiv:2304.07193, 2023",
                   "M. Caron et al., *Emerging Properties in Self-Supervised Vision Transformers* (DINO), "
                   "arXiv:2104.14294, 2021",
                   "J. Zhou et al., *iBOT: Image BERT Pre-Training with Online Tokenizer*, arXiv:2111.07832, 2021"],
        "bench": ["dinov2_vitb14", "probe_dinov2_vitl14", "probe_dinov2_vitb14", "probe_dinov2_vits14"],
        "tagline": "No labels: a student matches an EMA teacher across crops and masked patches.",
        "idea": [
            "**Self-supervised** learning takes its training signal from the images themselves. In **DINO** a "
            "student network sees several crops of an image (2 large global + several small local); a **teacher** "
            "-- an exponential moving average (EMA) of the student -- sees the global crops. The student is trained "
            "to match the teacher's softmax over learned prototypes, with centring and sharpening of the teacher "
            "to avoid collapse. Matching local to global crops forces features to encode *what the object is*.",
            "**iBOT** adds masked-patch prediction (the student predicts the teacher's per-patch outputs at masked "
            "positions), which gives good dense features. DINOv2's contribution is mostly scale and curation: "
            "LVD-142M (142M curated, de-duplicated images), a ViT-g/14 teacher, and smaller ViT-S/B/L models "
            "**distilled** from it.",
        ],
        "flow_title": "How a picture flows through it (ViT-B/14 at the 224 px used here; docs/models/dinov2.md)",
        "flow": [
            ["input", "ImageNet mean/std; 224 px (native 518)", "3x224x224"],
            ["patch embed", "conv14x14 stride 14 -> 16x16 grid", "256x768"],
            ["+ [CLS], + pos", "position embedding interpolated from the 518 px / 37x37 grid", "257x768"],
            ["12 x block", "t = t + g1*MHSA(LN(t)); t = t + g2*MLP(LN(t)) (LayerScale g, 12 heads)", "257x768"],
            ["norm + pool", "final LayerNorm, [CLS] token", "768"],
            ["head", "Linear(768 -> 20)", "20 logits"],
        ],
        "chars": {
            "Input here": "224 x 224 (position embeddings interpolated from 518)",
            "Inductive bias": "ViT (weak); LayerScale for stable deep training",
            "Data appetite": "very high in pretraining; designed for frozen-feature use downstream",
            "Pretraining": "LVD-142M, no labels; DINO + iBOT losses; S/B/L distilled from ViT-g/14",
            "FLOPs": "not reported here",
            "Strengths": "strong general-purpose and dense features without any labels",
            "Weaknesses": "encodes everything, including off-task content (a Nintendo Switch as a games console); "
                          "small parts cover few 14-px tokens at 224 px",
        },
        "hp_keys": ["train.lr", "train.layer_decay", "train.batch_size", "train.accum_steps", "train.freeze_epochs"],
        "regimes": "Linear probes for ViT-S/B/L-14 and a full fine-tune of ViT-B/14 (LLRD 0.75).",
        "lesson": "Clearly weaker than CLIP/SigLIP as a frozen feature here, and its fine-tune beats its probe "
                  "(+3.6 points, 5/5 folds) -- unlike the 'frozen DINOv2 rivals fine-tuning' result on standard "
                  "benchmarks.",
    },
    {
        "id": "clip", "name": "CLIP (ViT-B/16, ViT-L/14)", "short": "CLIP", "group": "Vision-language",
        "doc": "clip", "config": "clip_vitb16", "diagram": "clip", "year": "2021",
        "papers": ["A. Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, "
                   "arXiv:2103.00020, 2021"],
        "bench": ["clip_vitb16", "probe_clip_vitl14_336", "probe_clip_vitl14", "probe_clip_vitb16",
                  "zeroshot_clip_vitl14", "zeroshot_clip_vitl14_336", "zeroshot_clip_vitb16"],
        "tagline": "Two encoders trained so that an image and its own caption land on nearby unit vectors.",
        "idea": [
            "CLIP trains an **image encoder** and a **text encoder** jointly on 400M web (image, caption) pairs. In "
            "each batch of N pairs, every image must pick its own caption out of N (softmax over a row of the "
            "N x N similarity matrix) and every caption its own image (softmax over a column): a symmetric "
            "contrastive (InfoNCE) loss with a learned temperature, trained with very large batches (32k).",
            "Because classes are just text, the trained model classifies **zero-shot**: embed 'a photo of a "
            "resistor.' for every class and pick the class whose text embedding is closest to the image.",
        ],
        "flow_title": "How a picture flows through it (ViT-B/16 image tower at 224 px; docs/models/clip.md)",
        "flow": [
            ["input", "CLIP mean (0.4815, 0.4578, 0.4082) / std (0.2686, 0.2613, 0.2758)", "3x224x224"],
            ["patch embed", "16x16 patches", "196x768"],
            ["+ [CLS], + pos, ln_pre", "LayerNorm before the blocks", "197x768"],
            ["12 x block", "pre-norm Transformer, QuickGELU MLP (x * sigmoid(1.702x))", "197x768"],
            ["norm + pool", "LayerNorm -> [CLS] (= 768-d probe feature)", "768"],
            ["fine-tune head", "Linear(768 -> 20), 15,380 params (projection dropped)", "20 logits"],
            ["zero-shot path", "projection -> 512 (B) joint space; cosine vs 20 template-averaged text embeddings",
             "20 scores"],
        ],
        "chars": {
            "Input here": "224 x 224 (ViT-L/14 also at 336 for probes / zero-shot)",
            "Inductive bias": "ViT (weak); language supervision defines the feature space",
            "Data appetite": "web-scale pretraining (400M pairs); the downstream task can be zero-shot",
            "Pretraining": "OpenAI WIT, 400M image-text pairs, symmetric softmax contrastive loss",
            "FLOPs": "not reported here",
            "Strengths": "open-vocabulary zero-shot; strong on product-style web photos",
            "Weaknesses": "homonyms in zero-shot (switch, transformer, relay, fuse); features 'forgotten' by "
                          "fine-tuning at normal LRs; needs QuickGELU to load OpenAI weights correctly",
        },
        "hp_keys": ["train.lr", "train.layer_decay", "train.freeze_epochs", "train.weight_decay", "train.batch_size",
                    "train.accum_steps"],
        "regimes": "Zero-shot (B/16, L/14, L/14@336), linear probes (same three) and a full fine-tune of B/16 with "
                   "low LR and strong LLRD.",
        "lesson": "CLIP-B fine-tuning beats its probe (+4.0 points, 5/5 folds), unlike SigLIP: whether to freeze "
                  "depends on how good the frozen feature already is.",
    },
    {
        "id": "siglip", "name": "SigLIP / SigLIP 2", "short": "SigLIP", "group": "Vision-language",
        "doc": "siglip", "config": "siglip_b16", "diagram": "siglip", "year": "2023 / 2025",
        "papers": ["X. Zhai, B. Mustafa, A. Kolesnikov, L. Beyer, *Sigmoid Loss for Language Image Pre-Training*, "
                   "arXiv:2303.15343, 2023",
                   "M. Tschannen et al., *SigLIP 2*, arXiv:2502.14786, 2025",
                   "I. Alabdulmohsin et al., *Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design*, "
                   "arXiv:2305.13035, 2023 (the shape-optimised So400m tower)"],
        "bench": ["probe_siglip_so400m_384", "probe_siglip2_so400m_384_tta", "probe_siglip_so400m_384_tta",
                  "probe_siglip2_so400m_384", "zeroshot_siglip_so400m_384", "zeroshot_siglip2_so400m_384",
                  "siglip_so400m_384_partial_v2", "siglip_so400m_384_partial", "probe_siglip2_b16",
                  "probe_siglip_b16", "zeroshot_siglip2_b16", "zeroshot_siglip_b16", "siglip2_b16"],
        "tagline": "CLIP with an independent yes/no (sigmoid) decision per image-text pair -- and the final model.",
        "idea": [
            "CLIP's softmax normalises over the whole batch, so it needs huge batches and an all-gather of all "
            "embeddings. **SigLIP** turns every (image i, text j) pair in the batch into an independent **binary** "
            "decision: loss_ij = -log sigmoid(z_ij * (t * I_i . T_j + b)), with z_ij = +1 on the diagonal and -1 "
            "elsewhere, a learned temperature t and a bias b that offsets the N positives vs N^2 - N negatives. No "
            "batch-wide normalisation is needed.",
            "SigLIP image towers pool with a **MAP head** (multihead attention pooling: one learned query attends "
            "over all patch tokens) instead of a [CLS] token. **SigLIP 2** keeps the sigmoid loss and adds "
            "captioning-decoder, self-distillation and masked-prediction objectives plus more multilingual, "
            "de-biased data. The **So400m** tower is a shape-optimised ViT (width 1152, depth 27).",
        ],
        "flow_title": "How a picture flows through the FINAL model (SigLIP So400m/14 @ 384 px; shapes from timm "
                      "1.0.30 `vit_so400m_patch14_siglip_384` and exports/final/preprocess.json)",
        "flow": [
            ["input", "RGB; resize short side to 427 (bicubic); centre crop 384; x/255; mean = std = 0.5", "3x384x384"],
            ["patch embed", "conv14x14 stride 14 -> 27x27 grid (no [CLS] token)", "729x1152"],
            ["+ pos", "learned position embedding", "729x1152"],
            ["27 x block", "pre-norm Transformer, 16 heads x 72, MLP 1152->4304->1152 (mlp_ratio 3.7362)",
             "729x1152"],
            ["norm + MAP pool", "final LayerNorm; one learned query attends over 729 tokens -> MLP", "1152"],
            ["probe head", "standardise -> logistic regression (C = 0.01) -> log-softmax", "20"],
            ["zero-shot head", "log-softmax(50 * cos(f, template-averaged class text embeddings))", "20"],
            ["fusion + temperature", "(log p_probe + 0.25 * log p_zs) / T, T = 0.8977 -> softmax", "20 probs"],
            ["abstain", "answer only if max prob >= 0.5, else UNSURE", "class or UNSURE"],
        ],
        "chars": {
            "Input here": "224 x 224 (B/16) and 384 x 384 (So400m/14)",
            "Inductive bias": "ViT (weak); MAP attention pooling; language-defined feature space",
            "Data appetite": "web-scale pretraining (WebLI); downstream works frozen or zero-shot",
            "Pretraining": "WebLI image-text, pairwise sigmoid loss; SigLIP 2 adds captioning, self-distillation, "
                           "masked prediction",
            "FLOPs": "not reported here (CPU latency of the exported So400m graph: see Final model tab)",
            "Strengths": "best frozen features in this project by a wide margin; strong zero-shot",
            "Weaknesses": "So400m is large (1.6 GB ONNX, ~0.9 s/image on CPU; too big for full fine-tuning on 8 GB); "
                          "same homonym issue as CLIP; fine-tuning did not beat the probe",
        },
        "hp_keys": ["train.lr", "train.layer_decay", "train.freeze_epochs", "train.weight_decay", "train.batch_size",
                    "train.accum_steps"],
        "extra_config": "siglip_so400m_384_partial_v2",
        "regimes": "Zero-shot, linear probe (with and without flip-TTA), full fine-tune (B/16 SigLIP 2) and a "
                   "partial fine-tune of So400m (last 4 of 27 blocks + pooling + head).",
        "lesson": "The top seven methods are all SigLIP / SigLIP 2 So400m variants; the frozen probe beats every "
                  "fine-tune, including the partial fine-tune of the same backbone. The final model is this probe "
                  "plus the backbone's own zero-shot head.",
    },
]

# Explanations for config keys (what the knob does and what changing it does). Source: configs/base.yaml comments,
# docs/concepts/transfer_learning.md and the per-family docs.
HP_EXPLAIN = {
    "model.kwargs": "Passed to timm.create_model. `drop_path_rate` = stochastic depth: randomly skips residual "
                    "branches during training (a regulariser; higher = stronger). For the scratch CNN: `width` "
                    "(channels of the first stage; 2x width ~ 4x parameters) and `dropout` before the head.",
    "train.epochs": "Maximum epochs. Early stopping (`train.patience`) usually ends runs sooner.",
    "train.patience": "Stop after this many epochs without a new best fold-validation macro-F1.",
    "train.lr": "Peak learning rate after warm-up (for the head when `layer_decay` is set). Too high destroys "
                "pretrained features in the first epochs after unfreezing; too low under-fits.",
    "train.weight_decay": "AdamW decoupled decay. It pulls weights toward zero, i.e. away from the pretrained "
                          "solution; lighter decay keeps them closer to it.",
    "train.warmup_epochs": "Linear LR warm-up before cosine decay. Since commit d6d4e09 the warm-up restarts when "
                           "the backbone unfreezes.",
    "train.freeze_epochs": "Head-only epochs before the backbone unfreezes (LP-FT style), so a random head does not "
                           "send large gradients into pretrained weights. Since d6d4e09 BatchNorm statistics are "
                           "also frozen during this phase.",
    "train.layer_decay": "Layer-wise LR decay (LLRD): layer group i from the top gets lr * decay^i. Smaller values "
                         "keep lower (more generic) layers closer to their pretrained weights. timm models only; "
                         "null = off.",
    "train.batch_size": "Micro-batch per step; with `accum_steps` gives the effective batch. Small batches make "
                        "BatchNorm statistics noisy.",
    "train.accum_steps": "Gradient accumulation: effective batch = batch_size x accum_steps, to fit 8 GB VRAM.",
    "train.label_smoothing": "Target 1 - eps for the true class and eps spread over the rest; curbs "
                             "over-confidence.",
    "train.trainable_blocks": "Partial fine-tune: only the last N transformer blocks (+ pooling + head) receive "
                              "gradients; the rest stay frozen (no optimizer state).",
    "train.grad_checkpointing": "Recompute activations in the backward pass: ~30 % slower, much less memory.",
    "aug.rrc_scale": "RandomResizedCrop area range. A lower minimum = more aggressive crops (stronger "
                     "augmentation).",
    "aug.randaugment": "RandAugment: n random operations per image at magnitude m; null disables it.",
    "aug.color_jitter": "Random brightness/contrast/saturation jitter strength.",
}

# --------------------------------------------------------------------------------------------------------------
# Glossary (searchable).
# --------------------------------------------------------------------------------------------------------------
GLOSSARY = [
    ("Abstention / selective classification", "The model may answer UNSURE instead of a class. Here: answer only if "
     "the calibrated maximum probability is at least 0.5 (`abstain_threshold`)."),
    ("Accepted accuracy", "Accuracy computed only on the images the model chose to answer."),
    ("Accuracy", "Fraction of images whose predicted class equals the label."),
    ("AdamW", "Adam optimiser with weight decay applied directly to the weights (decoupled); used for every "
     "fine-tune."),
    ("AMP (fp16)", "Automatic mixed precision: most maths in 16-bit floats for speed and memory, with a gradient "
     "scaler against underflow."),
    ("BatchNorm", "Normalises each channel by batch statistics during training and by running averages at test "
     "time; used in ResNet, EfficientNet, MobileNet and the scratch CNN."),
    ("Calibration", "A model is calibrated if, among predictions made with confidence p, a fraction p is correct."),
    ("[CLS] token", "A learnable extra token whose final state summarises the image (ViT, DINOv2, CLIP). SigLIP uses "
     "a MAP head instead."),
    ("Cluster (duplicate cluster)", "A group of rows that are copies or near-copies of each other. Splits and folds "
     "assign whole clusters, never parts of one."),
    ("Compound scaling", "EfficientNet's rule of growing depth, width and resolution together by fixed ratios."),
    ("Confident learning", "Finding likely label errors by comparing each row's out-of-fold predicted probabilities "
     "with its given label (Northcutt et al. 2021)."),
    ("Confusion matrix", "Table of counts: row = true class, column = predicted class. The diagonal holds the "
     "correct predictions."),
    ("Contrastive loss (InfoNCE)", "CLIP's objective: in a batch, each image must pick its own caption among all "
     "captions (softmax), and vice versa."),
    ("Cosine similarity", "Dot product of two L2-normalised vectors; 1 = same direction. Used for zero-shot scores "
     "and for near-duplicate detection."),
    ("Coverage", "Fraction of images the model answers (does not abstain on)."),
    ("Cross-validation (grouped, 5-fold)", "The dev rows are split into 5 folds by cluster; each fold is "
     "validated once by a model trained on the other four. Scores are mean +/- SD over folds."),
    ("Data leakage", "Information from evaluation data reaching training, e.g. the same image in train and test. "
     "It inflates scores without improving the model."),
    ("Depthwise-separable convolution", "A per-channel spatial filter followed by a 1x1 channel mixer; ~8-9x "
     "cheaper than a full 3x3 convolution (MobileNet, EfficientNet; ConvNeXt uses a 7x7 depthwise conv)."),
    ("Dev split", "The rows used for all training and model selection (5 grouped folds)."),
    ("Distillation", "Training a student network to match a teacher's outputs (DINOv2's S/B/L models are distilled "
     "from ViT-g/14; the report's future work proposes distilling the final model into a small CNN)."),
    ("Drop path / stochastic depth", "Randomly skipping residual branches per sample during training "
     "(`drop_path_rate`)."),
    ("Early stopping", "Keeping the epoch with the best validation macro-F1 and stopping after `patience` epochs "
     "without improvement. Here the validation fold is also the scoring fold, which slightly flatters fine-tunes."),
    ("ECE (expected calibration error)", "Average |accuracy - confidence| over confidence bins, weighted by bin size "
     "(15 equal-width bins in `compclass.final`)."),
    ("EMA teacher", "A copy of the network whose weights are an exponential moving average of the student's (DINO)."),
    ("Exact duplicate", "Byte-identical files (same SHA-256)."),
    ("Fine-tuning", "Continuing to train a pretrained network (all or some layers) with a new head on the target "
     "task."),
    ("Fusion (logit fusion)", "Adding the zero-shot head's log-probabilities (weight 0.25) to the probe's before "
     "temperature scaling."),
    ("Gradient accumulation", "Summing gradients over several micro-batches before one optimiser step."),
    ("Gradient checkpointing", "Not storing intermediate activations and recomputing them in the backward pass."),
    ("Hard-swish", "x * ReLU6(x + 3) / 6, a cheap approximation of swish used in MobileNetV3."),
    ("HOG", "Histogram of Oriented Gradients: per-cell histograms of edge directions, block-normalised."),
    ("In-scope accuracy", "Accuracy over rows without the automated `ood_suspect` flag."),
    ("Inductive bias", "Assumptions built into an architecture (e.g. locality and weight sharing in CNNs) that "
     "reduce how much data it needs."),
    ("Inverted residual (MBConv)", "Thin -> 1x1 expand -> depthwise conv -> (SE) -> linear 1x1 project -> thin, "
     "with a shortcut between the thin ends."),
    ("Label smoothing", "Training targets of 1 - eps for the true class (eps = 0.1 here) to curb over-confidence."),
    ("LayerNorm", "Normalises each token's features; used in transformers and ConvNeXt."),
    ("LayerScale", "A learned per-channel factor on each residual branch, initialised small (DINOv2, ConvNeXt)."),
    ("Linear probe", "Freeze the backbone, extract one feature vector per image, train only a linear classifier "
     "(standardised logistic regression with C chosen by inner grouped CV)."),
    ("LLRD (layer-wise LR decay)", "Lower layers get geometrically smaller learning rates (`train.layer_decay`)."),
    ("Macro-F1", "Mean over the 20 classes of the per-class F1 (harmonic mean of precision and recall). The "
     "primary metric: every class counts equally."),
    ("MAP head", "Multihead attention pooling: a learned query token attends over all patch tokens to produce the "
     "image feature (SigLIP)."),
    ("Near-duplicate", "Visually the same picture with small edits (crop, resize, watermark), detected with pHash "
     "and CLIP cosine."),
    ("Noise floor", "The size of difference that sampling noise alone can produce (e.g. +/-1.6 points for the "
     "3-fold schedule ablation; +/-1.8 points SE on the 418-image test)."),
    ("ONNX / onnxruntime", "A portable graph format for neural networks and Microsoft's engine for running it. The "
     "final model is one ONNX graph (1.6 GB)."),
    ("OOD / out-of-scope", "Images that do not show an electronic component (consoles, logos, cartoons, people). "
     "Flagged automatically by a CLIP distractor screen (threshold 0.25); not human-verified."),
    ("Out-of-fold (OOF) prediction", "A prediction for a dev row made by the model trained on the other folds."),
    ("Pareto front", "Methods for which no other method is both cheaper and more accurate."),
    ("Partial fine-tune", "Training only the last N blocks of a large backbone (`train.trainable_blocks`)."),
    ("Patch embedding", "Cutting an image into fixed-size patches and linearly projecting each to a token vector."),
    ("Patch merging", "Swin's downsampling: concatenate 2x2 neighbouring tokens and project to twice the channels."),
    ("pHash", "Perceptual hash: a 64-bit fingerprint of an image's low-frequency content; similar images have a "
     "small Hamming distance."),
    ("Position embedding", "Learned vectors added to tokens so the transformer knows where each patch came from."),
    ("Precision / recall", "Precision: of the images predicted as class c, the fraction that are c. Recall: of the "
     "images of class c, the fraction predicted as c."),
    ("Prompt template", "Text such as 'a photo of a {class}.'; zero-shot averages the embeddings of several "
     "templates per class."),
    ("RandAugment", "Applies n random image operations at magnitude m (n = 2, m = 9 by default here)."),
    ("Reliability diagram", "Accuracy vs mean confidence per confidence bin; a calibrated model lies on the "
     "diagonal."),
    ("Residual connection", "y = x + F(x): a block learns a correction to its input (ResNet and every transformer)."),
    ("Sealed test", "Rows held out from all training and selection, read exactly once for the final model."),
    ("Self-attention", "Every token computes a weighted average of all tokens' values, with weights from "
     "query-key similarity."),
    ("Shifted windows", "Swin alternates window grids offset by half a window so information crosses window "
     "borders."),
    ("Sigmoid loss", "SigLIP's objective: an independent binary match / no-match decision for every image-text pair "
     "in the batch."),
    ("Sign test", "Counts how many paired differences are positive and asks whether that count is unusual if there "
     "is no effect."),
    ("Squeeze-and-excitation (SE)", "Channel attention: global-pool -> small MLP -> sigmoid -> rescale channels."),
    ("Standard error (SE)", "The standard deviation of an estimate across hypothetical resamples; for accuracy "
     "sqrt(p(1-p)/n)."),
    ("Temperature scaling", "Dividing logits by one scalar T fitted on out-of-fold scores to fix over/under-"
     "confidence."),
    ("TTA (test-time augmentation)", "Averaging predictions or features over transformed copies of the input "
     "(here: horizontal flip)."),
    ("Web holdout", "Wikimedia Commons photos collected after the model was frozen; a second sealed set from a "
     "different source."),
    ("Wilson interval", "A confidence interval for a proportion that behaves well for small n or p near 0/1."),
    ("Zero-shot", "Classifying with no training images: compare the image embedding with text embeddings of class "
     "prompts."),
]

# --------------------------------------------------------------------------------------------------------------
# Reproduce: commands per phase (README.md "Reproducing each phase", docs/DEMO.md, script docstrings).
# --------------------------------------------------------------------------------------------------------------
REPRODUCE = [
    ("Set-up", "Laptop or PC", [
        "uv sync                      # .venv from pyproject.toml + uv.lock",
        "uv run pytest -q             # ~10 s CPU smoke tests on a synthetic manifest",
        "export HF_HOME=~/models_cache/hf TORCH_HOME=~/models_cache/torch",
    ]),
    ("Phase 0 - dataset v2", "CPU; expects the Kaggle folders under data/raw/electronic-component-recognizer/", [
        "uv run --no-config tools/build_dataset_v2.py   # data/v2/{manifest.json,leakage_report.json,review/}",
    ]),
    ("Phase 1 - probes, zero-shot, classical", "GPU for feature extraction", [
        "uv run python -m compclass.features --model siglip_so400m_384 --results results/phase1/raw/runs.jsonl",
        "uv run python -m compclass.zeroshot --model siglip_so400m_384 --results results/phase1/raw/runs.jsonl",
        "uv run python -m compclass.classical --results results/phase1/raw/runs.jsonl",
        "uv run python -m compclass.features --model siglip_so400m_384 --tta --results results/phase1b/runs.jsonl",
    ]),
    ("Phase 2 - fine-tune benchmark", "GPU; one YAML per family", [
        "uv run python -m compclass.train --config configs/convnext_tiny.yaml --fold 0 1 2 3 4",
        "uv run python -m compclass.train --config configs/siglip_so400m_384_partial_v2.yaml --fold 0 1 2",
    ]),
    ("Phase 3 - final model", "GPU; the sealed-test look is already spent", [
        "uv run python -m compclass.final fit",
        "uv run python -m compclass.final export",
        "uv run python -m compclass.final test --confirm         # done once: results/final/final_test_v2.json",
        "uv run python -m compclass.final test --manifest data/web_holdout_v1/manifest.json --confirm   # done once",
    ]),
    ("Phase 4 - analysis tables, report figures, demo", "CPU", [
        "uv run python scripts/analysis_tables.py",
        "uv run python scripts/make_report_figures.py",
        "uv run python -m compclass.demo --source <file|folder|url|webcam-index>",
    ]),
    ("Original-Kaggle evaluation", "GPU PC, repo root", [
        "uv run python scripts/eval_original_kaggle.py   # results/kaggle_original/{summary.json,per_file.csv,...}",
    ]),
    ("This explorer", "CPU, seconds; reads only repository files", [
        "uv run --no-config scripts/build_explorer.py   # writes docs/explorer/index.html",
    ]),
]

# --------------------------------------------------------------------------------------------------------------
# Plain-words explanations shown in the Data audit tab.
# --------------------------------------------------------------------------------------------------------------
AUDIT_EXPLAIN = {
    "ood": "**Out-of-scope** images show something other than an electronic component -- a games console under "
           "*switch*, a brand logo under *fuse*, a toy robot under *transformer*. A CLIP screen compared each image "
           "against the 20 class prompts and nine distractor concepts and flagged it when the distractor share "
           "exceeded 0.25. The screen is automated and was not human-verified (e.g. resistor schematics are arguably "
           "in scope). Flagged rows were kept in training and evaluation.",
    "exact": "**Exact duplicates** are byte-identical files saved more than once, often under different file names. "
             "They were collapsed to one row each (SHA-256). Shown here: images whose copies were in *different "
             "legacy splits* -- the same picture sat in the old train set and the old validation/test set.",
    "near": "**Near-duplicates** are the same photo after a resize, crop, re-encode or watermark. They do not share a "
            "hash, so they were linked by perceptual hash (pHash) plus CLIP cosine similarity and put into one "
            "*cluster*; a cluster never straddles a split.",
    "conflict": "**Conflicting labels**: the identical picture appears in two class folders. At most one label can be "
                "right; all 13 exact-duplicate groups were excluded from v2. Two near-duplicate clusters also mix "
                "labels; their 4 rows were kept with a flag.",
    "issues": "**Label issues** (confident learning): out-of-fold probe probabilities say the given label is very "
              "unlikely while another class is very likely. These are model-flagged and unreviewed.",
    "classes": "**Typical in-scope examples**: per class, images the CLIP screen rated in scope with the highest "
               "zero-shot probability for their own label.",
}
