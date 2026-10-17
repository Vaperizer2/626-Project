# Data quality: label noise, out-of-scope web images, and confident learning

The dataset is the Kaggle "electronic-component-recognizer" scrape: 20 class folders filled by image search.
Image search returns whatever matches the *word*, so the labels are only as good as the query. This page explains
the kinds of noise present, what v2 already handles, and how to find the rest.

## 1. Kinds of noise

| kind | example here | effect |
|---|---|---|
| exact duplicates | 587 redundant files in 533 groups | inflated scores via train/test leakage — **fixed in v2** (collapsed) |
| conflicting labels | 13 byte-identical images filed under two classes | no correct answer — **excluded in v2** |
| near-duplicates | 91 clusters, 234 rows; 2 clusters with mixed labels (4 rows flagged `near_dup_label_conflict`) | leakage — **fixed** by grouping clusters into one fold |
| out-of-scope ("homonym") images | Nintendo Switch consoles under `switch`, relay races under `relay`, brand logos under `fuse` (visual spot-checks recorded in `tools/build_dataset_v2.py`) | the label is "right" for the query and wrong for the task — **flagged, not removed** |
| plain label errors | a capacitor in the `diode` folder | wrong supervision; also corrupts evaluation |
| format artefacts | 345 transparent images composited on white, 15 animations (first frame) | style shortcuts (white background) |

Label noise is not unique to scraped data: C. G. Northcutt, A. Athalye, J. Mueller (2021), *Pervasive Label Errors in
Test Sets Destabilize Machine Learning Benchmarks*, arXiv:2103.14749, estimated an average of ≈3.3 % label errors in the
test sets of 10 popular benchmarks.

## 2. The v2 out-of-scope screen

`tools/build_dataset_v2.py` scores every image with OpenCLIP ViT-B/32 (`laion2b_s34b_b79k`) zero-shot against the 20
component concepts **plus 9 distractor concepts** (person, car, food, logo, text_screenshot, cartoon_game_art, circuit_schematic, website_screenshot, collage).
`ood_score` = softmax probability mass on the distractors; rows with `ood_score ≥ 0.25` get the
`ood_suspect` flag. Nothing is excluded automatically: the flag is a **review queue**.

Result: **816 of 2,930 rows (27.8 %) are flagged** — 704 dev rows and
112 of 418 sealed-test rows. Flag rates per class (all splits):

| class | flagged | rows | rate |
|---|---:|---:|---:|
| `switch` | 121 | 177 | 68 % |
| `transformer` | 64 | 128 | 50 % |
| `led` | 55 | 124 | 44 % |
| `transistor` | 61 | 146 | 42 % |
| `relay` | 61 | 156 | 39 % |
| `diode` | 56 | 145 | 39 % |
| `jumper wire` | 62 | 163 | 38 % |
| `crystal oscillator` | 50 | 151 | 33 % |
| `fuse` | 46 | 142 | 32 % |
| `resistor` | 43 | 153 | 28 % |
| `battery holder` | 31 | 155 | 20 % |
| `breadboard` | 22 | 131 | 17 % |
| `inductor coil` | 26 | 161 | 16 % |
| `power supply module` | 23 | 154 | 15 % |
| `capacitor` | 16 | 116 | 14 % |
| `microcontroller board` | 20 | 154 | 13 % |
| `heat sink` | 19 | 160 | 12 % |
| `connector` | 11 | 110 | 10 % |
| `ic chip` | 15 | 157 | 10 % |
| `potentiometer` | 14 | 147 | 10 % |

The flags are suspicions with **unmeasured precision and recall**: a spot check found most flagged images out of scope
even in the 0.2–0.5 score band, but no labelled audit exists yet. Review material is in `data/v2/review/`:
`ood_suspects.csv` (id, label, path, split, fold, `ood_score`, `top_distractor`, `zeroshot_class`, `label_p`, …),
`ood_suspects_top80.jpg` (contact sheet), `zeroshot_scores.csv` (scores for *every* row, so the threshold can be moved),
plus `conflicts.csv`/`.jpg` and `near_dup_*` for the other checks.

## 3. Why it matters for model comparison

* **Self-supervised and vision-language features represent off-topic images well** (a console is a very
  recognisable console), so a probe can *fit* them — CV scores then partly measure "can you recognise the scrape
  pattern of the word *switch*".
* **Out-of-scope rows in the sealed test** (112 / 418) mean the test score mixes two questions. Report
  both subsets ([`evaluation.md`](evaluation.md) §6).
* **Fixing data usually beats changing models** at this scale. If review removes rows, the split fingerprint
  changes; build a new manifest version rather than editing v2 in place (`load_manifest` rejects hand edits).

## 4. Confident learning — finding label errors with OOF probabilities

C. G. Northcutt, L. Jiang, I. L. Chuang (2021), *Confident Learning: Estimating Uncertainty in Dataset Labels*,
JAIR 70, arXiv:1911.00068. Idea: use **out-of-fold** predicted probabilities (never from a model that trained on
the row) to estimate the joint distribution of given vs true labels.

1. For each class j, the threshold t_j = mean predicted probability of class j over rows *labelled* j (the
   model's typical self-confidence for that class).
2. A row labelled i is counted as "probably class j" if p_j ≥ t_j (taking the largest such j). The counts form the
   *confident joint* C[i][j].
3. Off-diagonal entries are candidate label errors; rank them by margin (e.g. p_j − p_i) and review the top ones.

This repo already saves the needed OOF probabilities: `results/oof/probe_<key>.npz` (keys `ids`, `y`, `proba`,
`classes`) after `compclass.features` runs on all 5 folds. A minimal version (no extra dependency):

```python
import numpy as np
z = np.load("results/oof/probe_dinov2_vitb14.npz", allow_pickle=True)
P, y, ids, classes = z["proba"], z["y"], z["ids"], list(z["classes"])
t = np.array([P[y == j, j].mean() for j in range(len(classes))])      # per-class thresholds
above = P >= t                                                         # rows x classes
cand = np.where(above, P, -1).argmax(1)                                # most likely class above threshold
suspect = above.any(1) & (cand != y)
order = np.argsort(-(P[np.arange(len(y)), cand] - P[np.arange(len(y)), y]) * suspect)
for i in order[:25]:
    print(ids[i], classes[y[i]], "->", classes[cand[i]], round(float(P[i, cand[i]]), 3))
```

The `cleanlab` library implements the full method (`uv add cleanlab`). Cross-check suspects against the
`ood_suspect` flag: rows that are *both* are the cheapest wins for human review. Prefer OOF probabilities from a
**probe** (the backbone could not memorise the noisy label) or an ensemble of probes over fine-tuned models.

## 5. Try this

```bash
# per-class flag rates from the manifest
uv run python -c "import json,collections; m=json.load(open('data/v2/manifest.json')); t=collections.Counter(r['label'] for r in m['rows']); f=collections.Counter(r['label'] for r in m['rows'] if 'ood_suspect' in r['flags']); [print(f'{k:22s} {f[k]:4d}/{t[k]}') for k in sorted(t, key=lambda k: -f[k]/t[k])]"
# out-of-fold probabilities for confident learning
uv run python -m compclass.features --model dinov2_vitb14
# zero-shot (also writes results/oof/zeroshot_<key>.npz with cosine scores)
uv run python -m compclass.zeroshot --model clip_vitl14
```
