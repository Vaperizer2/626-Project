# Speed harness (frozen)

This harness measures single-image speed of the component classifier on legion's RTX 3060 Laptop GPU, plus
development-fold accuracy, for every candidate in the speed hillclimb. Only numbers this harness prints count.
Do not edit it on your branch. If it is wrong, report it to the coordinator.

Everything runs on legion from any directory:

```bash
H=/home/vaper/Work/Prototypes/626-wt/harness/tools/speed
$H/speed eval    <candidate_dir>          # dev-fold CV accuracy through your graph and EP
$H/speed time    <candidate_dir>          # batch-1 latency + batch-8 throughput on the GPU
$H/speed compare <reference_dir> <candidate_dir>   # deltas, McNemar, latency ratio, verdict
```

The reference for acceptance is the deployed model:
`/home/vaper/Work/Prototypes/626-artifacts/harness/candidates/baseline_fp32`.
Run `eval` and `time` on your candidate, then
`$H/speed compare /home/vaper/Work/Prototypes/626-artifacts/harness/candidates/baseline_fp32 <candidate_dir>`.

`eval` and `time` take `/tmp/626-gpu-bench.lock` themselves for the GPU part, so they never overlap another
agent's timing. Do not wrap them in `flock` (it works, but is unnecessary). If another job holds the lock they
print `waiting for /tmp/626-gpu-bench.lock` and continue when it is free.

## Candidate contract

A candidate is a directory. Its name is the candidate id in the results, so make it unique
(`<track>_<what>`, for example `distill_vitb16_fp16`). Put it under
`/home/vaper/Work/Prototypes/626-artifacts/<your-track>/`, never in git.

| file | required | content |
|---|---|---|
| `model.onnx` (+ external data files next to it) | yes | backbone graph: input `pixel_values` float32 `[N,3,S,S]` → output `embedding` float32 `[N,D]` |
| `preprocess.json` | yes | how images become `pixel_values` (below) |
| `runtime.json` | no | execution provider and options; default CUDA EP with onnxruntime defaults |

Graph rules, checked on load:

- Exactly one input named `pixel_values` and one output named `embedding`, both float32. A float16 model keeps
  float32 IO (`onnxruntime.transformers.float16.convert_float_to_float16(model, keep_io_types=True)`).
- Batch dimension symbolic (preferred) or a static 1. `time` skips batch-8 throughput for a static-1 graph.
- `D` static. The head is not part of your graph; the harness appends it (below).
- External data files must sit in the candidate directory. The harness reflink-copies the graph into
  `artifacts/harness/work/<id>/<sha12>/` (instant on this btrfs), so the candidate directory is never written.

`preprocess.json` fields (eval and time read exactly these; extra fields are ignored):

```json
{"input_size": 384, "crop_pct": 0.9, "interpolation": "bicubic", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
 "zero_shot_fusion": true, "built_from": "<git sha / recipe note, recorded in results>"}
```

Images are decoded by Pillow, resized so the short side is `round(input_size / crop_pct)` with that
interpolation, centre-cropped to `input_size`, scaled to [0,1], then normalised with `mean`/`std`. This is
`compclass.data.build_transforms(spec, None, train=False)`, the transform the deployed head was fitted on.
The deployed model uses 384 px, short side 427, bicubic, mean = std = 0.5.

`zero_shot_fusion: true` declares that `embedding` is the SigLIP So400m-384 pooled image embedding (D = 1152,
same space as the deployed backbone, for example the same backbone at lower precision, or a student distilled
onto it). The harness then fuses the fixed zero-shot text head into every fitted head, exactly as deployed. With
`false`, the head is the linear probe alone. Fusion is worth about +0.2 pp CV on the deployed backbone, so a
probe-only candidate starts slightly behind.

`runtime.json`:

```json
{"provider": "CUDAExecutionProvider", "options": {}}
{"provider": "TensorrtExecutionProvider",
 "options": {"trt_fp16_enable": true, "trt_engine_cache_enable": true, "trt_engine_cache_path": "trt_cache",
             "trt_timing_cache_enable": true, "trt_timing_cache_path": "trt_cache",
             "trt_max_workspace_size": 2147483648}}
```

- `options` go to the provider unchanged, except `trt_engine_cache_path` / `trt_timing_cache_path`, which are
  resolved inside the harness work dir for that candidate hash. TensorRT profiles default to batch 1 (opt) to
  8 (max) at `input_size`; set `trt_profile_*_shapes` yourself to override.
- Optional keys: `graph_optimization_level` (default `ORT_ENABLE_ALL`) and `session_config` (a dict of
  `add_session_config_entry` pairs).
- TensorRT runs as `[TensorRT, CUDA, CPU]` and CUDA as `[CUDA, CPU]`. The session fails if the requested
  provider did not load. ORT's default CUDA EP has `use_tf32=1`, so "fp32" on CUDA means fp32 weights with TF32
  matmuls. That is what the deployed app gets (`ort::ep::CUDA::default()`), and it is what the baseline uses.

`tools/speed/variants.py` builds the common variants of an existing candidate:

```bash
PY=/home/vaper/Work/Prototypes/626-artifacts/harness/venv/bin/python
$PY $H/variants.py runtime  <src> <dst> '<runtime.json as JSON>'   # same graph, other EP / options
$PY $H/variants.py fp16     <src> <dst>                             # needs ~5 GB RAM; run it on carlpc2
$PY $H/variants.py truncate <src> <dst> --drop K                    # degrade: drop the last K blocks
```

## What eval measures

1. Builds `with_head.onnx`: your graph plus the deployed head (standardise → linear probe → log-softmax, plus
   the zero-shot head when fused, temperature, softmax), outputs `embedding` and `probs`. Both eval and time run
   this file, so eval measures the graph that is timed and TensorRT builds one engine per candidate.
2. Runs all 2,512 development images (manifest fingerprint `431315e87d2612cb`; the sealed test and web holdout
   are never read) through it on your provider, one image per run, and keeps `embedding`.
3. For each of the 5 grouped development folds, fits the deployed head recipe on the other 4 folds
   (`compclass.features.fit_probe`: StandardScaler + logistic regression, C from (0.001, 0.003, 0.01, 0.1, 1) by
   grouped 3-fold CV on macro-F1; then `compclass.final.fused_scores` with T = 50, w = 0.25 and the deployed text
   embeddings) and predicts the held-out fold.
4. Reports mean ± sd (ddof 1) over folds of accuracy and macro-F1, per-fold values with the chosen C, pooled
   out-of-fold accuracy / macro-F1, probe-only accuracy, and, for information, the out-of-fold temperature,
   abstain threshold, coverage and accepted accuracy. Non-finite embeddings are counted in `n_errors` and
   zeroed (they then count as whatever the head predicts on a zero vector).
5. Saves per-image flags (`ids`, `y`, `pred`, `correct`, `fold`) and embeddings under the work dir.

## What time measures

Under the GPU lock, a session on your provider runs `with_head.onnx` with inputs and outputs bound on the GPU
(IO binding), so a run is the model, not host copies. Input is the first dev image (batch 1) or first 8 (batch 8).
Each run is `run_with_iobinding` + `synchronize_outputs`, timed with `perf_counter_ns`.

- A second session holds the baseline. Every batch-1 run of yours is followed by one baseline run: 5 rounds of
  60 pairs. The median of the 300 per-pair ratios (yours / baseline) is the speed number
  (`paired.ratio_median`); `paired.round_ratios` are the 5 per-round medians. legion's GPU runs at its thermal
  and power limits and its SM clock steps every few seconds, so absolute latency drifts (see Baseline and noise);
  the two runs of a pair see the same clock.
- Warmup per session and batch size: at least 20 runs and 10 s.
- Batch 1: reports median, p10, p90, min, max and the 5 round medians of your 300 runs (measured while
  alternating with the baseline).
- Batch 8: 5 rounds x 12 runs, unpaired. Reports images/s = 8 / median batch latency. Skipped for a static-1 graph
  or a TensorRT profile whose max batch is below 8.
- Output checks: every output row finite, probabilities summing to 1, and each embedding within cosine 0.999 of
  the same image run alone through a plain `session.run`. A batch-8 failure sets `batch8.valid = false`; `compare`
  then ignores that throughput. Failed rows count in `n_errors`.
- `ep_assignment` in eval and time records lists the subgraphs and node counts each provider received, so a
  partial fallback to CUDA or CPU is visible.
- Records GPU name, driver, CUDA driver, ORT / cuDNN / TensorRT versions, effective provider options, NVML
  samples every 100 ms (SM and memory clock, temperature, power, utilisation), other GPU processes, load average,
  and the raw per-run latencies (`raw_latencies`, an npz).

## Acceptance rule (encoded in compare)

Candidate B is **ACCEPTABLE** against reference A when both hold:

- mean CV accuracy(B) ≥ mean CV accuracy(A) − 0.5 percentage points, and
- McNemar does not show B significantly worse: it is not the case that B has more discordant errors than A
  (B wrong where A is right, versus the reverse) with exact two-sided McNemar p < 0.05 on the 2,512 paired
  out-of-fold predictions.

Otherwise **REJECT**. Speed is the hillclimb metric; this rule only guards accuracy. `compare` also prints:

- the batch-1 latency ratio B/A, from the same-session ratios to the baseline when both time records have them
  (B/A = ratio(B) / ratio(A), with ratio(baseline) = 1), else from the two absolute medians;
- `beyond noise`: true when B's five round ratios all lie outside A's range (unpaired: the round medians do not
  overlap) and the ratio is at least 3% away from 1. The 3% is above the largest drift of one candidate's paired
  ratio between sessions (2.3%, Baseline and noise). A smaller real gain shows as not beyond noise; re-run
  `time` on both sides at different times before claiming it;
- batch-8 throughputs, where valid.

## Results

Every `eval`, `time` and `compare` appends one JSON line to
`/home/vaper/Work/Prototypes/626-artifacts/harness/results.jsonl` with `kind`, `candidate` (directory name),
`candidate_sha` (hash of `model.onnx`, its external data, `preprocess.json`, `runtime.json`), `git_sha` and
`git_dirty` of this harness (code only; docs are ignored), `n_images`, `n_errors`, `n_timed_runs`, plus the
fields above. Time records carry `batch1`, `paired` (null for the baseline itself, which has
`is_reference: true`) and `batch8`. `compare` uses the latest `eval` and `time` records whose `candidate_sha`
matches the directory's current content, so rebuilding a candidate in place forces a fresh eval and time.

A quick look at everything recorded so far:

```bash
jq -c 'select(.kind=="compare") | {b, d_cv_acc, mcnemar, ratio: .latency.ratio, verdict}' \
    /home/vaper/Work/Prototypes/626-artifacts/harness/results.jsonl
```

## Baseline and noise

Measured on legion on 2026-10-31: RTX 3060 Laptop GPU, driver 615.78.08 (CUDA 13.4), onnxruntime-gpu 1.31.1
(CUDA 13.0 build), cuDNN 9.27, TensorRT 10.16.1. Raw records are in `results.jsonl`; the development runs that
shaped this protocol are in `artifacts/harness/dev/`.

**The eval recipe reproduces the recorded numbers.** `check_recipe.py` runs eval's head code on the dev
features recorded on carlpc2 (torch, fp16 autocast):

- benchmark.csv's probe CV 0.8252 and its per-fold accuracies are reproduced exactly with the C grid of the
  time, (0.01, 0.1, 1, 10). Commit cbf1877 widened the grid to (0.001 ... 1), which the deployed head and this
  harness use. With it, the probe's pooled out-of-fold accuracy is 0.8248, exactly as in fit_report.json.
- The fused out-of-fold accuracy is 0.8268 against the recorded 0.8264 (one image). carlpc2 reproduces 0.8264
  exactly with the same code and features. On legion, fold 3's inner-CV choice between C = 0.01 and C = 0.1
  (macro-F1 margin about 2e-4) resolves the other way.

**Baseline** (`candidates/baseline_fp32`: the deployed graph's trunk, CUDA EP defaults):

| | value |
|---|---|
| CV accuracy, mean ± sd over 5 folds | 0.8268 ± 0.0151 (folds 0.8008 0.8270 0.8330 0.8350 0.8383) |
| CV macro-F1 | 0.8254 ± 0.0149 |
| pooled out-of-fold accuracy (recorded deployed fit: 0.8264) | 0.8268 |
| batch-1 latency, median (p10 to p90), standalone runs | 116.4 ms (116.1 to 123.9) and 119.0 ms (118.7 to 119.1) in two runs |
| batch-8 throughput | 8.6 to 9.6 img/s |

The trunk cut is exact. On CPU in strict fp32, deployed graph vs candidate + `head.npz` in numpy: probabilities
agree to 1.2e-7, and batch-1 vs batch-4 embeddings are identical. On CUDA EP they agree to 7.8e-4 (TF32), with
argmax 16/16. Two evals of the baseline produce identical embeddings and predictions.

**Accuracy noise.** Numerically equivalent versions of the same model score CV 0.8244 to 0.8288: strict fp32
0.8244, the recorded torch features 0.8268, TF32 0.8268, TensorRT fp16 0.8280 (0.8276 from another engine
build), fp16 0.8288. Each differs from the baseline on 12 to 22 images, with McNemar p ≥ 0.15. The cause is the
recipe: tiny embedding differences flip the inner-CV choice of C in a fold, which moves that fold by up to
0.6 pp. A candidate within about 0.25 pp of the baseline is therefore indistinguishable from it. The 0.5 pp
tolerance sits about twice that band. Real losses are caught: dropping the last block costs 1.20 pp (82 vs 52
discordant, p = 0.012) and dropping four costs 7.56 pp. Both are REJECT.

**Latency noise and N.** The laptop GPU runs these loads at its limits: 81 to 88 °C, power 69 to 80 W, SM clock
stepping between about 900 and 1530 MHz every few seconds. That sets the noise, not the sample count:

- Within a run, p10 to p90 is about ±0.3% of the median, so the median of 300 runs has a standard error near
  0.03%. More runs per side would not help.
- Absolute baseline medians drifted from 109.0 to 120.9 ms (11%) over one afternoon. A blocked A-then-B protocol
  gave TensorRT round ratios spread 13% inside one run, because blocks straddled clock steps.
- Alternating single runs (the frozen protocol) gives each pair one clock state. An A/A check (the baseline
  against a copy with a no-op option) measured 1.0003 and 0.9999. Per-round ratios agree within 0.3% inside a
  run. One candidate's paired ratio moved at most 0.2% between sessions for fp16 and truncated graphs, 0.9% for
  TensorRT (2.3% counting a session at a cooler GPU state), and 1.2% for strict fp32.
- Hence N = 5 rounds x 60 pairs (300 candidate + 300 baseline runs). Pairing per run handles the clock steps.
  300 pairs keep the median ratio insensitive to the few pairs that straddle a step, and 5 rounds give the
  in-run spread check. `compare` requires a ratio at least 3% from 1 before it calls a difference beyond noise.

**Variants measured** (CV accuracy ± sd; paired batch-1 ratio to the baseline over two sessions; verdict):

| candidate | change | CV acc | macro-F1 | batch-1 (ratio) | batch-8 | verdict |
|---|---|---|---|---|---|---|
| baseline_fp16 | fp16 weights and math, CUDA EP | 0.8288 ± 0.0170 | 0.8271 | 65.0 ms (0.573 to 0.574) | 16.5 to 18.2 img/s | ACCEPTABLE |
| baseline_trt_fp16 | TensorRT EP, fp16, profile 1..8 | 0.8280 ± 0.0134 | 0.8264 | 27.0 ms (0.228 to 0.230) | invalid (see gotchas) | ACCEPTABLE |
| baseline_fp32_notf32 | CUDA EP `use_tf32=0` | 0.8244 ± 0.0155 | 0.8233 | 185 ms (1.466 to 1.483) | 5.1 to 5.3 img/s | ACCEPTABLE (slower) |
| degraded_drop1block | last block removed | 0.8149 ± 0.0165 | 0.8130 | 114 ms (0.963) | 9.2 to 9.4 img/s | REJECT |
| degraded_drop4blocks | last 4 blocks removed | 0.7512 ± 0.0180 | 0.7482 | 102 ms (0.853) | 10.2 to 10.4 img/s | REJECT |

The truncated graphs confirm the latency is the work: removing 1 and 4 of 27 blocks gives ratios 0.963 and
0.853, against 26/27 = 0.963 and 23/27 = 0.852.

## Gotchas for candidate authors

- **IR version.** onnx 1.23 writes IR version 14; onnxruntime 1.31 reads at most 13. Set
  `model.ir_version = 10` (or load and edit an exported graph) before saving a hand-built graph.
- **Dynamic batch from torch.export.** The deployed graph is static at batch 1 because `torch.export`
  specialises any example dimension of size 1. Export with an example batch of 2 and a `Dim("batch")`.
  `export-baseline` instead rewrote the deployed graph's Reshape targets (`tools/speed/baseline.py`).
- **External data.** Keep `model.onnx.data` (any `location`) in the candidate directory as a regular file.
  onnxruntime 1.31 rejects external data that resolves outside the model directory, symlinks included.
- **Input name.** The deployed graph's input is called `input`; candidates must call it `pixel_values`, and
  the output `embedding` (the pooled 1152-d feature, before any head).
- **TF32.** CUDA EP defaults to `use_tf32=1`. A candidate may change any provider option, and the change is part
  of the candidate (it is hashed and recorded).
- **TensorRT.** The first `eval` builds the engine inside the GPU lock (about 50 s for So400m fp16), cached
  under `artifacts/harness/work/<id>/<sha12>/trt_cache` and reused by `time`. A new candidate hash means a new
  build, and builds differ numerically (CV 0.8276 vs 0.8280 for two builds of the same graph). Put int8
  calibration or QDQ in the graph; the harness only passes provider options.
- **TensorRT is wrong at batch > 1 on the baseline graph.** With a 1..8 profile, batch-1 output is correct
  (embedding cosine to CUDA fp32: median 0.99988, min 0.995). At batches 2, 4 and 8, every row is wrong (cosine
  0.42 to 0.92, even for 8 copies of one image), in fp32 and fp16 alike, while CUDA EP is exact on the same
  graph. `time` catches this and marks `batch8.valid = false`. If your graph must serve batches under TensorRT,
  check that flag. For batch-1 deployment, a profile of min = opt = max = 1 is legitimate.
- **VRAM.** `time` holds your session and the baseline's (about 2 GB at batch 1) together, next to about 1.1 GB
  for the resident `nemo-speech` process, on a 6 GB GPU. The baseline session is freed before batch 8.
- **Runtime.** eval takes about 5 min for the fp32 baseline (1 min for TensorRT fp16), plus head fitting on the
  CPU. time takes 2 to 4 min. Both hold the GPU lock for their GPU part.
- **Zero-shot fusion** uses the deployed text embeddings (`exports/final/head.npz["W"]`, 20 x 1152), T = 50,
  w = 0.25. It is only meaningful for So400m-space embeddings; the harness refuses it for other dimensions but
  cannot check the space itself, so set it truthfully.
- **The `probs` output of `with_head.onnx` uses seeded random head weights** of the deployed shapes. Timing does
  not depend on them; never read accuracy from it.
- **CPU preprocessing is not timed.** Pillow decode + resize + crop + normalise of a dev image takes about
  5 ms single-threaded (median of 64, measured at load average 19), next to about 110 ms for the baseline
  model. Once a model gets near 20 ms, preprocessing is a fifth of what the app's user waits for.
- **Absolute latency depends on the GPU's thermal state.** Back-to-back runs keep it at 81 to 88 °C, where it
  clocks lower than a cool GPU serving one click at a time. Quote speedups as the paired ratio; quote absolute
  ms with the `gpu_during` clocks next to them. Power-dense kernels throttle harder: strict fp32 drove the SM clock
  to 900 MHz in one run.
- **legion is shared.** Other agents compile Rust and run their own jobs; `time` records the load average, GPU
  processes and clocks with every run.
