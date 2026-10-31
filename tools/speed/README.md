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

- A second session holds the baseline. Batch-1 rounds alternate: 60 runs of yours, 20 of the baseline, five
  times. The per-round ratio (your round median / the baseline's) is the speed number (`paired.ratio_median`),
  because legion's GPU sits at its 80 W cap and that cap moves with CPU load, shifting absolute latency by 3-5%
  between minutes (see Baseline and noise). Both sessions see the same clocks inside a round.
- Warmup per session and batch size: at least 20 runs and 10 s.
- Batch 1: reports median, p10, p90, min, max and the 5 round medians of your 300 runs.
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
- `beyond noise`: true when B's five round ratios all lie outside A's range (or, unpaired, the round medians do
  not overlap);
- batch-8 throughputs, where valid.

## Results

Every `eval`, `time` and `compare` appends one JSON line to
`/home/vaper/Work/Prototypes/626-artifacts/harness/results.jsonl` with `kind`, `candidate` (directory name),
`candidate_sha` (hash of `model.onnx`, its external data, `preprocess.json`, `runtime.json`), `git_sha` and
`git_dirty` of this harness, `n_images`, `n_errors`, `n_timed_runs`, plus the fields above. `compare` uses the
latest `eval` and `time` records whose `candidate_sha` matches the directory's current content, so rebuilding
a candidate in place forces a fresh eval and time.

## Baseline and noise

BASELINE_AND_NOISE

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
- **TensorRT.** The first `eval` builds the engine inside the GPU lock (minutes for So400m), cached under
  `artifacts/harness/work/<id>/<sha12>/trt_cache` and reused by `time`. A new candidate hash means a new
  build. Put int8 calibration or QDQ in the graph; the harness only passes provider options.
- **Zero-shot fusion** uses the deployed text embeddings (`exports/final/head.npz["W"]`, 20 x 1152), T = 50,
  w = 0.25. It is only meaningful for So400m-space embeddings; the harness refuses it for other dimensions but
  cannot check the space itself, so set it truthfully.
- **The `probs` output of `with_head.onnx` uses seeded random head weights** of the deployed shapes. Timing does
  not depend on them; never read accuracy from it.
- **CPU preprocessing is not timed.** Pillow decode + resize + crop + normalise of a dev image takes about
  5 ms single-threaded (median of 64, measured at load average 19), next to about 110 ms for the baseline
  model. Once a model gets near 20 ms, preprocessing is a fifth of what the app's user waits for.
- **legion is shared.** Other agents compile Rust and run their own jobs; `time` records the load average, GPU
  processes and clocks with every run. A `nemo-speech` process keeps about 1.1 GB of VRAM; budget about
  4.9 GB for your session.
