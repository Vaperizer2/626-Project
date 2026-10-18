# STATUS — resume checkpoint (updated 2026-10-18 ~16:15 SAST) — ALL PLANNED PHASES COMPLETE

If an assistant session dies, a new one should read this file first, then `CODEBASE.md` and `EXPERIMENT_LOG.md`.

## Goal
Comparative report on model families for 20-class electronic-component classification (Kaggle
"electronic-component-recognizer"), and one exportable model at ~80–85% for an OpenCV demo on internet photos.
Repo doubles as a learning resource (`docs/LEARNING_PATH.md`). uv for all envs. Branch `refine-01`.

## Evaluation contract (do not break)
- Data: `data/v2/manifest.json` (fingerprint `431315e87d2612cb`): 2930 rows, 2787 duplicate clusters, sealed
  `test` = 418 rows, `dev` = 2512 rows in 5 cluster-grouped folds. Built by `tools/build_dataset_v2.py`.
- Model selection ONLY on dev grouped CV. Sealed test evaluated exactly once at the end (`--sealed-test`).
- Second sealed set: `data/web_holdout_v1/` (Wikimedia Commons photos; being built) — final eval/demo only.
- Primary metric: macro-F1; also report accuracy and "in-scope" scores (rows without the `ood_suspect` flag).

## Plan progress
- [x] Phase 0 — data v2, `src/compclass` package, PC env (`~/Projects/626-Project` on `ssh:carlpc2-science`).
- [x] Phase 1 — probe/zero-shot sweep (`results/phase1/`). Best: SigLIP So400m/384 linear probe
      82.5% acc / 82.4% macro-F1 (86.1% in-scope). Zero-shot So400m 77.5%. HOG+SVM 22%.
      Label audit: 69 confident issues (`results/phase1/label_issues_v2.*`); dropping OOD rows from training
      HURT (in-scope 86.1 -> 85.2), dropping issues +0.2 (noise) -> keep v2.0 training data unchanged.
- [~] Phase 2 — fine-tune benchmark, folds 0-2, running on the PC (jobs below).
      21:11 SAST: So400m partial FT fold0 = 0.7948 acc / 0.7936 F1 (probe fold0 = 0.8048 / 0.8041); ~25 min/fold,
      ~1.7 min/epoch, 137M trainable params, 4.3 GB peak VRAM. ETA: 2A ~22:50, 2B ~01:00-01:30 SAST.
- [x] Phase 2 docs — `docs/` (paper citations still to verify).
- [x] Phase 2 — results/benchmark.csv (fixed schedule, 9 FT families x 5 folds + all probes/zero-shot). Schedule fix: no
      consistent effect (paired folds 0-2, 14/30 improved). Leakage simulation: legacy-style eval inflates +5.5 (SigLIP) to
      +13.6 pts (ResNet50). Best FT: SigLIP2-B 73.3% F1 < its probe 75.8%.
- [x] Phase 3 — final model = SigLIP So400m frozen + probe fused with zero-shot head (src/compclass/final.py; exports/final/).
      SEALED TEST USED (once, 2026-10-18): 83.7% acc / 83.4% macro-F1; in-scope 86.9%; 87.1% coverage @ 90.9% accepted acc
      (abstain < 0.5); ECE 0.034. Probe-only 84.4% (fusion gain ~0 with deployable preprocessing). ONNX on PC only
      (1.64 GB, batch 1): ORT parity 6e-6, 892 ms/img CPU; OpenCV DNN segfaults on this graph.
- [x] Phase 4 Report — docs/report/REPORT.md (+ figures/tables via scripts/make_report_figures.py; analysis CSVs via
      scripts/analysis_tables.py, byte-identical reproduction). README + EXPERIMENT_LOG rewritten. 54 arXiv refs checked.
- [x] Phase 4 Demo — `uv run python -m compclass.demo --source <file|folder|URL|webcam>` (docs/DEMO.md). Parity 64/64,
      0.9 s/img CPU, 67 ms/img onnxruntime-gpu (PC). Webcam/window modes untested (no camera/display available).
- [x] Web holdout — data/web_holdout_v1 (407 Wikimedia photos, images gitignored; rebuild with tools/build_web_holdout.py
      from the cached tar artifact) evaluated ONCE: 0.821 acc (95% CI 0.780-0.855) / 0.816 macro-F1; probe-only 0.779 ->
      zero-shot head +4.2 pts out-of-source. Weak: led (->diode), switch (->ic chip), transformer (->inductor coil).
- Corrections recorded by the report agent: probe>=fine-tune holds for SIGLIP only (CLIP-B/DINOv2-B/ConvNeXt-T/ResNet-50
  fine-tunes beat their probes 5/5 folds); schedule ablation confounded with BN freezing; hflip TTA mixed.

## Running PC jobs (Claude Science job ids; they run detached, serialised by flock ~/.gpu.lock)
| job | id | what | outputs on PC |
|---|---|---|---|
| 2A | 36b782da-2c70-4ef0-ab86-a6d2a45eefc0 | siglip_so400m_384_partial, siglip2_b16, clip_vitb16 | `~/Projects/626-Project/results/phase2/runs_A.jsonl` |
| 2B | f527a56f-0fe9-401c-973f-0aaabe4248e7 | dinov2_vitb14 vit_b16 convnext_tiny swin_t efficientnet_v2_s resnet50 mobilenetv3_large scratch_cnn | `.../results/phase2/runs_B.jsonl` |
| ~~C~~ | b32ad78a-… | CANCELLED (would have mixed old/new schedule across folds) | — |
| C2 (overnight, MAIN benchmark) | c00df89a-332a-4b75-bfc6-976b25aae7e1 | FIXED schedule, 5 folds: siglip2_b16 clip_vitb16 vit_b16 dinov2_vitb14 convnext_tiny swin_t efficientnet_v2_s resnet50 mobilenetv3_large; So400m/So400m2 TTA probes; So400m partial v2 folds 0-2; scratch_cnn 5 folds | `~/Projects/626-Project/results/phase2/runs_fixed.jsonl`, `runs_v2/*/fold*/val_preds.npz`, `results/phase1b/runs.jsonl` |
Note: 2A/2B ran with the OLD LR schedule (warm-up not restarted at unfreeze) -> report them as the schedule-bug
ablation; C2 (fixed schedule, all 5 folds) is the main fine-tune table.
2A result (folds 0-2, old schedule; results/phase2/raw_oldschedule/): So400m partial FT 0.8216 acc/0.8206 F1
(probe 0.8229/0.8216, 23 min/fold); SigLIP2-B full FT 0.7201/0.7199 (< its probe 0.7533/0.7518!); CLIP-B FT
0.6956/0.6908 (> probe 0.6665/0.6598).
If job order on the lock differs (C before B), that's fine — each job extracts its own code tarball.
Per-fold predictions: `~/Projects/626-Project/runs/<name>-<hash>/fold<k>/val_preds.npz` (best.pt checkpoints also there).
Collect: `host.compute.create('ssh:carlpc2-science').attach_job('<id>').result()` (repl), or
`c.download('/home/carl/Projects/626-Project/results/phase2/runs_A.jsonl')`. Job scratch dirs:
`~/.claude-science-scratch/.claude-science/jobs/<id>/` (nohup.out = log).

## Findings since Phase 1
- Probe + zero-shot logit fusion (same SigLIP So400m tower): logp_probe + 0.25*log_softmax(50*cos_sim) gives
  OOF 83.4% acc / 83.2% F1 (vs 82.5/82.4), in-scope 86.7%; all 5/5 folds improve (fold acc 0.805/0.829/0.835/0.833/0.824 -> 0.807/0.841/0.837/0.845/0.838). Weight picked on OOF -> fix
  T=50, w=0.25 a priori and confirm on sealed test. Final export = image tower + 2 linear heads in one ONNX.
- Web-holdout sub-agent (Wikimedia, data/web_holdout_v1) was still running when the laptop went off; check
  whether data/web_holdout_v1/manifest.json exists, else re-dispatch (task text is in the session archive).

## Known issues to fix before the final model
- FIXED d6d4e09: warm-up restart at unfreeze; BN frozen in head-only phase; img_size only for transformers.
- Open (minor): grad-accum last partial group under-weighted; docs paper citations not yet verified online.
- PC login shell was switched fish -> bash for job submission; switch back at the end: `chsh -s /bin/fish carl`.

## Next steps
1. Collect 2A/2B -> `results/phase2/`, build `results/benchmark.csv` (probe + fine-tune, same folds 0-2).
2. Ablations on top-2 families (probe vs partial vs full FT, resolution, legacy split vs v2 = leakage effect).
3. Final model: likely SigLIP So400m (probe or partial FT); calibrate + abstain threshold; consider distilling to
   a small CNN for real-time OpenCV; export ONNX + parity check; evaluate sealed test + web holdout once.
4. OpenCV demo app (file/folder/URL/webcam) and comparative report.

## Second plan (2026-10-18 afternoon) — complete
- IEEE paper: docs/paper/main.tex -> main.pdf (8 pp, 47 refs from arXiv/CrossRef; fill \placeholder affiliation/email/ack/URL).
- Interactive explorer: docs/explorer/index.html (offline, 10 tabs, 11 family pages); rebuild with scripts/build_explorer.py.
- Original Kaggle evaluation: results/kaggle_original/ (never-seen 83.0 %; legacy-protocol replay 87.7 % vs old 71.3 %).
- GPUI app: apps/gpui-demo (gpui =0.2.2, ort =2.0.0-rc.14 load-dynamic; build with `cargo run --release --locked`;
  keep Cargo.lock: libc pinned to 0.2.189 because 0.2.191 breaks xattr). Window not yet launched by anyone — first
  run is the user's; headless parity 64/64. Model weights exports/final/model.onnx.data now on the laptop too.

## Open items (user's call)
- Switch the PC login shell back: `chsh -s /bin/fish carl` (temporarily bash for job submission).
- Run the GPUI app once on the laptop desktop and report any window/portal/clipboard issues.
- Optional next work: relabel the 69 audited issues + rebuild "switch" (Nintendo contamination); distil the model into
  a small CNN for pure OpenCV-DNN/real-time CPU use; collect target-domain camera photos; test webcam/window demo
  modes and onnxruntime-gpu on the RTX 3060 laptop (untested).
