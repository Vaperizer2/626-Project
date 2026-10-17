# STATUS — resume checkpoint (updated 2026-10-17 ~22:00 SAST, laptop going off overnight)

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
- [ ] Phase 2 ablations; Phase 3 final model + ONNX export; Phase 4 OpenCV demo + report.

## Running PC jobs (Claude Science job ids; they run detached, serialised by flock ~/.gpu.lock)
| job | id | what | outputs on PC |
|---|---|---|---|
| 2A | 36b782da-2c70-4ef0-ab86-a6d2a45eefc0 | siglip_so400m_384_partial, siglip2_b16, clip_vitb16 | `~/Projects/626-Project/results/phase2/runs_A.jsonl` |
| 2B | f527a56f-0fe9-401c-973f-0aaabe4248e7 | dinov2_vitb14 vit_b16 convnext_tiny swin_t efficientnet_v2_s resnet50 mobilenetv3_large scratch_cnn | `.../results/phase2/runs_B.jsonl` |
| C (overnight) | b32ad78a-0ef8-44bb-b9e1-f4acb56b5c1a | So400m + So400m2 probe with hflip TTA; So400m partial FT v2 (4 blocks, lr 2e-5, fixed schedule) folds 0-2; folds 3-4 for siglip2_b16 clip_vitb16 convnext_tiny vit_b16 dinov2_vitb14 | `.../results/phase1b/runs.jsonl`, `.../results/phase2/runs_C.jsonl`, `runs_C3.jsonl` |
Note: 2A/2B ran with the OLD LR schedule (warm-up not restarted at unfreeze); C uses the fixed one (commit d6d4e09+).
If job order on the lock differs (C before B), that's fine — each job extracts its own code tarball.
Per-fold predictions: `~/Projects/626-Project/runs/<name>-<hash>/fold<k>/val_preds.npz` (best.pt checkpoints also there).
Collect: `host.compute.create('ssh:carlpc2-science').attach_job('<id>').result()` (repl), or
`c.download('/home/carl/Projects/626-Project/results/phase2/runs_A.jsonl')`. Job scratch dirs:
`~/.claude-science-scratch/.claude-science/jobs/<id>/` (nohup.out = log).

## Findings since Phase 1
- Probe + zero-shot logit fusion (same SigLIP So400m tower): logp_probe + 0.25*log_softmax(50*cos_sim) gives
  OOF 83.4% acc / 83.2% F1 (vs 82.5/82.4), in-scope 86.7%; 4/5 folds improve. Weight picked on OOF -> fix
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
