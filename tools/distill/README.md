# Rerun the distillation track

These scripts distil the deployed SigLIP So400m-384 trunk into smaller students and export them as speed-harness
candidates. Results and the reasoning behind them are in `EXPERIMENT_LOG.md` (2026-10-31).
The 2026-10-31 run trained on carlpc2 (RTX 4060 Ti 8 GB), from a copy of this checkout at
`/home/carl/626-runs/distill/repo` with `uv sync --locked` and `imagehash` installed. carlpc2 is gone, and the pool
and the teacher cache were lost with it (README, "Where training runs"). A rerun now starts at step 1 on the laptop
(RTX 3060 Laptop, 6 GB), the only GPU. Change the `/home/carl/...` paths in `run_carlpc2.sh` and
`run_fold_diag.sh` first. The B/16 student used 5.7 GB at batch 48 on carlpc2, so expect to lower `--batch` on 6 GB.
On the laptop, GPU steps take `/tmp/626-gpu-bench.lock` instead of `/tmp/626-gpu.lock`. Heavy CPU-only work can run
on a Boat.dev CPU sandbox (`boat` CLI). The commands below are the ones the 2026-10-31 run used.

1. Build the disjoint pool. Every subdirectory of `--component` and `--generic` is one source.

   ```bash
   .venv/bin/python -I tools/distill/pool.py --component RAW --generic RAW_GENERIC --out POOL \
     --v2 data/v2 --v2-clip DIR_WITH_clip_b32_embeddings --holdout data/web_holdout_v1 \
     --cap yemenoglu_circuit-components=500
   ```

   Look at `POOL/review/nearest_kept.jpg` and `POOL/review/dropped_boundary.jpg` before you go on. Both should
   show different photos on each side of every pair.

2. Cache the teacher's embeddings of every view, under the GPU lock (about 100 min for 270k views):

   ```bash
   flock /tmp/626-gpu.lock .venv/bin/python tools/distill/teacher.py --pool POOL --out TEACHER \
     --views-component 8 --views-generic 2
   ```

3. Train the students and screen them on the development folds. Edit the paths at the top of the script first.

   ```bash
   flock /tmp/626-gpu.lock tools/distill/run_carlpc2.sh siglip2_b16_256
   ```

   The screen prints the teacher's out-of-fold accuracy (0.8264) next to each student's. If the teacher line
   differs, the screen no longer matches the deployed recipe.

4. Export a candidate and run the speed harness on it on legion:

   ```bash
   .venv/bin/python tools/distill/export.py export --run RUN --dest CANDIDATE --pool POOL \
     --labels exports/final/labels.json --commit "$(git rev-parse --short HEAD)"
   ```

   Then run `eval`, `time` and `compare` as [`tools/speed/README.md`](../speed/README.md) describes. To build a
   TensorRT fp16 variant, run that harness's `variants.py runtime`.

To test whether unlabeled development images close the gap for one fold, run `tools/distill/run_fold_diag.sh K`.
Only fold K's accuracy from its fine-tuned run is clean.
