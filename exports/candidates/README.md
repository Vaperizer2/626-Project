# Efficiency-study candidates (2026-11-05)

One ONNX graph per candidate, image tower plus the deployed recipe's two heads (probe refitted on all development
rows, zero-shot head, fused with T 50 / w 0.25; uncalibrated), built by `tools/views/export_candidate.py`. The
`.onnx` files are git-ignored (0.24 to 1.2 GB); `export_report.json`, `labels.json` and `preprocess.json` are
tracked. Rebuild with

    uv run python tools/views/export_candidate.py --model mobileclip2_s3 --out exports/candidates/mobileclip2_s3
    uv run python tools/views/export_candidate.py --model pe_core_l14_336 --out exports/candidates/pe_core_l14_336
    uv run python tools/views/export_candidate.py --model siglip_so400m_384 --out exports/candidates/siglip_so400m_384

These are timing and size artefacts for `results/efficiency/step4_latency.md`, not releases: `compclass.final`
owns calibration, the abstain threshold and the release manifests.
