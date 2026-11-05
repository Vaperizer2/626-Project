References (5-fold CV macro-F1): MobileCLIP2-S3 probe 0.7920, MobileCLIP2-S3 fused 0.8013, SigLIP So400m fused 0.8403

| Run | trainable | epochs (best) | min/fold | peak VRAM | CV macro-F1 (± SD) | CV acc | vs S3 probe | vs S3 fused | vs SigLIP fused | folds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mobileclip2_s3_full | 122.9 M | 13.4 (4.4) | 9.6 | 2895 MB | 0.7923 ± 0.0202 | 0.7934 | +0.03 ± 0.79 (3/5) | -0.91 ± 0.97 (2/5) | -4.80 ± 1.00 (0/5) | [0.774, 0.769, 0.807, 0.797, 0.815] |
| mobileclip2_s3_partial | 60.6 M | 9.6 (2.6) | 2.4 | 1496 MB | 0.7890 ± 0.0227 | 0.7918 | -0.30 ± 0.72 (2/5) | -1.24 ± 0.65 (1/5) | -5.13 ± 0.67 (0/5) | [0.756, 0.786, 0.785, 0.801, 0.817] |
