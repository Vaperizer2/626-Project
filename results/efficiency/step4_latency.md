# Step 4: measured batch-1 latency on the RTX 4070 Laptop GPU (2026-11-05)

## PyTorch eager, fp16 autocast, inference form (tools/views/latency.py)

| Tower | px | params | GFLOPs | batch-1 median ms | p90 ms | vs first row | batch-8 img/s | peak VRAM MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| siglip_so400m_384@384 | 384 | 428 M | 604 | 30.35 | 31.17 | 1.00x | 35 | 848 |
| siglip_so400m_384@308 | 308 | 428 M | 401 | 18.23 | 19.03 | 0.60x | 58 | 830 |
| pe_core_l14_336@336 | 336 | 316 M | 352 | 18.56 | 19.37 | 0.61x | 63 | 620 |
| mobileclip2_s3@256 | 256 | 123 M | 30 | 10.62 | 11.86 | 0.35x | 376 | 289 |
| convnext_tiny@224 | 224 | 28 M | 9 | 3.91 | 4.44 | 0.13x | 1393 | 66 |

## onnxruntime 1.31.1 CUDA EP, exported fp16 graphs, IO binding (tools/views/ort_latency.py)

| Graph | file MB | median ms | p90 ms | min ms |
|---|---:|---:|---:|---:|
| siglip_so400m_384/model_fp16.onnx | 821 | 32.66 | 33.55 | 29.38 |
| pe_core_l14_336/model_fp16.onnx | 607 | 26.08 | 26.99 | 24.97 |
| mobileclip2_s3/model_fp16.onnx | 240 | 8.96 | 9.65 | 7.95 |

## Follow-up: SigLIP at 308 px, the Fast trunk's resolution, fp16 (not INT8)

Feature cache `crop100@308` (tools/views/extract_views.py `siglip_so400m_384@308 crop100`), fused OOF via
`fusion_table.py --siglip-view crop100@308`; graph `exports/candidates/siglip_so400m_308` (`--px 308`).

| Row | probe F1 | zero-shot F1 | fused F1 | fused acc | vs SigLIP@384 fused |
|---|---:|---:|---:|---:|---:|
| SigLIP So400m @384, crop 1.0 | 0.8330 | 0.7643 | 0.8403 | 0.8420 | reference |
| SigLIP So400m @308, crop 1.0 | 0.8228 | 0.7569 | 0.8222 | 0.8244 | −1.81 ± 0.21 pt (0/5) |

One ORT CUDA EP session, fp16 graphs, 5 × 60 interleaved: SigLIP@384 31.94 ms, SigLIP@308 20.35 ms,
MobileCLIP2-S3 8.88 ms (`latency_ort_cuda_fp16_with308.json`).
