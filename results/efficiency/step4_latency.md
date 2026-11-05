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
