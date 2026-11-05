"""Efficiency study, step 4: batch-1 latency of the candidate image towers on this machine's GPU, one harness.

    uv run python tools/views/latency.py --model siglip_so400m_384 --model siglip_so400m_384@308 \
        --model pe_core_l14_336 --model mobileclip2_s3 --model convnext_tiny

Every tower runs under the same conditions: PyTorch eager, CUDA, fp16 autocast, channels-first, a static
[1, 3, S, S] input already on the GPU, timed with CUDA events. ``key@px`` overrides the input size (the Fast model
runs SigLIP at 308 px). Rounds interleave the models (5 rounds x 60 runs each after warm-up), so clock drift on a
laptop GPU hits every model alike; the medians and p90 over all 300 runs are reported, plus batch-8 throughput,
peak inference VRAM, parameters and GFLOPs (torch FlopCounterMode). Weights are random (pretrained=False): latency
does not depend on their values. Models are put in their inference form first (timm reparameterize_model: FastViT's
training-time branches fold into single convolutions, identical outputs). The deployed and Fast models add a 20-way linear head to the SigLIP tower, which
is below the timer's resolution; Fast additionally runs INT8 under TensorRT, which this harness cannot reproduce,
so its row here is the fp16 upper bound of that trunk.
"""
import argparse, json, statistics, sys, time
import numpy as np
import torch

sys.path.insert(0, "src")
from compclass.models import build_model
from timm.utils.model import reparameterize_model

ROUNDS, RUNS_B1, RUNS_B8, WARMUP, WARMUP_S = 5, 60, 10, 30, 5.0


def gflops(m, px):
    from torch.utils.flop_counter import FlopCounterMode

    with FlopCounterMode(display=False) as fc:  # grad on: the module tracker needs grad_fns under attention pooling
        m(torch.zeros(1, 3, px, px))
    return fc.get_total_flops() / 1e9


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="registry key, optionally key@px")
    ap.add_argument("--json", help="write the measurements here as well")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    models, meta = {}, {}
    for spec_s in a.model:
        key, _, px = spec_s.partition("@")
        kw = {"input_size": int(px)} if px else {}
        m, spec = build_model(key, 0, pretrained=False, **kw)
        m.eval()
        n_mod = sum(1 for _ in m.modules())
        m = reparameterize_model(m)  # inference form: FastViT's multi-branch blocks fold into single convs (no-op for ViTs)
        m.eval()
        if sum(1 for _ in m.modules()) != n_mod:
            print(f"{spec_s}: reparameterised, {n_mod} -> {sum(1 for _ in m.modules())} modules", flush=True)
        px = spec.input_size
        n = sum(p.numel() for p in m.parameters())
        gf = gflops(m, px)
        models[spec_s] = m.to(dev).half()
        meta[spec_s] = {"key": key, "px": px, "params_m": n / 1e6, "gflops": gf}
        print(f"{spec_s}: {px} px, {n/1e6:.0f} M params, {gf:.0f} GFLOPs", flush=True)

    def run(m, x):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            return m(x)

    def timed(m, x, runs):
        out = []
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(runs):
            s.record(); run(m, x); e.record(); e.synchronize()
            out.append(s.elapsed_time(e))
        return out

    x1 = {k: torch.randn(1, 3, v["px"], v["px"], device=dev, dtype=torch.half) for k, v in meta.items()}
    x8 = {k: torch.randn(8, 3, v["px"], v["px"], device=dev, dtype=torch.half) for k, v in meta.items()}
    for k, m in models.items():  # warm-up: cuDNN autotune, allocator, clocks
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        t0 = time.perf_counter(); n = 0
        while n < WARMUP or time.perf_counter() - t0 < WARMUP_S:
            run(m, x1[k]); n += 1
        torch.cuda.synchronize()
        meta[k]["peak_vram_mb"] = (torch.cuda.max_memory_allocated() - base) / 2**20 + sum(p.numel() * 2 for p in m.parameters()) / 2**20
        run(m, x8[k]); torch.cuda.synchronize()
    b1 = {k: [] for k in models}; b8 = {k: [] for k in models}
    for r in range(ROUNDS):
        for k, m in models.items():
            b1[k] += timed(m, x1[k], RUNS_B1)
            b8[k] += timed(m, x8[k], RUNS_B8)
        print(f"round {r+1}/{ROUNDS} " + "  ".join(f"{k} {statistics.median(b1[k]):.2f} ms" for k in models), flush=True)
    rows = []
    for k in models:
        v = meta[k]
        v.update(b1_median_ms=float(np.median(b1[k])), b1_p90_ms=float(np.percentile(b1[k], 90)),
                 b1_min_ms=float(min(b1[k])), b8_img_per_s=8 / (np.median(b8[k]) / 1e3))
        rows.append(v)
    ref = rows[0]
    print(f"\n{torch.cuda.get_device_name(0)}, torch {torch.__version__}, fp16 autocast, batch 1 static input on GPU, "
          f"{ROUNDS}x{RUNS_B1} runs interleaved\n")
    print("| Tower | px | params | GFLOPs | batch-1 median ms | p90 ms | vs first row | batch-8 img/s | peak VRAM MB |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v in rows:
        print(f"| {v['key']}@{v['px']} | {v['px']} | {v['params_m']:.0f} M | {v['gflops']:.0f} | {v['b1_median_ms']:.2f} | "
              f"{v['b1_p90_ms']:.2f} | {v['b1_median_ms']/ref['b1_median_ms']:.2f}x | {v['b8_img_per_s']:.0f} | {v['peak_vram_mb']:.0f} |")
    if a.json:
        json.dump({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "rows": rows}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
