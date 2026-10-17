"""GPU witness: prove every registry model downloads, builds and trains on this machine.

    uv run python scripts/gpu_witness.py --out results/gpu_witness.json

For each model: download pretrained weights (into $HF_HOME), then time fp16-AMP training steps
(forward + backward + AdamW step) at batch 16 and the model's native input size, recording parameters,
peak VRAM and steps/s; plus fp16 inference throughput. On OOM it retries with gradient checkpointing.
Then it exercises the other entry points end-to-end on a synthetic manifest: 1-epoch fine-tunes through
the real ``compclass.train`` CLI, ONNX export + parity, zero-shot (also caches the text towers), the linear
probe and the classical baseline. Everything lands in one JSON file.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

from compclass.data import make_synthetic_manifest
from compclass.models import REGISTRY, build_model


def bench(model: torch.nn.Module, size: int, batch: int = 16, warm: int = 3, iters: int = 10) -> dict:
    model.cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(batch, 3, size, size, device="cuda")
    y = torch.randint(0, 20, (batch,), device="cuda")
    for i in range(warm + iters):
        if i == warm:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    it_s = iters / (time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 2**20
    del opt
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(3):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
    return {"train_it_s": round(it_s, 2), "train_img_s": round(it_s * batch, 1), "peak_vram_mb": round(peak),
            "infer_img_s": round(iters * batch / (time.perf_counter() - t0), 1)}


def witness_models(batch: int) -> list[dict]:
    out = []
    for key, entry in REGISTRY.items():
        rec: dict = {"key": key, "family": entry.family, "timm_name": entry.timm_name, "batch": batch}
        try:
            t0 = time.perf_counter()
            model, spec = build_model(key, 20, pretrained=entry.timm_name is not None)
            rec.update(build_s=round(time.perf_counter() - t0, 1), input_size=spec.input_size,
                       params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 2))
            try:
                rec.update(bench(model, spec.input_size, batch))
                rec["grad_checkpointing"] = False
            except torch.cuda.OutOfMemoryError:
                import gc

                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                model.set_grad_checkpointing(True)
                rec.update(bench(model, spec.input_size, batch))
                rec["grad_checkpointing"] = True
            rec["ok"] = True
        except Exception as e:  # record and continue: one broken model must not hide the others
            rec.update(ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
        print(json.dumps({k: v for k, v in rec.items() if k != "trace"}), flush=True)
        out.append(rec)
        model = None
        torch.cuda.empty_cache()
    return out


def step(name: str, fn) -> dict:
    t0 = time.perf_counter()
    try:
        return {"step": name, "ok": True, "result": fn(), "seconds": round(time.perf_counter() - t0, 1)}
    except Exception as e:
        return {"step": name, "ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}


def cli(*args: str) -> dict:
    p = subprocess.run([sys.executable, "-m", *args], capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2000:])
    return {"cmd": " ".join(args), "stdout_tail": p.stdout[-1500:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/gpu_witness.json")
    ap.add_argument("--work", default="results/cache/witness")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    work = Path(args.work)
    report = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "capability": torch.cuda.get_device_capability(), "models": witness_models(args.batch)}

    synth = make_synthetic_manifest(work / "synth", n_classes=3, per_class=40, test_per_class=4)
    common = [f"data.manifest={synth}", "train.epochs=1", "train.warmup_epochs=0", "train.freeze_epochs=0",
              f"output.runs_dir={work / 'runs'}", f"output.results={work / 'runs.jsonl'}", "data.num_workers=4"]
    steps = [step(f"train_cli_{c}", lambda c=c: cli("compclass.train", "--config", f"configs/{c}.yaml", "--fold", "0",
                                                     "--set", *common, "train.batch_size=16"))
             for c in ("scratch_cnn", "convnext_tiny", "vit_b16")]

    def do_export():
        from compclass.export import export_onnx

        ck = glob.glob(str(work / "runs" / "convnext_tiny-*" / "fold0" / "best.pt"))[0]
        return export_onnx(ck, work / "export_convnext_tiny")

    steps.append(step("export_onnx_convnext_tiny", do_export))
    from compclass import classical, features, zeroshot

    for key in [k for k, e in REGISTRY.items() if e.zeroshot]:
        steps.append(step(f"zeroshot_{key}", lambda key=key: zeroshot.run_zeroshot(
            str(synth), key, [0], results=str(work / "runs.jsonl"))[0]["accuracy"]))
    steps.append(step("linear_probe_dinov2_vits14", lambda: features.run_probe(
        str(synth), "dinov2_vits14", [0], results=str(work / "runs.jsonl"), cache_dir=work / "features")[0]["macro_f1"]))
    steps.append(step("classical", lambda: classical.run_classical(str(synth), [0], results=str(work / "runs.jsonl"))[0]["accuracy"]))
    report["steps"] = steps
    runs = work / "runs.jsonl"
    report["train_runs"] = [{k: r.get(k) for k in ("name", "params", "train_time_s", "train_img_per_s", "peak_vram_mb",
                                                   "accuracy", "macro_f1")}
                            for r in map(json.loads, runs.read_text().splitlines()) if r["method"] == "finetune"] \
        if runs.exists() else []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    ok = sum(r["ok"] for r in report["models"])
    print(f"models ok {ok}/{len(report['models'])}; steps ok {sum(s['ok'] for s in steps)}/{len(steps)} -> {args.out}")


if __name__ == "__main__":
    main()
