"""time: batch-1 latency and batch-8 throughput of backbone + deployed head on legion's GPU.

Inputs and outputs stay on the GPU through IO binding, so a run measures the graph, not host copies. Each run is
``run_with_iobinding`` plus ``synchronize_outputs``, timed with ``perf_counter_ns``. Warmup runs until both a
minimum count and a minimum duration pass (cuDNN autotuning, TensorRT first-run work, GPU boost ramp).

legion's laptop GPU runs these at its thermal and power limits (87 C, SM clock stepping between ~900 and 1500 MHz
every few seconds), which moves absolute latency by up to 11% over an afternoon. So every batch-1 run of the
candidate is followed by one run of the baseline in a second session, and the median of the 300 per-pair ratios
candidate/baseline is the speed number: both runs of a pair see the same clock.
"""

from __future__ import annotations

import importlib.metadata as md
import os
import threading
import time

import numpy as np

from harness import (BASELINE, INPUT, MAX_BATCH, OUTPUT, PROBS, Candidate, append_result, deployed_graph,
                     gpu_lock, load_candidate, make_session, provider_info)

WARMUP_RUNS, WARMUP_S = 20, 10.0
ROUNDS, RUNS_B1, RUNS_B8 = 5, 60, 12
MIN_COS = 0.999


class GpuSampler:
    """NVML samples (SM/memory clock, temperature, power, utilisation) every 100 ms while active."""

    def __init__(self):
        import pynvml as nv

        self.nv = nv
        nv.nvmlInit()
        self.h = nv.nvmlDeviceGetHandleByIndex(0)
        self.samples: list[tuple[float, ...]] = []
        self._stop = threading.Event()

    def sample(self) -> tuple[float, ...]:
        nv, h = self.nv, self.h
        return (nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM), nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU), nv.nvmlDeviceGetPowerUsage(h) / 1000,
                nv.nvmlDeviceGetUtilizationRates(h).gpu)

    def _loop(self):
        while not self._stop.wait(0.1):
            self.samples.append(self.sample())

    def __enter__(self):
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()

    def summary(self) -> dict:
        a = np.array(self.samples or [self.sample()])
        return {"sm_clock_mhz_median": float(np.median(a[:, 0])), "sm_clock_mhz_min": float(a[:, 0].min()),
                "mem_clock_mhz_median": float(np.median(a[:, 1])), "temp_c_max": float(a[:, 2].max()),
                "power_w_median": float(np.median(a[:, 3])), "util_pct_median": float(np.median(a[:, 4])),
                "n_samples": len(a)}

    def static(self) -> dict:
        nv, h = self.nv, self.h
        procs = []
        for p in nv.nvmlDeviceGetComputeRunningProcesses(h):
            try:
                name = open(f"/proc/{p.pid}/comm").read().strip()
            except OSError:
                name = "?"
            procs.append({"pid": p.pid, "name": name, "mib": (p.usedGpuMemory or 0) // 2**20})
        return {"gpu": nv.nvmlDeviceGetName(h), "driver": nv.nvmlSystemGetDriverVersion(),
                "cuda_driver": nv.nvmlSystemGetCudaDriverVersion(), "max_sm_clock_mhz": nv.nvmlDeviceGetMaxClockInfo(
                    h, nv.NVML_CLOCK_SM), "idle": dict(zip(("sm_mhz", "mem_mhz", "temp_c", "power_w", "util_pct"),
                                                          self.sample())), "gpu_processes": procs}


class Runner:
    """One IO-bound input batch on one session; ``once`` is a timed, synchronised run."""

    def __init__(self, sess, x: np.ndarray, dim: int, n_classes: int):
        import onnxruntime as ort

        self.sess, self.n, self.io = sess, len(x), sess.io_binding()
        self.io.bind_ortvalue_input(INPUT, ort.OrtValue.ortvalue_from_numpy(np.ascontiguousarray(x), "cuda", 0))
        self.outs = {OUTPUT: ort.OrtValue.ortvalue_from_shape_and_type([self.n, dim], np.float32, "cuda", 0),
                     PROBS: ort.OrtValue.ortvalue_from_shape_and_type([self.n, n_classes], np.float32, "cuda", 0)}
        for name, v in self.outs.items():
            self.io.bind_ortvalue_output(name, v)

    def once(self) -> float:
        t = time.perf_counter_ns()
        self.sess.run_with_iobinding(self.io)
        self.io.synchronize_outputs()
        return (time.perf_counter_ns() - t) / 1e6

    def warmup(self) -> int:
        t_end, n = time.monotonic() + WARMUP_S, 0
        while n < WARMUP_RUNS or time.monotonic() < t_end:
            self.once()
            n += 1
        return n

    def rounds(self, rounds: int, runs: int) -> np.ndarray:
        return np.array([[self.once() for _ in range(runs)] for _ in range(rounds)])

    def check(self, ref: np.ndarray) -> tuple[int, float]:
        """Rows whose output is non-finite, whose probs do not sum to 1, or whose embedding disagrees with ``ref``
        (the same images run one at a time), and the lowest cosine to ``ref``."""
        probs, emb = self.outs[PROBS].numpy(), self.outs[OUTPUT].numpy()
        cos = np.nan_to_num((emb * ref).sum(1) / (np.linalg.norm(emb, axis=1) * np.linalg.norm(ref, axis=1)), nan=-1)
        ok = np.isfinite(probs).all(1) & np.isfinite(emb).all(1) & (np.abs(probs.sum(1) - 1) < 1e-3) & (cos >= MIN_COS)
        return int((~ok).sum()), float(cos.min())


def stats(lat: np.ndarray, n: int) -> dict:
    flat = lat.ravel()
    return {"batch": n, "rounds": lat.shape[0], "runs_per_round": lat.shape[1], "median_ms": float(np.median(flat)),
            "p10_ms": float(np.percentile(flat, 10)), "p90_ms": float(np.percentile(flat, 90)),
            "min_ms": float(flat.min()), "max_ms": float(flat.max()),
            "round_medians_ms": np.median(lat, 1).round(3).tolist(), "img_per_s": float(n * 1000 / np.median(flat))}


def max_batch(c: Candidate) -> int:
    if c.fixed_batch:
        return 1
    prof = c.runtime["options"].get("trt_profile_max_shapes") if c.runtime["provider"] == "TensorrtExecutionProvider" \
        else None
    if not prof:
        return MAX_BATCH
    return max(int(p.split(":")[1].split("x")[0]) for p in prof.split(",") if p.split(":")[0] == INPUT)


def versions() -> dict:
    out = {}
    for pkg in ("onnxruntime-gpu", "nvidia-cudnn-cu13", "nvidia-cuda-runtime", "tensorrt-cu13-libs"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out


def run_time(path: str) -> dict:
    from evaluate import dev_rows, preprocessed

    c, ref = load_candidate(path), load_candidate(BASELINE)
    paired = c.sha != ref.sha
    m, rows = dev_rows()
    C = len(m.classes)
    X = preprocessed(c, m, rows[:MAX_BATCH])
    with gpu_lock():
        gpu = GpuSampler()
        static = gpu.static()
        load0 = os.getloadavg()
        sess = make_session(c, deployed_graph(c, C))
        info = provider_info(sess)
        rsess = make_session(ref, deployed_graph(ref, C)) if paired else None
        with gpu:
            single = np.concatenate([sess.run([OUTPUT], {INPUT: X[i:i + 1]})[0] for i in range(MAX_BATCH)])
            r1 = Runner(sess, X[:1], c.dim, C)
            warm = r1.warmup()
            if paired:
                rr = Runner(rsess, preprocessed(ref, m, rows[:1]), ref.dim, C)
                rr.warmup()
                runs = np.array([[(r1.once(), rr.once()) for _ in range(RUNS_B1)] for _ in range(ROUNDS)])
                lat, rlat = runs[..., 0], runs[..., 1]
            else:
                lat = r1.rounds(ROUNDS, RUNS_B1)
            bad, cos = r1.check(single[:1])
            b1 = {**stats(lat, 1), "warmup_runs": warm, "n_bad_outputs": bad, "cos_vs_plain_run": cos}
            rr = rsess = None  # batch 8 is unpaired; two So400m-size sessions plus batch-8 buffers overflow 6 GB
            b8 = None
            if max_batch(c) >= MAX_BATCH:
                r8 = Runner(sess, X, c.dim, C)
                r8.warmup()
                lat8 = r8.rounds(ROUNDS, RUNS_B8)
                bad, cos = r8.check(single)
                b8 = {**stats(lat8, MAX_BATCH), "n_bad_outputs": bad, "min_cos_vs_batch1": cos, "valid": bad == 0}
        del sess
    pair = None
    if paired:
        ratios = lat / rlat
        rounds = np.median(ratios, 1)
        pair = {"reference": ref.id, "reference_sha": ref.sha, "reference_median_ms": float(np.median(rlat)),
                "reference_round_medians_ms": np.median(rlat, 1).round(3).tolist(),
                "round_ratios": rounds.round(4).tolist(), "ratio_median": float(np.median(ratios)),
                "ratio_min": float(rounds.min()), "ratio_max": float(rounds.max())}
    raw = c.work / f"latency-{time.strftime('%Y%m%dT%H%M%S')}.npz"
    np.savez(raw, b1=lat, **({"ref": rlat} if paired else {}), **({"b8": lat8} if b8 else {}))
    n_err = b1["n_bad_outputs"] + (b8["n_bad_outputs"] if b8 else 0)
    rec = append_result({"kind": "time", "candidate": c.id, "candidate_sha": c.sha, "candidate_dir": str(c.dir),
                         "built_from": c.preprocess.get("built_from"), "n_images": MAX_BATCH if b8 else 1,
                         "n_errors": n_err, "n_timed_runs": ROUNDS * (RUNS_B1 + (RUNS_B8 if b8 else 0)),
                         "batch1": b1, "is_reference": not paired, "paired": pair, "batch8": b8, **info, **versions(),
                         **static, "gpu_during": gpu.summary(), "loadavg_1m": load0[0],
                         "loadavg_1m_after": os.getloadavg()[0], "raw_latencies": str(raw)})
    print(f"[{c.id}] batch-1 median {b1['median_ms']:.2f} ms (p10 {b1['p10_ms']:.2f}, p90 {b1['p90_ms']:.2f}; "
          f"round medians {b1['round_medians_ms']})")
    if pair:
        print(f"[{c.id}] vs {ref.id} in the same session: ratio {pair['ratio_median']:.4f} "
              f"(rounds {pair['round_ratios']}; baseline median {pair['reference_median_ms']:.2f} ms)")
    if b8:
        print(f"[{c.id}] batch-8 {b8['img_per_s']:.1f} img/s (median {b8['median_ms']:.2f} ms/batch)"
              + ("" if b8["valid"] else f"  INVALID: {b8['n_bad_outputs']} rows disagree with batch-1 "
                                         f"(min cos {b8['min_cos_vs_batch1']:.4f})"))
    print(f"[{c.id}] GPU {rec['gpu_during']}, load {load0[0]:.1f}, errors {n_err}")
    return rec
