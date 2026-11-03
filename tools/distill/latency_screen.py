"""UNOFFICIAL batch-1 latency screen of ONNX graphs on onnxruntime CUDA EP (fp32), interleaved rounds.

    flock /tmp/626-gpu-bench.lock HARNESS_VENV/bin/python tools/distill/latency_screen.py A.onnx B.onnx ...

Each graph gets its own session; rounds alternate over graphs so drift and boost clocks hit all alike. Input and
output stay on the GPU (IO binding). Prints the median of round medians and the range per graph, in ms.
"""
import sys
import time

import numpy as np
import onnxruntime as ort

ort.preload_dlls()
sessions = {}
for p in sys.argv[1:]:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    s = ort.InferenceSession(p, so, providers=["CUDAExecutionProvider"])
    assert s.get_providers()[0] == "CUDAExecutionProvider", s.get_providers()
    i = s.get_inputs()[0]
    size = i.shape[-1]
    x = ort.OrtValue.ortvalue_from_numpy(np.random.default_rng(0).standard_normal((1, 3, size, size), np.float32), "cuda", 0)
    b = s.io_binding()
    b.bind_ortvalue_input(i.name, x)
    b.bind_output(s.get_outputs()[0].name, "cuda")
    sessions[p] = (s, b)
for s, b in sessions.values():
    t0 = time.time()
    while time.time() - t0 < 5:
        s.run_with_iobinding(b)
        b.synchronize_outputs()
rounds = {p: [] for p in sessions}
for _ in range(7):
    for p, (s, b) in sessions.items():
        ts = []
        for _ in range(50):
            t = time.perf_counter_ns()
            s.run_with_iobinding(b)
            b.synchronize_outputs()
            ts.append((time.perf_counter_ns() - t) / 1e6)
        rounds[p].append(float(np.median(ts)))
for p, r in rounds.items():
    print(f"{p.rsplit('/', 1)[-1]:40s} median {np.median(r):7.2f} ms  range {min(r):.2f}-{max(r):.2f}  (7 rounds x 50)")
