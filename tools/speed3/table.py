"""Decision table for the speed3 track from the frozen harness's results.jsonl (tools/speed2/table.py's columns).

    table.py [--tsv]

Every row is compared against baseline_fp32 (acceptance) and against the speed2 winner whose shapes it keeps (speed).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("speed2_table", Path(__file__).resolve().parents[1] / "speed2/table.py")
t2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t2)

W = "speed2_int8_res294_mlp3840"
t2.ROWS = {
    W: ("speed2_int8_res294", "speed2 winner: res294 + MLP 3840 (FLAP fluct, bias-comp), INT8"),
    "speed3_int8_c1": (W, "C1: fc2 least-squares refit per block, 512 pool images (external)"),
    "speed3_int8_q1": (W, "Q1: QAT distillation, LoRA 64 all GEMMs, pool images (external)"),
    "speed3_int8_q2": (W, "Q2: Q1 on the dev images (transductive)"),
}

if __name__ == "__main__":
    t2.main()
