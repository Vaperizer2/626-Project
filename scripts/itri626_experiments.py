"""ITRI626 three-model experiment on ONE fixed train/validation/test split of data v2.

    uv run python scripts/itri626_experiments.py                 # everything (GPU recommended, ~15-25 min)
    uv run python scripts/itri626_experiments.py --steps head reference collect   # CPU-only parts

Split (tools/export_splits.py -> data/v2/splits/): train = dev folds 1-4 (2,010 images), validation = dev fold 0
(502), test = the sealed test set (418). Model selection (early stopping, the head's weight decay) uses the
validation split only; the test split is scored once per model for the final comparison.

Four main models (four families; three use pretrained weights). D was added on 2026-11-05 (efficiency study,
docs/EFFICIENCY_STUDY.md); A-C and their recorded numbers are unchanged:

  A  custom CNN baseline     scratch_cnn, configs/scratch_cnn.yaml + train.epochs=200 train.patience=40, random init
                             (schedule from the 2026-11 improvement study, docs/SCRATCH_CNN_IMPROVEMENT_REPORT.md)
  B  ConvNeXt-T fine-tuned   configs/convnext_tiny.yaml + train.epochs=20 train.patience=6 (main-benchmark recipe),
                             ImageNet-22k -> 1k weights (timm convnext_tiny.fb_in22k_ft_in1k)
  C  SigLIP So400m/14@384    frozen image tower (timm vit_so400m_patch14_siglip_384.webli) + nn.Linear(1152, 20)
                             trained on standardised cached features (compclass.linear_head); weight decay from
                             {1e-4, 1e-3, 1e-2, 1e-1} at lr 1e-3, chosen on the validation split
  D  MobileCLIP2-S3@256      frozen FastViT-MCi3 conv-attention hybrid (timm fastvit_mci3.apple_mclip2_dfndr2b, 123M)
                             + nn.Linear(3072, 20), trained exactly as C's head (same code, grid and selection rule);
                             eval view = the registry's crop fraction 1.0 (step ``head_D``)

Reference rows (consistency checks, not extra models), on the same test rows:
  R1 scikit-learn logistic-regression probe (compclass.features.fit_probe) on the same train split
  R2 the deployed fused model (exports/final/head.npz via compclass.final.fused_scores; its head was fit on all
     dev rows = train + validation, so it has seen 502 more training images than A-C)
  R3 the packaged Small model (exports/mobileclip2_s3/head.npz: D's tower with the deployed recipe, head fit on all
     dev rows; step ``reference_D``), the fused counterpart of D as R2 is of C

Outputs (results/itri626/): metrics.json, per_class.csv, histories.json, environment.json, hyperparameters.csv,
efficiency.csv, head_grid.csv, preds/<model>_{val,test}.npz, runs_index.json; checkpoints under runs/itri626/.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from compclass import train as T  # noqa: E402
from compclass.config import load_config  # noqa: E402
from compclass.data import ImageDataset, build_transforms, get_dev, get_fold, get_test, load_manifest  # noqa: E402
from compclass.evaluate import extended_metrics  # noqa: E402

OUT = ROOT / "results" / "itri626"
PREDS = OUT / "preds"
MANIFEST = "data/v2/manifest.json"
FOLD = 0  # validation fold; train = folds 1-4
RUNS_DIR = "runs/itri626"
SIGLIP = "siglip_so400m_384"
SMALL = "mobileclip2_s3"
WD_GRID = (1e-4, 1e-3, 1e-2, 1e-1)
HEAD_KW = dict(lr=1e-3, epochs=200, patience=20, batch_size=256, seed=0)
SMOKE: list[str] = []  # --smoke: tiny CPU dry run of the plumbing (numbers meaningless)

MODELS = {
    "A_scratch_cnn": {"label": "Custom CNN (scratch)", "family": "Plain VGG-style CNN (no pretraining)",
                      "config": "configs/scratch_cnn.yaml", "sets": ["train.epochs=200", "train.patience=40"]},
    "B_convnext_tiny": {"label": "ConvNeXt-T (fine-tuned)", "family": "ConvNeXt (modernised CNN)",
                        "config": "configs/convnext_tiny.yaml", "sets": ["train.epochs=20", "train.patience=6"]},
    "C_siglip_so400m_linear": {"label": "SigLIP So400m ViT (frozen + linear)",
                               "family": "Vision Transformer, SigLIP image-text pretraining"},
    "D_mobileclip2_s3_linear": {"label": "MobileCLIP2-S3 hybrid (frozen + linear)",
                                "family": "FastViT conv-attention hybrid, MobileCLIP 2 image-text pretraining"},
}
# The two frozen-tower models share one code path: registry key, run folder, weight-decay grid file, table strings.
FROZEN = {
    "C_siglip_so400m_linear": dict(key=SIGLIP, run="siglip_so400m_linear", grid="head_grid.csv", tag="C",
                                   variant="ViT-So400m/14, 384 px, SigLIP (WebLI)",
                                   weights="timm vit_so400m_patch14_siglip_384.webli (frozen)"),
    "D_mobileclip2_s3_linear": dict(key=SMALL, run="mobileclip2_s3_linear", grid="head_grid_D.csv", tag="D",
                                    variant="FastViT-MCi3 hybrid, 256 px, MobileCLIP 2 (DFNDR-2B)",
                                    weights="timm fastvit_mci3.apple_mclip2_dfndr2b (frozen)"),
}
REFS = {"R1_sklearn_probe": "SigLIP So400m + sklearn logistic-regression probe (same train split)",
        "R2_deployed_fused": "Deployed final model (probe + zero-shot fusion; head fit on train+val)",
        "R3_small_fused": "Small model (MobileCLIP2-S3 probe + zero-shot fusion; head fit on train+val)"}
PROFILE_ONLY: list[str] = []  # --profile-only: time just these models here and merge into the recorded profile.json


def jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")


def jload(path: Path):
    return json.loads(path.read_text())


def save_preds(name: str, split: str, ids, y, probs, classes) -> None:
    PREDS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PREDS / f"{name}_{split}.npz", ids=np.asarray(ids), y=np.asarray(y),
                        probs=np.asarray(probs, dtype=np.float32), classes=np.asarray(classes))


def cfg_for(name: str):
    spec = MODELS[name]
    sets = spec["sets"] + [f"output.runs_dir={RUNS_DIR}", "output.results=results/itri626/runs.jsonl"] + SMOKE
    return load_config(spec["config"], sets), sets


# ---------------------------------------------------------------------------------------------- A, B
def step_train(name: str) -> None:
    cfg, sets = cfg_for(name)
    rec = T.run(cfg, FOLD, test=True)
    run_dir = Path(rec["run_dir"])
    m = load_manifest(MANIFEST)
    v = np.load(run_dir / "val_preds.npz")
    save_preds(name, "val", v["ids"], v["y"], v["probs"], m.classes)
    t = np.load(run_dir / "test_preds.npz")
    save_preds(name, "test", t["ids"], t["y"], t["probs"], m.classes)
    keep = {k: rec[k] for k in ("name", "model", "config_hash", "fold", "params", "trainable_params", "input_size",
                                "epochs_run", "best_epoch", "train_time_s", "train_img_per_s", "peak_vram_mb",
                                "effective_batch", "amp", "run_dir")}
    keep["preprocess"] = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)["preprocess"]
    keep.update(cli=f"uv run python -m compclass.train --config {MODELS[name]['config']} --fold {FOLD} --test "
                    f"--set {' '.join(sets)}", config=cfg, history=rec["history"])
    jdump(keep, OUT / "runs" / f"{name}.json")


# ---------------------------------------------------------------------------------------------- C + refs
def siglip_features(key: str = SIGLIP):
    from compclass.features import extract_features

    m = load_manifest(MANIFEST)
    dev, test = get_dev(m), get_test(m, confirm=True)
    kw = {"num_workers": 0} if os.name == "nt" else {}  # Windows: a directly-run script hangs spawning loader workers
    Fdev = extract_features(key, dev, m, **kw)  # cached by row ids (results/cache/features/)
    Fte = extract_features(key, test, m, **kw)
    pos = {r["id"]: i for i, r in enumerate(dev)}
    tr, va = get_fold(m, FOLD)
    y = lambda rows: np.array([m.class_to_idx[r["label"]] for r in rows])  # noqa: E731
    return m, (tr, Fdev[[pos[r["id"]] for r in tr]], y(tr)), (va, Fdev[[pos[r["id"]] for r in va]], y(va)), \
        (test, Fte, y(test))


def siglip_spec() -> dict:
    """PreprocessSpec of the SigLIP tower, read from timm's pretrained config (what registry._timm resolves)."""
    import timm

    from compclass.models.registry import REGISTRY, PreprocessSpec

    c = timm.models.get_pretrained_cfg(REGISTRY[SIGLIP].timm_name).to_dict()
    return PreprocessSpec(c["input_size"][-1], tuple(c["mean"]), tuple(c["std"]), c["interpolation"],
                          c["crop_pct"]).to_dict()


def frozen_spec(key: str) -> dict:
    """Eval preprocessing of a frozen tower: timm's pretrained config for SigLIP (model C, as recorded); for the
    other towers the registry's spec, which carries the view they are registered with (crop fraction 1.0 for D)."""
    if key == SIGLIP:
        return siglip_spec()
    from compclass.models import build_model

    return build_model(key, num_classes=0, pretrained=False)[1].to_dict()


def step_head(name: str = "C_siglip_so400m_linear") -> None:
    """Linear head on a frozen tower's cached features: C (SigLIP) and D (MobileCLIP2-S3), one code path."""
    from compclass.linear_head import select_weight_decay

    fz = FROZEN[name]
    m, (tr, Xtr, ytr), (va, Xva, yva), (te, Xte, yte) = siglip_features(fz["key"])
    t0 = time.perf_counter()
    best, table = select_weight_decay(Xtr, ytr, Xva, yva, m.classes, grid=WD_GRID, **HEAD_KW)
    fit_s = time.perf_counter() - t0
    pd.DataFrame(table).to_csv(OUT / fz["grid"], index=False)
    save_preds(name, "val", [r["id"] for r in va], yva, best.predict_proba(Xva), m.classes)
    save_preds(name, "test", [r["id"] for r in te], yte, best.predict_proba(Xte), m.classes)
    run_dir = ROOT / RUNS_DIR / fz["run"] / f"fold{FOLD}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best.state(), run_dir / "head.pt")
    n_head = sum(p.numel() for p in best.model.parameters())
    jdump({"name": fz["run"], "model": fz["key"], "fold": FOLD, "feature_dim": int(Xtr.shape[1]),
           "trainable_params": int(n_head), "epochs_run": len(best.history), "best_epoch": best.best_epoch,
           "train_time_s": round(fit_s, 1), "grid_time_note": "train_time_s covers all 4 weight-decay runs (CPU)",
           "settings": best.settings, "weight_decay_grid": table, "preprocess": frozen_spec(fz["key"]),
           "run_dir": str(run_dir.relative_to(ROOT)), "history": best.history}, OUT / "runs" / f"{name}.json")
    print(f"[{fz['tag']}] selected wd", best.settings["weight_decay"], "best epoch", best.best_epoch,
          "val F1 %.4f" % best.best_val_macro_f1, flush=True)


def step_reference() -> None:
    from compclass.features import fit_probe
    from compclass.final import fused_scores, log_softmax

    m, (tr, Xtr, ytr), (va, Xva, yva), (te, Xte, yte) = siglip_features()
    t0 = time.perf_counter()
    s = fit_probe(Xtr, ytr, np.array([r["cluster"] for r in tr]))
    save_preds("R1_sklearn_probe", "val", [r["id"] for r in va], yva, s.predict_proba(Xva), m.classes)
    save_preds("R1_sklearn_probe", "test", [r["id"] for r in te], yte, s.predict_proba(Xte), m.classes)
    head = dict(np.load("exports/final/head.npz"))
    pr = np.exp(log_softmax(fused_scores(Xte, head) / head["temperature"]))
    save_preds("R2_deployed_fused", "test", [r["id"] for r in te], yte, pr, m.classes)
    ref = jload(ROOT / "results" / "final" / "final_test_v2.json")
    acc = float((pr.argmax(1) == yte).mean())
    assert SMOKE or abs(acc - ref["accuracy"]) < 1e-9, (acc, ref["accuracy"])  # same rows, same head -> same answer
    path = OUT / "runs" / "references.json"
    refs = jload(path) if path.exists() else {}
    refs.update({"R1_sklearn_probe": {"best_C": s.best_params_["logisticregression__C"],
                                "fit_time_s": round(time.perf_counter() - t0, 1)},
           "R2_deployed_fused": {"matches_results_final_final_test_v2_json": True, "accuracy": acc,
                                 "temperature": float(head["temperature"]), "C": float(head["C"])}})
    jdump(refs, path)


def step_reference_D() -> None:
    """R3: the packaged Small model (exports/mobileclip2_s3, tools/views/package_candidate.py) on the same test rows.
    Its one sealed-test look is results/final_test_mobileclip2_s3.json; this recomputes the same predictions from the
    cached features and asserts they match that record, as step_reference does for R2."""
    from compclass.final import log_softmax

    m, _, _, (te, Xte, yte) = siglip_features(SMALL)
    head = dict(np.load("exports/mobileclip2_s3/head.npz"))
    lp = log_softmax(((Xte - head["mu"]) / head["sd"]) @ head["coef"].T + head["intercept"])
    e = Xte @ head["proj_w"].T + head["proj_b"]
    e = e / np.linalg.norm(e, axis=1, keepdims=True)
    pr = np.exp(log_softmax((lp + head["w_zs"] * log_softmax(head["t_zs"] * e @ head["W"].T)) / head["temperature"]))
    save_preds("R3_small_fused", "test", [r["id"] for r in te], yte, pr, m.classes)
    ref = jload(ROOT / "results" / "final_test_mobileclip2_s3.json")
    acc = float((pr.argmax(1) == yte).mean())
    assert SMOKE or abs(acc - ref["accuracy"]) < 1e-9, (acc, ref["accuracy"])
    refs = jload(OUT / "runs" / "references.json")
    refs["R3_small_fused"] = {"matches_results_final_test_mobileclip2_s3_json": True, "accuracy": acc,
                              "temperature": float(head["temperature"]), "C": float(head["C"])}
    jdump(refs, OUT / "runs" / "references.json")


# ---------------------------------------------------------------------------------------------- profile
def _time(fn, n: int, warm: int, sync) -> float:
    for _ in range(warm):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - t0) / n * 1000


def _c_tower_reference(gpu: bool, dev, sync) -> dict:
    """Model C's frozen tower timed on THIS machine with step_profile's procedure (random weights and input: the
    forward cost does not depend on their values). A model profiled on another PC than A-C carries this as its
    same-machine anchor, so its ratio to C does not mix two computers."""
    from compclass.models import build_model

    net, spec = build_model(SIGLIP, num_classes=0, pretrained=False)
    net.eval()
    X = torch.randn(32, 3, spec.input_size, spec.input_size)
    out = {"model": "C's tower (SigLIP So400m, no head)", "input_size": spec.input_size}
    with torch.no_grad():
        if gpu:
            net.to(dev)
            x1, xb = X[:1].to(dev), X.to(dev)

            def f1():
                with torch.autocast("cuda", dtype=torch.float16):
                    net(x1).float().softmax(1)

            def fb():
                with torch.autocast("cuda", dtype=torch.float16):
                    net(xb).float().softmax(1)

            out["gpu_ms_per_image_batch1_fp16"] = round(_time(f1, 50, 10, sync), 2)
            out["gpu_ms_per_image_batch32_fp16"] = round(_time(fb, 10, 3, sync) / 32, 3)
            net.cpu()
            torch.cuda.empty_cache()
        torch.set_num_threads(8)
        x1 = X[:1]
        out["cpu_ms_per_image_batch1_fp32_8threads"] = round(_time(lambda: net(x1).softmax(1), 5, 2, lambda: None), 1)
    return out


def step_profile() -> None:
    """Parameters, checkpoint size and forward-pass latency (preprocessed tensor in, probabilities out;
    JPEG decoding and resizing excluded) for the main models. With ``--profile-only`` just the named models are
    timed and merged into the recorded profile.json, each stamped with the machine it ran on (D was profiled on a
    different PC than A-C)."""
    from timm.utils.model import reparameterize_model

    from compclass.models import build_model

    m = load_manifest(MANIFEST)
    rows = get_fold(m, FOLD)[1][:64]
    rec = jload(OUT / "runs" / "profile.json") if PROFILE_ONLY else {}
    rec.pop("_note", None)
    gpu = torch.cuda.is_available()
    dev = torch.device("cuda" if gpu else "cpu")
    sync = torch.cuda.synchronize if gpu else (lambda: None)
    for name in (PROFILE_ONLY or MODELS):
        if name in FROZEN:
            fz = FROZEN[name]
            model, spec = build_model(fz["key"], num_classes=0, pretrained=True)
            model = reparameterize_model(model.eval())  # inference form: folds FastViT's branches; a no-op for the ViT
            st = torch.load(ROOT / RUNS_DIR / fz["run"] / f"fold{FOLD}" / "head.pt", weights_only=False)
            lin = torch.nn.Linear(st["weight"].shape[1], st["weight"].shape[0])
            lin.load_state_dict({"weight": st["weight"], "bias": st["bias"]})
            mu, sd = st["mu"], st["sd"]

            class Wrapped(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.trunk, self.lin = model, lin
                    self.register_buffer("mu", mu)
                    self.register_buffer("sd", sd)

                def forward(self, x):
                    return self.lin((self.trunk(x).float() - self.mu) / self.sd)

            net = Wrapped().eval()
            assert spec.to_dict() == frozen_spec(fz["key"]), (spec.to_dict(), frozen_spec(fz["key"]))
            ck_files = [ROOT / RUNS_DIR / fz["run"] / f"fold{FOLD}" / "head.pt"]
            trainable = st["weight"].numel() + st["bias"].numel()
        else:
            run = jload(OUT / "runs" / f"{name}.json")
            ck = torch.load(Path(run["run_dir"]) / "best.pt", map_location="cpu", weights_only=False)
            cfg = ck["config"]
            net, spec = build_model(cfg["model"]["key"], len(m.classes), pretrained=False,
                                    **cfg["model"].get("kwargs", {}))
            net.load_state_dict(ck["model"])
            net.eval()
            ck_files = [Path(run["run_dir"]) / "best.pt"]
            trainable = run["trainable_params"]
        params = sum(p.numel() for p in net.parameters())
        ds = ImageDataset(rows, m.root, m.class_to_idx, build_transforms(spec, None, train=False))
        X = torch.stack([ds[i][0] for i in range(len(ds))])
        r = {"params": int(params), "trainable_params": int(trainable), "input_size": spec.input_size,
             "checkpoint_files": [str((ROOT / p).resolve().relative_to(ROOT)) for p in ck_files],
             "checkpoint_mb": round(sum(p.stat().st_size for p in ck_files) / 2**20, 2),
             "fp32_weights_mb": round(params * 4 / 2**20, 1)}
        with torch.no_grad():
            if gpu:
                net.to(dev)
                x1, xb = X[:1].to(dev), X[:32].to(dev)

                def f1():
                    with torch.autocast("cuda", dtype=torch.float16):
                        net(x1).float().softmax(1)

                def fb():
                    with torch.autocast("cuda", dtype=torch.float16):
                        net(xb).float().softmax(1)

                r["gpu_ms_per_image_batch1_fp16"] = round(_time(f1, 50, 10, sync), 2)
                r["gpu_ms_per_image_batch32_fp16"] = round(_time(fb, 10, 3, sync) / 32, 3)
                net.cpu()
                torch.cuda.empty_cache()
            torch.set_num_threads(8)
            x1 = X[:1]
            n = 5 if name.startswith("C_") else 30
            r["cpu_ms_per_image_batch1_fp32_8threads"] = round(_time(lambda: net(x1).softmax(1), n, 2, lambda: None), 1)
        if PROFILE_ONLY:
            r["machine"] = {"gpu": torch.cuda.get_device_name(0) if gpu else None, "cpu": platform.processor(),
                            "platform": platform.platform()}
            if name != "C_siglip_so400m_linear":
                r["machine"]["c_tower_reference"] = _c_tower_reference(gpu, dev, sync)
        rec[name] = r
        print("[profile]", name, r, flush=True)
        del net
    rec["_note"] = ("Forward pass of a preprocessed tensor to softmax probabilities, after warm-up; excludes JPEG "
                    "decode and resizing. C = frozen SigLIP So400m image tower + standardisation + linear head; "
                    "its checkpoint is only the head (the frozen backbone is the public timm weight file). D = the "
                    "same with the MobileCLIP2-S3 tower in its inference form (timm reparameterize_model); an entry "
                    "with a 'machine' field was timed on that machine, the others on the PC in environment.json.")
    jdump(rec, OUT / "runs" / "profile.json")


# ---------------------------------------------------------------------------------------------- env
def step_environment() -> None:
    import importlib.metadata as md

    def ver(p):
        try:
            return md.version(p)
        except md.PackageNotFoundError:
            return None

    pk = ["torch", "torchvision", "timm", "open_clip_torch", "scikit-learn", "numpy", "pandas", "onnxruntime",
          "onnx", "pillow", "matplotlib", "nbconvert", "ipykernel"]
    env = {"python": platform.python_version(), "platform": platform.platform(), "packages": {p: ver(p) for p in pk},
           "cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env["gpu"] = {"name": p.name, "memory_gib": round(p.total_memory / 2**30, 2),
                      "compute_capability": f"{p.major}.{p.minor}"}
        try:
            env["nvidia_driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                                  capture_output=True, text=True).stdout.strip()
        except OSError:
            pass
    cpu = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                if ln.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else platform.processor()
    mem = next((int(ln.split()[1]) for ln in Path("/proc/meminfo").read_text().splitlines()
                if ln.startswith("MemTotal")), None) if Path("/proc/meminfo").exists() else None
    env.update(cpu=cpu, cpu_logical_cores=os.cpu_count(), ram_gib=round(mem / 2**20, 1) if mem else None)
    jdump(env, OUT / "environment.json")


# ---------------------------------------------------------------------------------------------- collect
def describe_aug(a: dict) -> str:
    parts = [f"RandomResizedCrop(scale={tuple(a['rrc_scale'])}, ratio=({a['rrc_ratio'][0]:.2f},{a['rrc_ratio'][1]:.2f}))"]
    if a.get("hflip"):
        parts.append(f"HorizontalFlip(p={a['hflip']})")
    if a.get("vflip"):
        parts.append(f"VerticalFlip(p={a['vflip']})")
    if a.get("randaugment"):
        parts.append(f"RandAugment(n={a['randaugment']['n']}, m={a['randaugment']['m']})")
    if a.get("color_jitter"):
        parts.append(f"ColorJitter({a['color_jitter']})")
    if a.get("random_erasing"):
        parts.append(f"RandomErasing(p={a['random_erasing']})")
    if a.get("mixup") or a.get("cutmix"):
        parts.append(f"Mixup({a.get('mixup')})/CutMix({a.get('cutmix')})")
    return "; ".join(parts)


def step_collect() -> None:
    m = load_manifest(MANIFEST)
    classes = m.classes
    profile = jload(OUT / "runs" / "profile.json") if (OUT / "runs" / "profile.json").exists() else {}
    metrics, per_class, histories, hp = {"split": {}, "models": {}, "references": {}}, [], {}, []
    for s, f in (("train", "train"), ("val", "val"), ("test", "test")):
        metrics["split"][s] = int(len(pd.read_csv(ROOT / "data" / "v2" / "splits" / f"{f}.csv")))
    metrics["split"].update(manifest_fingerprint=m.fingerprint, seed=m.raw["split_policy"]["seed"], val_fold=FOLD)
    test_ids = None
    for name, spec in MODELS.items():
        run = jload(OUT / "runs" / f"{name}.json")
        t = np.load(PREDS / f"{name}_test.npz")
        v = np.load(PREDS / f"{name}_val.npz")
        test_ids = t["ids"] if test_ids is None else test_ids
        assert (t["ids"] == test_ids).all(), "all models must be scored on the same test rows in the same order"
        tm, vm = extended_metrics(t["y"], t["probs"], classes), extended_metrics(v["y"], v["probs"], classes)
        rec = {"label": spec["label"], "family": spec["family"], "test": tm,
               "val_at_best_epoch": {k: vm[k] for k in ("accuracy", "macro_f1", "balanced_accuracy", "log_loss")},
               "best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"], "train_time_s": run["train_time_s"],
               "trainable_params": run["trainable_params"], "profile": profile.get(name)}
        if "config_hash" in run:
            rec["config_hash"] = run["config_hash"]
            rec["peak_vram_mb"] = run.get("peak_vram_mb")
        metrics["models"][name] = rec
        histories[name] = run["history"]
        for i, c in enumerate(classes):
            pc = tm["per_class"][c]
            per_class.append({"model": name, "class": c, "precision": pc["precision"], "recall": pc["recall"],
                              "f1": pc["f1"], "support": pc["support"]})
        if name in FROZEN:
            st = run["settings"]
            hp.append({"model": name, "name": spec["label"], "variant": FROZEN[name]["variant"],
                       "pretrained_weights": FROZEN[name]["weights"], "input_size": run["preprocess"]["input_size"],
                       "batch_size": st["batch_size"], "grad_accumulation": 1,
                       "epochs": f"max {st['max_epochs']} (ran {st['epochs_run']}, best epoch {run['best_epoch']})",
                       "early_stopping": f"val macro-F1, patience {st['patience']}", "optimiser": "AdamW",
                       "lr": st["lr"], "schedule": "constant", "weight_decay": st["weight_decay"],
                       "weight_decay_selection": "grid {1e-4,1e-3,1e-2,1e-1} on validation macro-F1",
                       "loss": "cross-entropy", "label_smoothing": 0.0, "class_weights_or_sampling": "none (uniform)",
                       "augmentation": "none (features of the eval-preprocessed image: {interpolation} resize of the "
                                       "short side to {resize_short_side}, centre crop {input_size}, normalise mean "
                                       "{mean} std {std})".format(**run["preprocess"]),
                       "amp": "fp16 feature extraction", "seed": st["seed"], "split_seed": metrics["split"]["seed"],
                       "trainable_params": run["trainable_params"],
                       "total_params": (profile.get(name) or {}).get("params")})
        else:
            cfg, t_, a = run["config"], run["config"]["train"], run["config"]["aug"]
            hp.append({"model": name, "name": spec["label"],
                       "variant": "width 32, 4 stages, dropout 0.3" if cfg["model"]["key"] == "scratch_cnn"
                       else "ConvNeXt-Tiny, drop_path 0.1",
                       "pretrained_weights": "none (random init, He)" if not cfg["model"].get("pretrained", True)
                       else "timm convnext_tiny.fb_in22k_ft_in1k (ImageNet-22k -> 1k)",
                       "input_size": run["input_size"], "batch_size": t_["batch_size"],
                       "grad_accumulation": t_.get("accum_steps", 1),
                       "epochs": f"max {t_['epochs']} (ran {run['epochs_run']}, best epoch {run['best_epoch']})",
                       "early_stopping": f"val macro-F1, patience {t_['patience']}", "optimiser": "AdamW",
                       "lr": t_["lr"],
                       "schedule": (f"linear warm-up {t_.get('warmup_epochs', 0)} ep + cosine to 0"
                                    + (f"; head-only first {t_['freeze_epochs']} ep (warm-up restarts at unfreeze)"
                                       if t_.get("freeze_epochs") else "")
                                    + (f"; layer-wise LR decay {t_['layer_decay']}" if t_.get("layer_decay") else "")),
                       "weight_decay": t_["weight_decay"], "weight_decay_selection": "fixed (config)",
                       "loss": "cross-entropy", "label_smoothing": t_.get("label_smoothing", 0.0),
                       "class_weights_or_sampling": "none (uniform shuffling)",
                       "augmentation": describe_aug(a), "amp": t_.get("amp"), "seed": t_.get("seed", 0),
                       "split_seed": metrics["split"]["seed"],
                       "trainable_params": run["trainable_params"], "total_params": run["params"]})
    refs = jload(OUT / "runs" / "references.json")
    for name, label in REFS.items():
        t = np.load(PREDS / f"{name}_test.npz")
        assert (t["ids"] == test_ids).all()
        tm = extended_metrics(t["y"], t["probs"], classes)
        metrics["references"][name] = {"label": label, "test": tm, **refs[name]}
    metrics["head_weight_decay_grid"] = pd.read_csv(OUT / "head_grid.csv").to_dict("records")
    metrics["head_weight_decay_grid_D"] = pd.read_csv(OUT / "head_grid_D.csv").to_dict("records")
    metrics["notes"] = {
        "train_loss": "A/B: label-smoothed (0.1) cross-entropy on augmented batches = the optimised objective; "
                      "train_ce: plain cross-entropy on the same batches; C, D: no smoothing, no augmentation",
        "val_loss": "plain cross-entropy (no label smoothing) on the validation split, eval preprocessing",
        "selection": "best epoch by validation macro-F1 (A, B, C, D); C's and D's weight decay by validation "
                     "macro-F1; the test split was not used for any choice",
        "R2": "deployed head was fit on train+val (all dev rows); reference only",
        "R3": "Small model: D's tower with the deployed recipe, head fit on train+val; reference only",
        "D": "added 2026-11-05; eval view crop fraction 1.0 (registry), C keeps timm's 0.9; profiled on another PC"}
    jdump(metrics, OUT / "metrics.json")
    pd.DataFrame(per_class).to_csv(OUT / "per_class.csv", index=False)
    jdump(histories, OUT / "histories.json")
    hp = pd.DataFrame(hp)
    for col in ("trainable_params", "total_params"):
        hp[col] = pd.array(hp[col], dtype="Int64")
    hp.to_csv(OUT / "hyperparameters.csv", index=False)
    eff = [{"model": n, **{k: (v["gpu"] if k == "machine" else v) for k, v in (profile.get(n) or {}).items()
                           if k != "checkpoint_files"}} for n in MODELS]  # machine: set only where it differs (D)
    pd.DataFrame(eff).to_csv(OUT / "efficiency.csv", index=False)
    print(pd.DataFrame({n: {k: r["test"][k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")}
                        for n, r in {**metrics["models"], **metrics["references"]}.items()}).T.round(4), flush=True)


STEPS = {"environment": step_environment, "train_A": lambda: step_train("A_scratch_cnn"),
         "train_B": lambda: step_train("B_convnext_tiny"), "head": step_head, "reference": step_reference,
         "head_D": lambda: step_head("D_mobileclip2_s3_linear"), "reference_D": step_reference_D,
         "profile": step_profile, "collect": step_collect}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", default=list(STEPS), choices=list(STEPS))
    ap.add_argument("--smoke", action="store_true", help="1 epoch x 2 steps, 3 head epochs: plumbing test only")
    ap.add_argument("--profile-only", nargs="+", default=[], choices=list(MODELS), metavar="MODEL",
                    help="profile step: time only these models on this machine and merge into runs/profile.json")
    a = ap.parse_args(argv)
    PROFILE_ONLY.extend(a.profile_only)
    if a.smoke:
        SMOKE.extend(["train.epochs=1", "train.max_steps_per_epoch=2", "data.num_workers=0", "train.batch_size=8"])
        HEAD_KW.update(epochs=3, patience=2)
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.steps:
        t0 = time.perf_counter()
        print(f"=== {s}", flush=True)
        STEPS[s]()
        print(f"=== {s} done in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
