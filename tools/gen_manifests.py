"""Write the desktop app's model manifests (`<model dir>/model.json`) and their index (`exports/models.json`).

    python tools/gen_manifests.py [--exports exports] [--check]

A manifest is derived from the files an export already ships (preprocess.json, labels.json, runtime.json with its
providers, optimisation level and session_config, model_multi.json, export_report.json) plus the size and sha256 of
every file the app reads. The only facts written here by hand are in MODELS below: which graph file, the head ids,
titles, readout outputs and decision rules. Every derived field is asserted against its source, and so are the
checksums the export recorded. With the `onnx` module the graph's input and outputs are checked too (the harness venv
has it; plain python3 skips that check).

--check regenerates in memory and exits 1 if a committed model.json or the index differs, or if a graph's IO can be
checked neither against a recorded export report nor with the `onnx` module (then run it with the harness venv's
python). Run it after replacing an export; rerun without --check to rewrite. Adding a model is a row in MODELS (or a
hand-written model.json in its folder plus a line in exports/models.json; the app needs no code change).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

try:
    import onnx
except ImportError:
    onnx = None

SCHEMA = "model-manifest/1"
INDEX_SCHEMA = "model-index/1"
FILTERS = {"bicubic": "pillow_bicubic", "bilinear": "pillow_bilinear"}
PROVIDERS = {"TensorrtExecutionProvider": "tensorrt", "CUDAExecutionProvider": "cuda", "CPUExecutionProvider": "cpu"}
# Session-specific paths: the app puts its engine and timing caches in a per-user directory of its own.
APP_OWNED_OPTIONS = {"trt_engine_cache_path", "trt_timing_cache_path"}
# The ORT optimisation level the app has always used for these graphs (ort GraphOptimizationLevel::Level3).
LEGACY_OPT_LEVEL = "ORT_ENABLE_LAYOUT"
GPU_THEN_CPU = [{"ep": "cuda"}, {"ep": "cpu"}]


def classify(head_id, title, short, output, calibrated):
    """A classify head. Calibrated heads abstain below the export's threshold; the others always answer and flag
    a top-1 below it as low confidence (the threshold was copied from the deployed rule, not tuned for them)."""
    return dict(id=head_id, title=title, short=short, output=output, calibrated=calibrated)


MODELS = [
    dict(dir="itri626/A_scratch_cnn", id="A_scratch_cnn", graph="model.onnx", external=[], precision="fp32",
         heads=[classify("a", "A · Custom CNN (scratch)", "A · CNN", "probs", False)],
         providers=GPU_THEN_CPU, opt_level=LEGACY_OPT_LEVEL),
    dict(dir="itri626/B_convnext_tiny", id="B_convnext_tiny", graph="model.onnx", external=[], precision="fp32",
         heads=[classify("b", "B · ConvNeXt-T (fine-tuned)", "B · ConvNeXt-T", "probs", False)],
         providers=GPU_THEN_CPU, opt_level=LEGACY_OPT_LEVEL),
    dict(dir="final", id="final", graph="model_multi.onnx", external=["model.onnx.data"], precision="fp32",
         heads=[classify("c", "C · SigLIP ViT + linear head", "C · SigLIP + linear", "probs_c", False),
                classify("deployed", "Deployed (fused SigLIP)", "Deployed", "probs", True)],
         providers=GPU_THEN_CPU, opt_level=LEGACY_OPT_LEVEL),
    # providers, optimisation level and batch come from its runtime.json (TensorRT fp16 + int8, engine cache)
    dict(dir="fast", id="fast", graph="model.onnx", external=["model.onnx.data"], precision="int8_qdq",
         heads=[classify("fast", "Fast (fused SigLIP, 308 px INT8)", "Fast", "probs", True)]),
    # efficiency-study candidate (2026-11-05): MobileCLIP2-S3 tower + the deployed recipe, tools/views/package_candidate.py
    dict(dir="mobileclip2_s3", id="mobileclip2_s3", graph="model.onnx", external=[], precision="fp32",
         heads=[classify("small", "Small (fused MobileCLIP2-S3, 256 px)", "Small", "probs", True)],
         providers=GPU_THEN_CPU, opt_level=LEGACY_OPT_LEVEL),
]
# The first default whose preferred provider registers is the app's start-up head; the last needs only its files.
DEFAULT = ["fast", "deployed"]
PROVENANCE_KEYS = ["model", "report_model", "trunk", "config_hash", "fold", "checkpoint_sha256", "manifest_fingerprint"]
DETAIL_FILES = ["model_multi.json", "export_report.json", "fit_report.json", "runtime.json"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def load(p: Path):
    return json.loads(p.read_text())


def check(cond, msg):
    if not cond:
        raise SystemExit(f"gen_manifests: {msg}")


def recorded_io(d: Path, graph: str):
    """(input name, input shape, output names) as the export recorded them, or None."""
    mm = d / "model_multi.json"
    if graph == "model_multi.onnx" and mm.is_file():
        m = load(mm)
        return m["input"]["name"], m["input"]["shape"], [o["name"] for o in m["outputs"]]
    er = d / "export_report.json"
    if er.is_file() and "io" in load(er):
        io = load(er)["io"]
        return io["inputs"][0][0], io["inputs"][0][1], [o[0] for o in io["outputs"]]
    return None


def recorded_sha(d: Path, graph: str) -> dict[str, str]:
    """file name -> sha256 recorded by the export step."""
    out = {}
    mm = d / "model_multi.json"
    if graph == "model_multi.onnx" and mm.is_file():
        m = load(mm)
        out[m["file"]] = m["sha256"]
        for name in ("model.onnx.data", "labels.json"):
            out[name] = m["built_from"][name]
    er = d / "export_report.json"
    if er.is_file() and "onnx_sha256" in load(er):
        out[graph] = load(er)["onnx_sha256"]
    return out


def graph_io(path: Path):
    if onnx is None:
        return None
    g = onnx.load(str(path), load_external_data=False).graph
    shape = lambda vi: [d.dim_value or d.dim_param for d in vi.type.tensor_type.shape.dim]
    dtype = lambda vi: onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type)
    inits = {i.name for i in g.initializer}
    inputs = [i for i in g.input if i.name not in inits]
    return [(i.name, dtype(i), shape(i)) for i in inputs], [(o.name, dtype(o), shape(o)) for o in g.output]


def runtime_of(d: Path, spec: dict):
    """(graph_optimization_level, session_config, providers, max batch): from runtime.json when the export has one."""
    rp = d / "runtime.json"
    if not rp.is_file():
        return spec["opt_level"], {}, spec["providers"], 1
    rt = load(rp)
    providers = []
    for name in rt.get("provider_order", [rt["provider"]]):
        entry = {"ep": PROVIDERS[name]}
        if name == rt["provider"]:
            opts = {k: v for k, v in rt["options"].items() if k not in APP_OWNED_OPTIONS}
            if opts:
                entry["options"] = opts
        providers.append(entry)
    session_config = rt.get("session_config", {})
    check(all(isinstance(v, str) for v in session_config.values()), f"{rp}: session_config values must be strings")
    # onnxruntime reads "A,B" as one unknown optimizer name and disables nothing
    check("," not in session_config.get("optimization.disable_specified_optimizers", ""),
          f"{rp}: optimization.disable_specified_optimizers takes names separated by ';'")
    return rt.get("graph_optimization_level", "ORT_ENABLE_ALL"), session_config, providers, rt.get("static_batch", 1)


def build(exports: Path, spec: dict, checking: bool) -> dict:
    d = exports / spec["dir"]
    pre = load(d / "preprocess.json")
    labels = load(d / "labels.json")
    check(labels and all(isinstance(x, str) for x in labels), f"{d}/labels.json is not a list of names")
    size = pre["input_size"]
    files = [spec["graph"], *spec["external"], "labels.json"]
    for f in files:
        check((d / f).is_file(), f"{d / f} is missing")
    sums = {f: sha256(d / f) for f in files}
    for f, want in recorded_sha(d, spec["graph"]).items():
        check(sums.get(f) == want, f"{d / f}: sha256 {sums.get(f)} but the export recorded {want}")
    opt_level, session_config, providers, max_batch = runtime_of(d, spec)

    io = recorded_io(d, spec["graph"])
    actual = graph_io(d / spec["graph"])
    check(not (checking and io is None and actual is None),
          f"{d}: no recorded IO and no onnx module, so --check cannot verify {spec['graph']}'s IO against the graph "
          "(run it with a python that has onnx, e.g. the harness venv)")
    if io is None and actual is None and (d / "model.json").is_file():
        # writing, with nothing recorded and no onnx module: keep the IO the committed manifest states, and say so
        old = load(d / "model.json")
        h, w = old["input"]["size"]
        io = (old["input"]["name"], [old["input"]["batch"]["max"], 3, h, w], [x["task"]["readout"]["output"] for x in old["heads"]])
        print(f"{d}: graph IO not checked (no recorded IO, no onnx module); kept from model.json", file=sys.stderr)
    if actual is not None:
        (in_name, in_dtype, in_shape), = actual[0]
        check(in_dtype == "FLOAT", f"{spec['graph']}: input dtype {in_dtype}, the app feeds float32")
        out_names = [o[0] for o in actual[1]]
        if io is not None:
            check(io == (in_name, in_shape, out_names), f"{spec['graph']}: graph IO {actual} differs from recorded {io}")
        io = (in_name, in_shape, out_names)
    check(io is not None, f"{d}: no recorded IO and no onnx module to read it from the graph")
    in_name, in_shape, out_names = io
    check(list(in_shape) == [max_batch, 3, size, size], f"{spec['graph']}: input {in_shape} != [{max_batch},3,{size},{size}]")

    heads = []
    for h in spec["heads"]:
        check(h["output"] in out_names, f"{spec['graph']} has no output {h['output']} (outputs {out_names})")
        threshold = pre["abstain_threshold"]
        if h["calibrated"]:
            calibration = {"kind": "temperature", "t": pre["temperature"], "in_graph": True}
            decision = {"kind": "abstain", "below": threshold}
        else:
            calibration = {"kind": "none"}
            decision = {"kind": "commit", "low_confidence_below": threshold}
        heads.append({
            "id": h["id"], "title": h["title"], "short": h["short"],
            "task": {"kind": "classify", "readout": {"kind": "probs", "output": h["output"]}, "labels": "labels.json",
                     "calibration": calibration, "decision": decision},
        })

    provenance = {k: pre[k] for k in PROVENANCE_KEYS if k in pre}
    provenance["details"] = [f for f in DETAIL_FILES if (d / f).is_file()]
    manifest = {
        "schema": SCHEMA,
        "id": spec["id"],
        "graph": {"file": spec["graph"], "external_data": spec["external"], "precision": spec["precision"]},
        "files": {f: {"bytes": (d / f).stat().st_size, "sha256": sums[f]} for f in sorted(files)},
        "input": {
            "name": in_name, "dtype": "f32", "layout": pre.get("layout", "NCHW"), "size": [size, size],
            "batch": {"max": max_batch},
            "preprocess": {
                "kind": "resize_short_center_crop", "resize_short_side": pre["resize_short_side"],
                "filter": FILTERS[pre["interpolation"]], "color": pre.get("color_order", "RGB"),
                "scale": pre["scale"], "mean": pre["mean"], "std": pre["std"],
            },
        },
        "heads": heads,
        "runtime": {"graph_optimization_level": opt_level,
                    **({"session_config": session_config} if session_config else {}), "providers": providers},
        "provenance": provenance,
    }
    verify(manifest, d, pre)
    return manifest


def verify(m: dict, d: Path, pre: dict) -> None:
    """Re-read the sources and assert every derived field equals them."""
    p = m["input"]["preprocess"]
    check(m["input"]["size"] == [pre["input_size"]] * 2, "size")
    for k in ("resize_short_side", "scale", "mean", "std"):
        check(p[k] == pre[k], f"preprocess {k}")
    check(p["filter"] == FILTERS[pre["interpolation"]], "filter")
    check(m["input"]["layout"] == "NCHW" and p["color"] == "RGB", "only RGB NCHW input is supported by the app")
    check(p["resize_short_side"] >= pre["input_size"] and 0 not in p["std"], "preprocess ranges")
    for f, e in m["files"].items():
        check(e["bytes"] == (d / f).stat().st_size, f"{f} size")
    rp = d / "runtime.json"
    if rp.is_file():
        rt = load(rp)
        first = m["runtime"]["providers"][0]
        want = {k: v for k, v in rt["options"].items() if k not in APP_OWNED_OPTIONS}
        check(PROVIDERS[rt["provider"]] == first["ep"] and first.get("options", {}) == want, "runtime options")
        check(m["runtime"].get("session_config", {}) == rt.get("session_config", {}), "runtime session_config")
    for h in m["heads"]:
        t = h["task"]
        check(t["decision"]["kind"] != "abstain" or t["calibration"]["kind"] != "none", "abstain needs calibration")


def render(obj) -> str:
    return json.dumps(obj, indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", default=str(Path(__file__).resolve().parents[1] / "exports"))
    ap.add_argument("--check", action="store_true", help="compare with the committed files instead of writing")
    ap.add_argument("--only", action="append", default=[], metavar="ID",
                    help="build only these manifests (a machine without every model's graph files); the index still lists all")
    a = ap.parse_args()
    exports = Path(a.exports)
    outputs = {}
    for spec in MODELS:
        if a.only and spec["id"] not in a.only:
            continue
        outputs[exports / spec["dir"] / "model.json"] = render(build(exports, spec, a.check))
    ids = [h["id"] for spec in MODELS for h in spec["heads"]]
    check(len(ids) == len(set(ids)), f"head ids are not unique: {ids}")
    check(all(h in ids for h in DEFAULT), f"default {DEFAULT} names an unknown head")
    index = {"schema": INDEX_SCHEMA, "default": DEFAULT, "manifests": [f"{s['dir']}/model.json" for s in MODELS]}
    outputs[exports / "models.json"] = render(index)
    stale = []
    for path, text in outputs.items():
        if a.check:
            if not path.is_file() or path.read_text() != text:
                stale.append(path)
        else:
            path.write_text(text)
            print(f"wrote {path}")
    if onnx is None:
        print("graph IO checked against the recorded export reports only (no onnx module)", file=sys.stderr)
    if stale:
        raise SystemExit("gen_manifests --check: out of date: " + ", ".join(str(p) for p in stale))
    if a.check:
        print(f"{len(outputs)} files up to date")


if __name__ == "__main__":
    main()
