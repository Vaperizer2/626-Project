"""Classify electronic components in pictures with the exported final model (see docs/DEMO.md).

    uv run python -m compclass.demo --source photo.jpg
    uv run python -m compclass.demo --source some_folder/ --no-window --out annotated/
    uv run python -m compclass.demo --source "https://upload.wikimedia.org/.../Arduino_Uno_board.jpg"
    uv run python -m compclass.demo --source 0 --every 15          # webcam 0; 'q' quits, 's' saves a frame

Pieces and why:

* Model: ``exports/final/model.onnx`` (frozen SigLIP So400m/384 + fused linear heads + temperature, written by
  ``python -m compclass.final export``) run with **onnxruntime**. OpenCV's own DNN module cannot load this
  graph (``cv2.dnn.readNetFromONNX`` segfaults on it, see exports/final/export_report.json), so OpenCV is used
  only for image I/O, webcam capture, drawing and the window. ``--backend torch`` runs the same model through
  PyTorch (``compclass.final.load_final``; needs ``head.npz`` and the timm weights).
* Preprocessing is read from ``preprocess.json`` and reproduces the training-time *eval* transform exactly
  (torchvision ``Resize(short side) -> CenterCrop -> ToTensor -> Normalize``). It is implemented with Pillow,
  which is what torchvision uses on PIL images, so resampling is bit-identical; OpenCV's bicubic differs.
  Webcam frames (BGR numpy arrays) are converted to PIL RGB first.
* Output: top-3 labels with calibrated probabilities (temperature-scaled softmax). If the top probability is
  below ``abstain_threshold`` (from preprocess.json, chosen on out-of-fold predictions so that accepted
  predictions are >= 90 % correct) the overlay says UNSURE instead of a confident label.

This module needs only numpy, Pillow, OpenCV and onnxruntime (torch only for ``--backend torch``), so it also
runs as a plain script outside the project venv: ``python src/compclass/demo.py ...``.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageOps

try:  # optional: AVIF decoding for URL / file inputs
    import pillow_avif  # noqa: F401
except ImportError:  # pragma: no cover
    pass

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}
MAX_URL_BYTES = 20 * 2**20
URL_TIMEOUT_S = 20.0
# Wikimedia (and some other hosts) reject Python's default "Python-urllib/x.y" agent with HTTP 403 and ask
# automated clients to identify themselves. This identifies the demo honestly as a script; override with
# --user-agent (e.g. add a contact address), or pass --user-agent "" to send urllib's default.
DEFAULT_USER_AGENT = "compclass-demo/0.2 (educational image-classification demo; Python urllib)"
_PIL_INTERP = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR,
               "nearest": Image.Resampling.NEAREST}


# ------------------------------------------------------------------------------------------- contract
def load_contract(model_dir: str | Path) -> tuple[dict, list[str]]:
    """Read ``preprocess.json`` and ``labels.json`` from an export directory and sanity-check them."""
    model_dir = Path(model_dir)
    pre = json.loads((model_dir / "preprocess.json").read_text())
    labels = json.loads((model_dir / "labels.json").read_text())
    for k in ("input_size", "resize_short_side", "interpolation", "mean", "std", "scale", "abstain_threshold"):
        if k not in pre:
            raise ValueError(f"{model_dir / 'preprocess.json'} is missing {k!r}")
    if pre.get("color_order", "RGB") != "RGB" or pre.get("layout", "NCHW") != "NCHW":
        raise ValueError("this demo implements RGB / NCHW preprocessing only")
    return pre, labels


# ------------------------------------------------------------------------------------------ preprocess
def resized_size(w: int, h: int, short: int) -> tuple[int, int]:
    """Output (w, h) of torchvision ``Resize(short)``: the short side becomes ``short``, the long side is
    scaled and *truncated* (``int(short * long / short_side)``), exactly like torchvision."""
    if w <= h:
        return short, int(short * h / w)
    return int(short * w / h), short


def preprocess(img: Image.Image, pre: dict) -> np.ndarray:
    """PIL image -> float32 array [1, 3, S, S], identical to compclass.data.build_transforms(train=False)."""
    img = img.convert("RGB")
    size, short = int(pre["input_size"]), int(pre["resize_short_side"])
    nw, nh = resized_size(*img.size, short)
    if (nw, nh) != img.size:  # torchvision skips resampling when the size is already right
        img = img.resize((nw, nh), _PIL_INTERP[pre["interpolation"]])
    if nw < size or nh < size:
        raise ValueError(f"resize_short_side {short} < input_size {size}")
    left, top = int(round((nw - size) / 2.0)), int(round((nh - size) / 2.0))  # torchvision CenterCrop
    img = img.crop((left, top, left + size, top + size))
    x = np.asarray(img, dtype=np.float32) / np.float32(1.0 / float(pre["scale"]))  # == ToTensor (/255)
    x = (x - np.asarray(pre["mean"], np.float32)) / np.asarray(pre["std"], np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    """OpenCV BGR uint8 frame -> PIL RGB image."""
    return Image.fromarray(np.ascontiguousarray(frame[:, :, ::-1]))


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])


# -------------------------------------------------------------------------------------------- backends
class OnnxBackend:
    """onnxruntime session on model.onnx. The exported graph has a static batch of 1 -> one image per run."""

    def __init__(self, model_dir: Path, device: str = "auto"):
        import onnxruntime as ort

        avail = ort.get_available_providers()
        want_cuda = device in ("auto", "cuda") and "CUDAExecutionProvider" in avail
        if device == "cuda" and not want_cuda:
            print(f"[demo] CUDAExecutionProvider not available (have {avail}); using CPU. "
                  "See docs/DEMO.md for onnxruntime-gpu.", file=sys.stderr)
        if want_cuda and hasattr(ort, "preload_dlls"):
            try:  # onnxruntime-gpu >= 1.21: load CUDA/cuDNN from the nvidia-* wheels (e.g. those torch installs)
                ort.preload_dlls()
            except Exception as e:  # pragma: no cover - depends on the machine
                print(f"[demo] ort.preload_dlls() failed: {e}", file=sys.stderr)
        providers = (["CUDAExecutionProvider"] if want_cuda else []) + ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(model_dir / "model.onnx"), providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.name = f"onnxruntime {ort.__version__} [{self.sess.get_providers()[0]}]"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(["probs"], {self.input_name: x.astype(np.float32, copy=False)})[0][0]


class TorchBackend:
    """Same model through PyTorch (compclass.final.load_final): reference / fallback path."""

    def __init__(self, model_dir: Path, device: str = "auto"):
        import torch

        from compclass.final import load_final

        self.torch = torch
        dev = "cuda" if device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
        self.model, _ = load_final(model_dir / "head.npz")
        self.model = self.model.to(dev)
        self.device = torch.device(dev)
        self.name = f"torch {torch.__version__} [{dev}]"

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x).to(self.device))[1][0].float().cpu().numpy()


# -------------------------------------------------------------------------------------------- results
@dataclass
class Prediction:
    source: str
    top: list[tuple[str, float]]  # top-3 (label, calibrated probability)
    unsure: bool
    ms: float  # preprocessing + inference time
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return "UNSURE" if self.unsure else self.top[0][0]

    def line(self) -> str:
        if self.error:
            return f"{self.source}\tERROR\t{self.error}"
        top = ", ".join(f"{lab} {p:.3f}" for lab, p in self.top)
        return f"{self.source}\t{self.label}\t{self.top[0][1]:.3f}\ttop3: {top}\t{self.ms:.0f} ms"

    def to_json(self) -> dict:
        return {"source": self.source, "label": None if self.error else self.label,
                "top3": [[lab, round(float(p), 5)] for lab, p in self.top], "unsure": self.unsure,
                "ms": round(self.ms, 1), "error": self.error}


class Classifier:
    def __init__(self, model_dir: str | Path = "exports/final", backend: str = "onnx", device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.pre, self.labels = load_contract(self.model_dir)
        self.threshold = float(self.pre["abstain_threshold"])
        self.backend = (OnnxBackend if backend == "onnx" else TorchBackend)(self.model_dir, device)

    def probs(self, img: Image.Image) -> np.ndarray:
        return np.asarray(self.backend(preprocess(img, self.pre)), dtype=np.float64)

    def predict(self, img: Image.Image, source: str = "") -> Prediction:
        t0 = time.perf_counter()
        p = self.probs(img)
        ms = (time.perf_counter() - t0) * 1000
        idx = np.argsort(-p)[:3]
        top = [(self.labels[i], float(p[i])) for i in idx]
        return Prediction(source, top, top[0][1] < self.threshold, ms)


# --------------------------------------------------------------------------------------------- sources
def parse_source(src: str) -> tuple[str, object]:
    """'0' -> ('webcam', 0); 'http(s)://...' -> ('url', str); existing dir -> ('folder', Path);
    existing file -> ('file', Path). Raises ValueError otherwise. Pure: does no I/O beyond stat()."""
    s = src.strip()
    if re.fullmatch(r"\d+", s):
        return "webcam", int(s)
    if re.match(r"(?i)^https?://", s):
        return "url", s
    p = Path(s).expanduser()
    if p.is_dir():
        return "folder", p
    if p.is_file():
        return "file", p
    raise ValueError(f"--source {src!r} is not a webcam index, http(s) URL, folder or existing file")


def list_folder(folder: Path, recursive: bool = False) -> list[Path]:
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def open_image(path_or_bytes: Path | bytes) -> Image.Image:
    """Decode with Pillow, apply the EXIF orientation (phone photos), return RGB.
    (The v2 training images were EXIF-transposed when they were built, so this matches training.)"""
    f = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    with Image.open(f) as im:
        im.load()
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):  # flatten transparency on white, like tools/build_dataset_v2.py
            rgba = im.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            bg.alpha_composite(rgba)
            return bg.convert("RGB")
        return im.convert("RGB")


def fetch_url(url: str, timeout: float = URL_TIMEOUT_S, max_bytes: int = MAX_URL_BYTES,
              user_agent: str | None = DEFAULT_USER_AGENT) -> bytes:
    """Download at most ``max_bytes`` with urllib. Raises ValueError with a readable message on failure."""
    headers = {"User-Agent": user_agent} if user_agent else {}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            n = r.headers.get("Content-Length")
            if n and n.isdigit() and int(n) > max_bytes:
                raise ValueError(f"{int(n) / 2**20:.1f} MB exceeds the {max_bytes / 2**20:.0f} MB limit")
            data = r.read(max_bytes + 1)
    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP {e.code} {e.reason}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ValueError(f"download failed: {getattr(e, 'reason', e)}") from None
    if len(data) > max_bytes:
        raise ValueError(f"response exceeds the {max_bytes / 2**20:.0f} MB limit")
    return data


def iter_still_images(kind: str, value, recursive: bool = False,
                      user_agent: str | None = DEFAULT_USER_AGENT) -> Iterator[tuple[str, Image.Image | None, str | None]]:
    """Yield (name, image or None, error or None) for file / folder / url sources."""
    if kind == "url":
        try:
            yield value, open_image(fetch_url(value, user_agent=user_agent)), None
        except Exception as e:  # noqa: BLE001 - report and carry on
            yield value, None, f"{type(e).__name__}: {e}" if not isinstance(e, ValueError) else str(e)
        return
    paths = [value] if kind == "file" else list_folder(value, recursive)
    if not paths:
        print(f"[demo] no images ({', '.join(sorted(IMAGE_EXTS))}) in {value}", file=sys.stderr)
    for p in paths:
        try:
            yield str(p), open_image(p), None
        except Exception as e:  # noqa: BLE001
            yield str(p), None, f"cannot decode image: {e}"


def output_name(source: str, index: int) -> str:
    stem = Path(source.split("?")[0].rstrip("/")).stem or "image"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:80]
    return f"{index:03d}_{stem}.jpg"


# ----------------------------------------------------------------------------------------------- drawing
def annotate(bgr: np.ndarray, pred: Prediction, threshold: float, max_side: int = 1024) -> np.ndarray:
    """Return a copy (downscaled so the long side <= max_side) with a readable top-3 box in the corner."""
    import cv2

    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    img = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy()
    h, w = img.shape[:2]
    fs = max(0.45, min(w, h) / 700)  # font scale follows image size
    th = max(1, round(fs * 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines: list[tuple[str, tuple[int, int, int], float]] = []  # text, BGR colour, relative size
    if pred.error:
        lines.append(("ERROR", (0, 0, 255), 1.3))
        lines.append((pred.error[:60], (255, 255, 255), 0.8))
    else:
        if pred.unsure:
            lines.append((f"UNSURE (top < {threshold:.0%})", (0, 165, 255), 1.3))
            lines.append(("best guesses:", (200, 200, 200), 0.8))
        else:
            lines.append((f"{pred.top[0][0]}  {pred.top[0][1]:.0%}", (80, 230, 80), 1.3))
        for i, (lab, p) in enumerate(pred.top[(0 if pred.unsure else 1):], start=(1 if pred.unsure else 2)):
            lines.append((f"{i}. {lab}  {p:.0%}", (255, 255, 255), 0.85))
    sizes = [cv2.getTextSize(t, font, fs * r, th)[0] for t, _, r in lines]
    pad, gap = round(10 * fs), round(8 * fs)
    bw = max(sz[0] for sz in sizes) + 2 * pad
    bh = sum(sz[1] for sz in sizes) + gap * (len(lines) - 1) + 2 * pad + round(6 * fs)
    bw, bh = min(bw, w), min(bh, h)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (bw, bh), (20, 20, 20), -1)
    img = cv2.addWeighted(overlay, 0.72, img, 0.28, 0)
    edge = (0, 165, 255) if pred.unsure or pred.error else (80, 230, 80)
    cv2.rectangle(img, (0, 0), (bw, bh), edge, max(2, th))
    y = pad
    for (text, col, r), (tw, tht) in zip(lines, sizes):
        y += tht
        cv2.putText(img, text, (pad, y), font, fs * r, col, th, cv2.LINE_AA)
        y += gap
    return img


def gui_available() -> bool:
    """opencv-python-headless (the project default) has no HighGUI; detect it instead of crashing."""
    import os

    import cv2

    # The Qt-based opencv-python wheel aborts the whole process (not a Python exception) when there is no
    # display, so check for one first on Linux (SSH sessions, servers, CI).
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        cv2.namedWindow("compclass-probe", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("compclass-probe")
        return True
    except cv2.error:
        return False


# -------------------------------------------------------------------------------------------- main loops
def run_still(clf: Classifier, kind: str, value, args) -> list[Prediction]:
    import cv2

    show = not args.no_window and gui_available()
    if not args.no_window and not show:
        print("[demo] this OpenCV build has no GUI (opencv-python-headless); running headless. "
              "See docs/DEMO.md 'Showing a window'.", file=sys.stderr)
        if args.out is None:
            args.out = "demo_out"
            print("[demo] writing annotated images to demo_out/", file=sys.stderr)
    out = Path(args.out) if args.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    preds = []
    for i, (name, img, err) in enumerate(iter_still_images(kind, value, args.recursive, args.user_agent)):
        if err:
            pred = Prediction(name, [], False, 0.0, error=err)
        else:
            try:
                pred = clf.predict(img, name)
            except Exception as e:  # noqa: BLE001
                pred = Prediction(name, [], False, 0.0, error=f"inference failed: {e}")
        print(pred.line(), flush=True)
        preds.append(pred)
        if img is None and not out and not show:
            continue
        vis = annotate(pil_to_bgr(img) if img is not None else np.full((480, 640, 3), 40, np.uint8), pred,
                       clf.threshold, args.max_side)
        if out:
            cv2.imwrite(str(out / output_name(name, i)), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if show:
            cv2.imshow("compclass demo  (any key: next, q: quit)", vis)
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                break
    if show:
        cv2.destroyAllWindows()
    return preds


def run_webcam(clf: Classifier, index: int, args) -> list[Prediction]:
    import cv2

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[demo] cannot open webcam {index}", file=sys.stderr)
        return []
    show = not args.no_window and gui_available()
    if not args.no_window and not show:
        print("[demo] no OpenCV GUI available; running headless (stop with Ctrl-C or --max-frames).",
              file=sys.stderr)
    out = Path(args.out or "webcam_frames")
    every, preds, last, n = max(1, args.every), [], None, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[demo] webcam returned no frame; stopping", file=sys.stderr)
                break
            if n % every == 0:  # inference is ~0.9 s/image on CPU, so only every Nth frame
                last = clf.predict(bgr_to_pil(frame), f"webcam{index}:frame{n}")
                print(last.line(), flush=True)
                preds.append(last)
                if not show and args.out:
                    out.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out / f"frame_{n:06d}.jpg"), annotate(frame, last, clf.threshold, args.max_side))
            n += 1
            if show:
                vis = annotate(frame, last, clf.threshold, args.max_side) if last else frame
                cv2.imshow("compclass webcam  (q: quit, s: save frame)", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    out.mkdir(parents=True, exist_ok=True)
                    path = out / f"saved_{time.strftime('%Y%m%d_%H%M%S')}_{n:06d}.jpg"
                    cv2.imwrite(str(path), vis)
                    print(f"[demo] saved {path}", flush=True)
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
    return preds


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="compclass.demo", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="image file | folder | http(s) URL | webcam index (e.g. 0)")
    ap.add_argument("--model-dir", default="exports/final", help="folder with model.onnx, preprocess.json, labels.json")
    ap.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="auto = CUDA if the installed onnxruntime/torch supports it, else CPU")
    ap.add_argument("--out", default=None, help="write annotated JPEGs here")
    ap.add_argument("--no-window", action="store_true", help="do not open an OpenCV window")
    ap.add_argument("--every", type=int, default=10, help="webcam: classify every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0, help="webcam: stop after this many frames (0 = never)")
    ap.add_argument("--recursive", action="store_true", help="folder: include sub-folders")
    ap.add_argument("--max-side", type=int, default=1024, help="long side of the annotated output image")
    ap.add_argument("--json", default=None, help="also write one JSON line per image to this file")
    ap.add_argument("--user-agent", default=DEFAULT_USER_AGENT,
                    help="HTTP User-Agent for URL sources ('' = urllib default)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        kind, value = parse_source(args.source)
    except ValueError as e:
        print(f"[demo] {e}", file=sys.stderr)
        return 2
    t0 = time.perf_counter()
    clf = Classifier(args.model_dir, args.backend, args.device)
    print(f"[demo] {clf.backend.name}; model loaded in {time.perf_counter() - t0:.1f} s; "
          f"{len(clf.labels)} classes; abstain below {clf.threshold:.2f}", file=sys.stderr)
    preds = run_webcam(clf, value, args) if kind == "webcam" else run_still(clf, kind, value, args)
    if args.json:
        with open(args.json, "a") as f:
            for p in preds:
                f.write(json.dumps(p.to_json()) + "\n")
    ok = [p for p in preds if not p.error]
    if len(preds) > 1:
        ms = [p.ms for p in ok]
        print(f"[demo] {len(ok)}/{len(preds)} images classified, {sum(p.unsure for p in ok)} UNSURE"
              + (f", median {np.median(ms):.0f} ms/image" if ms else ""), file=sys.stderr)
    return 0 if ok or not preds else 1


if __name__ == "__main__":
    sys.exit(main())
