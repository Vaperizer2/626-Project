"""Offline tests for compclass.demo (no model download, no network beyond a 127.0.0.1 test server)."""

from __future__ import annotations

import http.server
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from compclass import demo

PRE = {"input_size": 384, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "interpolation": "bicubic",
       "crop_pct": 0.9, "resize_short_side": 427, "color_order": "RGB", "layout": "NCHW", "scale": 1 / 255,
       "outputs": ["logits", "probs"], "abstain_threshold": 0.5}
LABELS = [f"class_{i}" for i in range(20)]


def synthetic(w: int = 640, h: int = 480, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) % 256], -1) + rng.integers(0, 30, (h, w, 3))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ----------------------------------------------------------------------------------------- preprocessing
@pytest.mark.parametrize("w,h", [(640, 480), (480, 640), (427, 900), (1000, 1000), (300, 200)])
def test_preprocess_shape_dtype_range(w, h):
    x = demo.preprocess(synthetic(w, h), PRE)
    assert x.shape == (1, 3, 384, 384) and x.dtype == np.float32 and x.flags["C_CONTIGUOUS"]
    assert -1.0 - 1e-6 <= x.min() and x.max() <= 1.0 + 1e-6


def test_preprocess_normalisation_and_channel_order():
    img = Image.new("RGB", (500, 700), (255, 0, 51))  # uniform colour survives resize + crop exactly
    x = demo.preprocess(img, PRE)[0]
    np.testing.assert_allclose(x[0], 1.0, atol=1e-6)            # R: (1 - .5) / .5
    np.testing.assert_allclose(x[1], -1.0, atol=1e-6)           # G
    np.testing.assert_allclose(x[2], (51 / 255 - 0.5) / 0.5, atol=1e-6)  # B, and RGB (not BGR) order


def test_bgr_frame_conversion_matches_rgb():
    img = synthetic(400, 300)
    bgr = np.asarray(img)[:, :, ::-1]
    np.testing.assert_array_equal(demo.preprocess(demo.bgr_to_pil(bgr), PRE), demo.preprocess(img, PRE))


def test_resized_size_matches_torchvision_rule():
    assert demo.resized_size(640, 480, 427) == (int(427 * 640 / 480), 427)
    assert demo.resized_size(480, 640, 427) == (427, int(427 * 640 / 480))
    assert demo.resized_size(500, 500, 427) == (427, 427)


def test_preprocess_equals_torchvision_eval_transform():
    """The demo's PIL/numpy pipeline must be bit-for-bit the training-time eval transform."""
    pytest.importorskip("torchvision")
    from compclass.data import build_transforms

    spec = SimpleNamespace(input_size=384, mean=PRE["mean"], std=PRE["std"], interpolation="bicubic", crop_pct=0.9)
    tf = build_transforms(spec, None, train=False)
    for w, h in [(640, 480), (481, 1023), (427, 427), (2000, 1500)]:
        img = synthetic(w, h, seed=w)
        ref = tf(img).numpy()[None]
        np.testing.assert_allclose(demo.preprocess(img, PRE), ref, atol=1e-6, rtol=0)


def test_contract_file_if_present():
    path = Path(__file__).resolve().parents[1] / "exports" / "final"
    if not (path / "preprocess.json").exists():
        pytest.skip("no exported model here")
    pre, labels = demo.load_contract(path)
    assert int(round(pre["input_size"] / pre["crop_pct"])) == pre["resize_short_side"]
    assert len(labels) == 20 and 0 < pre["abstain_threshold"] < 1


# ------------------------------------------------------------------------------------------------ sources
def test_parse_source(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"x")
    assert demo.parse_source("0") == ("webcam", 0)
    assert demo.parse_source(" 2 ") == ("webcam", 2)
    assert demo.parse_source("https://upload.wikimedia.org/x.jpg") == ("url", "https://upload.wikimedia.org/x.jpg")
    assert demo.parse_source("HTTP://example.org/a.png")[0] == "url"
    assert demo.parse_source(str(tmp_path)) == ("folder", tmp_path)
    assert demo.parse_source(str(tmp_path / "a.jpg")) == ("file", tmp_path / "a.jpg")
    for bad in [str(tmp_path / "missing.jpg"), "ftp://example.org/a.jpg", "-1"]:
        with pytest.raises(ValueError):
            demo.parse_source(bad)


def test_folder_listing_and_bad_files(tmp_path):
    synthetic(64, 48).save(tmp_path / "b.png")
    synthetic(64, 48).save(tmp_path / "a.JPG", "JPEG")
    synthetic(64, 48).save(tmp_path / "c.webp", "WEBP")
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "broken.jpg").write_bytes(b"\xff\xd8 definitely not a jpeg")
    (tmp_path / "sub").mkdir()
    synthetic(64, 48).save(tmp_path / "sub" / "d.png")
    names = [p.name for p in demo.list_folder(tmp_path)]
    assert names == ["a.JPG", "b.png", "broken.jpg", "c.webp"]
    assert len(demo.list_folder(tmp_path, recursive=True)) == 5
    got = {Path(n).name: (img, err) for n, img, err in demo.iter_still_images("folder", tmp_path)}
    assert got["broken.jpg"][0] is None and "cannot decode" in got["broken.jpg"][1]
    assert got["c.webp"][0].mode == "RGB" and got["c.webp"][0].size == (64, 48)


def test_rgba_is_flattened_on_white(tmp_path):
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(tmp_path / "t.png")
    assert demo.open_image(tmp_path / "t.png").getpixel((0, 0)) == (255, 255, 255)


def test_output_name_is_safe():
    assert demo.output_name("https://x.org/a/My%20Photo (1).jpg?w=5", 3) == "003_My_20Photo_1_.jpg"
    assert demo.output_name("some/dir/", 0) == "000_dir.jpg"


@pytest.fixture()
def local_server(tmp_path):
    payload = {"ok.png": None, "big.bin": b"0" * 4096, "bad.jpg": b"not an image"}
    buf = tmp_path / "ok.png"
    synthetic(80, 60).save(buf)
    payload["ok.png"] = buf.read_bytes()
    seen_ua = []

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            seen_ua.append(self.headers.get("User-Agent"))
            body = payload.get(self.path.lstrip("/"))
            if body is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen_ua
    srv.shutdown()


def test_url_download_and_errors(local_server):
    base, seen_ua = local_server
    name, img, err = next(demo.iter_still_images("url", f"{base}/ok.png"))
    assert err is None and img.size == (80, 60)
    assert seen_ua[-1] == demo.DEFAULT_USER_AGENT
    with pytest.raises(ValueError, match="limit"):
        demo.fetch_url(f"{base}/big.bin", max_bytes=1000)
    with pytest.raises(ValueError, match="404"):
        demo.fetch_url(f"{base}/missing.jpg")
    _, img, err = next(demo.iter_still_images("url", f"{base}/bad.jpg"))
    assert img is None and err
    _, img, err = next(demo.iter_still_images("url", "http://127.0.0.1:9/nothing.jpg"))  # closed port
    assert img is None and "download failed" in err
    demo.fetch_url(f"{base}/ok.png", user_agent="")
    assert seen_ua[-1].startswith("Python-urllib")


# --------------------------------------------------------------------------------- end-to-end, tiny model
def tiny_model_dir(tmp_path: Path) -> Path:
    """A 20-class stand-in with the real I/O contract: input [1,3,384,384] -> (logits, probs) [1,20]."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper, numpy_helper

    W = np.random.default_rng(0).normal(size=(3, 20)).astype(np.float32) * 4
    nodes = [helper.make_node("ReduceMean", ["input", "axes"], ["m"], keepdims=0),
             helper.make_node("MatMul", ["m", "W"], ["logits"]),
             helper.make_node("Softmax", ["logits"], ["probs"], axis=1)]
    g = helper.make_graph(nodes, "tiny", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 384, 384])],
                          [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 20]),
                           helper.make_tensor_value_info("probs", TensorProto.FLOAT, [1, 20])],
                          [numpy_helper.from_array(W, "W"), numpy_helper.from_array(np.array([2, 3], np.int64), "axes")])
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 9
    d = tmp_path / "export"
    d.mkdir()
    onnx.save(model, d / "model.onnx")
    (d / "preprocess.json").write_text(json.dumps(PRE))
    (d / "labels.json").write_text(json.dumps(LABELS))
    return d


def test_end_to_end_headless(tmp_path, capsys):
    pytest.importorskip("cv2")
    d = tiny_model_dir(tmp_path)
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for i in range(3):
        synthetic(300 + 50 * i, 200, seed=i).save(imgs / f"im{i}.jpg")
    (imgs / "broken.png").write_bytes(b"nope")
    out, js = tmp_path / "ann", tmp_path / "res.jsonl"
    rc = demo.main(["--source", str(imgs), "--model-dir", str(d), "--no-window", "--out", str(out), "--json", str(js)])
    assert rc == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 4 and sum("\tERROR\t" in ln for ln in lines) == 1
    assert len(list(out.glob("*.jpg"))) == 4
    recs = [json.loads(x) for x in js.read_text().splitlines()]
    good = [r for r in recs if not r["error"]]
    for r in good:
        assert len(r["top3"]) == 3 and r["top3"][0][1] >= r["top3"][1][1] >= r["top3"][2][1]
        assert r["unsure"] == (r["top3"][0][1] < PRE["abstain_threshold"])
        assert r["label"] == ("UNSURE" if r["unsure"] else r["top3"][0][0])


def test_unsure_overlay_and_threshold(tmp_path):
    pytest.importorskip("cv2")
    d = tiny_model_dir(tmp_path)
    clf = demo.Classifier(d)
    clf.threshold = 1.01  # nothing can clear this -> UNSURE
    pred = clf.predict(synthetic(), "x")
    pred.unsure = pred.top[0][1] < clf.threshold
    assert pred.unsure and pred.label == "UNSURE" and "UNSURE" in pred.line()
    vis = demo.annotate(demo.pil_to_bgr(synthetic(2000, 1500)), pred, clf.threshold, max_side=1024)
    assert vis.shape == (768, 1024, 3)
