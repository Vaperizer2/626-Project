# ITRI626 report models for the desktop app

The three models of the ITRI626 report (`results/itri626/metrics.json`), exported for `apps/gpui-demo`. Nothing was
retrained. Rebuild on the GPU PC (CPU only, ~20 min):

    cd ~/Projects/626-Project && export HF_HOME=$HOME/models_cache/hf TORCH_HOME=$HOME/models_cache/torch
    uv run --frozen python tools/export_itri626_models.py all      # or: ab | c

| report model | where | input | preprocess (eval) | graph outputs | ONNX test acc (418) | report |
|---|---|---|---|---|---|---|
| A scratch CNN | `A_scratch_cnn/model.onnx` (4.8 MB, single file) | `input` [1,3,160,160] | short side → 183, **bilinear**, centre-crop 160, mean/std 0.5 | `logits`, `probs` [1,20] | 148/418 = 0.3541 | 0.3541 |
| B ConvNeXt-T fine-tuned | `B_convnext_tiny/model.onnx` (112 MB, single file) | `input` [1,3,224,224] | short side → 256, bicubic, centre-crop 224, ImageNet mean/std | `logits`, `probs` [1,20] | 266/418 = 0.6364 | 0.6364 |
| C SigLIP So400m frozen + linear | `../final/model_multi.onnx` (2.1 MB graph + shared `model.onnx.data`) | `input` [1,3,384,384] | `../final/preprocess.json` (427, bicubic, 384, 0.5) | `logits_c`, `probs_c` [1,20] | 343/418 = 0.8206 | 0.8230 (344) |
| deployed final model | `../final/model.onnx` or `model_multi.onnx` | same as C | same as C | `logits`, `probs` | 350/418 = 0.8373 | 0.8373 |

Each A/B folder follows the app contract (`apps/gpui-demo/src/contract.rs`): `model.onnx` + `preprocess.json` +
`labels.json`. The label order is identical to `exports/final/labels.json` (asserted). `probs = softmax(logits)`;
A, B and C are **not** temperature-calibrated. `abstain_threshold` 0.5 is copied from the deployed model, not tuned
for these models (UNSURE rate on test at 0.5: A 72.7 %, B 35.2 %). `export_report.json` in each folder holds the parity
and accuracy evidence. Parity is onnxruntime vs PyTorch fp32 on all 418 test images: max |Δp| 6.0e-7 (A) and 1.7e-6 (B),
with 100 % argmax agreement.

## `exports/final/model_multi.onnx` (models C + deployed in one backbone pass)

This file is the deployed graph plus three nodes on the pooled 1152-d SigLIP feature (tensor `select`):
`logits_c = select @ c_weight + c_bias` and `probs_c = softmax(logits_c)`. The weights fold the C head's train-split
standardisation: `c_weight = (W/sd)ᵀ` and `c_bias = b − (W/sd)·mu`, with W, b, mu and sd taken from
`runs/itri626/siglip_so400m_linear/fold0/head.pt`. The file reads the **unchanged** `model.onnx.data` (sha a6d11a2d…)
and must sit in the same folder. On all 418 test images, `logits`/`probs` are bit-identical to `model.onnx`. The
outputs are described in `model_multi.json`. The C accuracy differs by one image from the report (343 vs 344). The
reason is that the report's head ran on fp16-autocast GPU features, while the ONNX graph computes the feature in
fp32. On 16 test images the minimum cosine similarity between the two is 0.99997 (max relative difference 0.37 %).

## What the Rust app must replicate

- **Model A uses bilinear resizing.** `contract.rs` currently rejects anything but `"bicubic"`. Pillow's
  `BILINEAR` is the same separable `Resample.c` path as `preprocess.rs`, with filter `1 − |x|` and support 1.0
  instead of the cubic kernel with support 2.0. Feeding A a bicubic resize gives 145/418 (−3 images) and changes 7.4 %
  of the predictions.
- The resize sets the short side to `resize_short_side` and truncates the long side (`int(short·long/short_side)`).
  Resampling is skipped when the size already matches. The centre crop uses `round_half_even((W−S)/2)`. These are
  the same as the deployed model.
- The tensor is `x/255`, then `(x − mean)/std` in float32, laid out NCHW RGB with batch 1 (static).
- The app's EXIF/alpha handling matches `compclass.demo.open_image`. None of the 418 test images has a non-trivial
  EXIF orientation or alpha channel, so the evaluation does not exercise that path.

## Git

`*.onnx` is in `.gitignore`. The JSON files, this README and `tools/export_itri626_models.py` are trackable. The
three `.onnx` files are ignored. A's 4.8 MB `model.onnx` and the 2.1 MB `model_multi.onnx` can be committed with
`git add -f`. B's 112 MB `model.onnx` must not be committed.
