"""compclass: electronic-component image classification across model families.

Modules: ``data`` (manifest, folds, transforms), ``models`` (registry of families), ``train`` (fine-tuning CLI),
``evaluate`` (metrics + results/runs.jsonl), ``features`` (frozen features + linear probe), ``zeroshot``
(CLIP/SigLIP prompts), ``classical`` (HOG + colour histogram + SVM), ``export`` (ONNX + sidecars).
"""

__version__ = "0.2.0"
