"""Model families. ``build_model(key, num_classes)`` is the single entry point."""

from .registry import REGISTRY, PreprocessSpec, build_model, describe, list_models

__all__ = ["REGISTRY", "PreprocessSpec", "build_model", "describe", "list_models"]
