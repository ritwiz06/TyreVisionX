"""Compatibility helpers for torchvision import/runtime edge cases."""
from __future__ import annotations

import warnings
import sys

import torch

_TV_LIB = None


def _register_dummy_nms_op() -> None:
    """Register a minimal torchvision::nms op signature if missing.

    Some torch/torchvision builds fail during import-time fake registration with:
    "RuntimeError: operator torchvision::nms does not exist".
    Defining the operator schema before importing torchvision avoids that crash.
    """
    global _TV_LIB
    _TV_LIB = torch.library.Library("torchvision", "DEF")
    try:
        _TV_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")
    except RuntimeError as exc:
        if "same name and overload name" not in str(exc):
            raise


def _clear_partial_torchvision_modules() -> None:
    keys = [k for k in sys.modules if k == "torchvision" or k.startswith("torchvision.")]
    for key in keys:
        sys.modules.pop(key, None)


def _should_retry_import(exc: Exception) -> bool:
    msg = str(exc)
    if isinstance(exc, RuntimeError) and "torchvision::nms does not exist" in msg:
        return True
    if isinstance(exc, AttributeError) and (
        "partially initialized module 'torchvision'" in msg or "has no attribute 'extension'" in msg
    ):
        return True
    return False


def _import_models_once():
    import torchvision
    from torchvision import models

    # Guard against partially initialized torchvision module state in long-lived kernels.
    if not hasattr(torchvision, "extension"):
        raise AttributeError("torchvision has no attribute 'extension' (likely partial/circular import state)")
    return models


def load_torchvision_models():
    """Import and return torchvision.models with a targeted nms workaround."""
    try:
        return _import_models_once()
    except Exception as exc:
        if not _should_retry_import(exc):
            # Fallback: if module exists but is partially initialized, try one recovery pass.
            tv_mod = sys.modules.get("torchvision")
            if tv_mod is None or hasattr(tv_mod, "extension"):
                raise

        try:
            if "torchvision::nms does not exist" in str(exc):
                _register_dummy_nms_op()
            _clear_partial_torchvision_modules()
            models = _import_models_once()

            warnings.warn(
                "Applied torchvision import compatibility workaround for this environment.",
                RuntimeWarning,
            )
            return models
        except Exception:
            raise exc
