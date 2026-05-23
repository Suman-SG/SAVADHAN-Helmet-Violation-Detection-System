"""Real-ESRGAN helper with automatic weight download and safe fallback.

This module exposes:
- `super_resolve(...)` for OCR-focused upscaling
- `get_sr_runtime_info()` to inspect which backend/weights are in use
"""

from __future__ import annotations

# ============ TORCHVISION COMPATIBILITY PATCH (MUST BE FIRST) ============
# Real-ESRGAN/basicsr imports from old torchvision.transforms.functional_tensor
# This module was renamed to _functional_tensor in newer torchvision.
# We need to patch sys.modules BEFORE any other imports.
import sys
try:
    import torchvision.transforms._functional_tensor as _ft
    sys.modules['torchvision.transforms.functional_tensor'] = _ft
except (ImportError, AttributeError):
    # Fallback: create a minimal shim if needed
    try:
        import torchvision.transforms.functional as ft
        sys.modules['torchvision.transforms.functional_tensor'] = ft
    except:
        pass
# =========================================================================

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_REAL_ESRGAN_MODEL = None
_REAL_ESRGAN_DEVICE = None
_REAL_ESRGAN_READY = False
_REAL_ESRGAN_WEIGHT_PATH = None
_REAL_ESRGAN_ERROR = None

REAL_ESRGAN_X2_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"


def _lanczos_upscale(image: np.ndarray, target_height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h >= target_height:
        return image
    scale = target_height / max(h, 1)
    new_w = max(1, int(w * scale))
    up = cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_LANCZOS4)
    blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(up, 1.45, blur, -0.45, 0)


def _ensure_realesrgan_model() -> bool:
    global _REAL_ESRGAN_MODEL
    global _REAL_ESRGAN_DEVICE
    global _REAL_ESRGAN_READY
    global _REAL_ESRGAN_WEIGHT_PATH
    global _REAL_ESRGAN_ERROR

    if _REAL_ESRGAN_READY:
        return True

    try:
        import urllib.request
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan.utils import RealESRGANer

        device = "cuda" if torch.cuda.is_available() else "cpu"

        weights_dir = Path("weights")
        weights_dir.mkdir(parents=True, exist_ok=True)
        weight_path = weights_dir / "RealESRGAN_x2.pth"

        if not weight_path.exists():
            urllib.request.urlretrieve(REAL_ESRGAN_X2_URL, str(weight_path))

        rrdb_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )

        model = RealESRGANer(
            scale=2,
            model_path=str(weight_path),
            model=rrdb_model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=(device == "cuda"),
            gpu_id=0 if device == "cuda" else None,
        )

        _REAL_ESRGAN_MODEL = model
        _REAL_ESRGAN_DEVICE = device
        _REAL_ESRGAN_READY = True
        _REAL_ESRGAN_WEIGHT_PATH = str(weight_path.resolve())
        _REAL_ESRGAN_ERROR = None
        return True
    except Exception as exc:
        _REAL_ESRGAN_READY = False
        _REAL_ESRGAN_MODEL = None
        _REAL_ESRGAN_DEVICE = None
        _REAL_ESRGAN_WEIGHT_PATH = None
        _REAL_ESRGAN_ERROR = str(exc)
        return False


def super_resolve(image: np.ndarray, target_height: int = 100, use_realesrgan: Optional[bool] = None) -> np.ndarray:
    """Super-resolve a BGR image to improve OCR readability.

    Args:
        image: Input BGR image
        target_height: Minimum output height
        use_realesrgan: None=auto, True=force attempt, False=disable and fallback
    """
    h = image.shape[0]
    if h >= target_height:
        return image

    if use_realesrgan is None:
        use_realesrgan = True

    if use_realesrgan and _ensure_realesrgan_model():
        try:
            outscale = max(2.0, float(target_height) / max(float(h), 1.0))
            sr_bgr, _ = _REAL_ESRGAN_MODEL.enhance(image, outscale=outscale)
            if sr_bgr.shape[0] < target_height:
                return _lanczos_upscale(sr_bgr, target_height)
            return sr_bgr
        except Exception:
            return _lanczos_upscale(image, target_height)

    return _lanczos_upscale(image, target_height)


def get_sr_runtime_info() -> dict:
    """Return backend/runtime details for reporting and diagnostics."""
    # Do not force model load here; just report current state.
    weight_exists = False
    weight_size_mb = 0.0
    if _REAL_ESRGAN_WEIGHT_PATH and os.path.exists(_REAL_ESRGAN_WEIGHT_PATH):
        weight_exists = True
        weight_size_mb = os.path.getsize(_REAL_ESRGAN_WEIGHT_PATH) / (1024 * 1024)

    return {
        "realesrgan_ready": _REAL_ESRGAN_READY,
        "device": _REAL_ESRGAN_DEVICE,
        "weight_path": _REAL_ESRGAN_WEIGHT_PATH,
        "weight_exists": weight_exists,
        "weight_size_mb": round(weight_size_mb, 2),
        "last_error": _REAL_ESRGAN_ERROR,
    }
