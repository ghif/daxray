"""DICOM loading and image preprocessing."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np


def load_cxr_image(path: str | Path, image_size: int = 224, resize_mode: str = "pad") -> np.ndarray:
    """Load a DICOM as normalized ``float32`` in ``(1, H, W)`` format."""
    if image_size <= 0 or resize_mode not in {"pad", "stretch"}:
        raise ValueError("image_size must be positive and resize_mode must be 'pad' or 'stretch'.")
    import pydicom
    from PIL import Image

    path_text = str(path)
    try:
        if path_text.startswith("gs://"):
            import fsspec
            with fsspec.open(path_text, mode="rb") as handle:
                raw = handle.read()
        else:
            raw = Path(path_text).read_bytes()
        ds = pydicom.dcmread(io.BytesIO(raw))
        pixels = ds.pixel_array.astype(np.float32)
    except Exception as exc:
        raise ValueError(f"Unable to read DICOM image at {path_text}") from exc

    pixels = pixels * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        pixels = pixels.max() - pixels
    low, high = float(pixels.min()), float(pixels.max())
    pixels = (pixels - low) / (high - low) if high > low else np.zeros_like(pixels)

    image = Image.fromarray(np.asarray(pixels * 255.0, dtype=np.uint8))
    if resize_mode == "stretch":
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    else:
        scale = min(image_size / image.width, image_size / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("L", (image_size, image_size), color=0)
        canvas.paste(resized, ((image_size - resized.width) // 2, (image_size - resized.height) // 2))
        image = canvas
    return np.asarray(image, dtype=np.float32)[None, :, :] / 255.0
