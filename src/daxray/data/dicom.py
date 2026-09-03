"""DICOM loading and image preprocessing."""

from __future__ import annotations

import io
import os
import tempfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Sequence

import numpy as np

PREPROCESSING_VERSION = "1"
DETECTION_PREPROCESSING_VERSION = "cxr-rait-detection-v1"


@dataclass(frozen=True)
class TransformDetails:
    """Transform details for mapping coordinates between original image and canvas."""

    orig_width: int
    orig_height: int
    target_width: int
    target_height: int
    scale: float
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int
    new_width: int
    new_height: int
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0
    photometric_interpretation: str = "MONOCHROME2"
    preprocessing_version: str = DETECTION_PREPROCESSING_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "orig_width": self.orig_width,
            "orig_height": self.orig_height,
            "target_width": self.target_width,
            "target_height": self.target_height,
            "scale": self.scale,
            "pad_left": self.pad_left,
            "pad_top": self.pad_top,
            "pad_right": self.pad_right,
            "pad_bottom": self.pad_bottom,
            "new_width": self.new_width,
            "new_height": self.new_height,
            "rescale_slope": self.rescale_slope,
            "rescale_intercept": self.rescale_intercept,
            "photometric_interpretation": self.photometric_interpretation,
            "preprocessing_version": self.preprocessing_version,
        }

    def canvas_to_orig_box(self, box: Sequence[float]) -> list[float]:
        """Maps [x1, y1, x2, y2] from target canvas coordinates to original image coordinates."""
        x1, y1, x2, y2 = box
        orig_x1 = max(0.0, min(float(self.orig_width), (x1 - self.pad_left) / self.scale))
        orig_y1 = max(0.0, min(float(self.orig_height), (y1 - self.pad_top) / self.scale))
        orig_x2 = max(0.0, min(float(self.orig_width), (x2 - self.pad_left) / self.scale))
        orig_y2 = max(0.0, min(float(self.orig_height), (y2 - self.pad_top) / self.scale))
        return [round(orig_x1, 2), round(orig_y1, 2), round(orig_x2, 2), round(orig_y2, 2)]

    def orig_to_canvas_box(self, box: Sequence[float]) -> list[float]:
        """Maps [x1, y1, x2, y2] from original image coordinates to target canvas coordinates."""
        x1, y1, x2, y2 = box
        canv_x1 = max(0.0, min(float(self.target_width), x1 * self.scale + self.pad_left))
        canv_y1 = max(0.0, min(float(self.target_height), y1 * self.scale + self.pad_top))
        canv_x2 = max(0.0, min(float(self.target_width), x2 * self.scale + self.pad_left))
        canv_y2 = max(0.0, min(float(self.target_height), y2 * self.scale + self.pad_top))
        return [round(canv_x1, 2), round(canv_y1, 2), round(canv_x2, 2), round(canv_y2, 2)]



@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    cached_bytes: int = 0
    preprocess_seconds: float = 0.0
    dicom_read_seconds: float = 0.0

    def as_dict(self) -> dict[str, int | float]:
        lookups = self.hits + self.misses
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_evictions": self.evictions,
            "cache_size_bytes": self.cached_bytes,
            "cache_hit_rate": self.hits / lookups if lookups else 0.0,
            "preprocess_seconds": self.preprocess_seconds,
            "dicom_read_seconds": self.dicom_read_seconds,
        }


class PreprocessingCache:
    """Bounded per-process cache for deterministic preprocessed images."""

    def __init__(self, mode: str = "memory", max_bytes: int = 1_073_741_824, read_workers: int = 8):
        if mode not in {"none", "memory", "ephemeral"}:
            raise ValueError("cache mode must be none, memory, or ephemeral.")
        if max_bytes <= 0:
            raise ValueError("cache max_bytes must be positive.")
        if read_workers <= 0:
            raise ValueError("cache read_workers must be positive.")
        self.mode = mode
        self.max_bytes = max_bytes
        self.read_workers = read_workers
        self.stats = CacheStats()
        self._lock = RLock()
        self._entries: OrderedDict[tuple[Any, ...], tuple[np.ndarray | Path, int]] = OrderedDict()
        self._counter = 0
        self._temporary = tempfile.TemporaryDirectory(prefix="daxray-preprocess-") if mode == "ephemeral" else None

    def get(self, key: tuple[Any, ...]) -> np.ndarray | None:
        if self.mode == "none":
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.stats.misses += 1
                return None
            value, _ = entry
            self._entries.move_to_end(key)
            self.stats.hits += 1
        return np.load(value, allow_pickle=False) if isinstance(value, Path) else value

    def put(self, key: tuple[Any, ...], image: np.ndarray) -> None:
        if self.mode == "none" or image.nbytes > self.max_bytes:
            return
        image = np.asarray(image, dtype=np.float32).copy()
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self.stats.cached_bytes -= old[1]
                if isinstance(old[0], Path):
                    old[0].unlink(missing_ok=True)
            value: np.ndarray | Path = image
            if self.mode == "ephemeral":
                path = Path(self._temporary.name) / f"{self._counter}.npy"
                self._counter += 1
                np.save(path, image, allow_pickle=False)
                value = path
            self._entries[key] = (value, image.nbytes)
            self.stats.cached_bytes += image.nbytes
            while self.stats.cached_bytes > self.max_bytes and self._entries:
                _, (evicted, size) = self._entries.popitem(last=False)
                self.stats.cached_bytes -= size
                self.stats.evictions += 1
                if isinstance(evicted, Path):
                    evicted.unlink(missing_ok=True)

    def close(self) -> None:
        with self._lock:
            self._entries.clear()
            self.stats.cached_bytes = 0
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def prewarm(self, paths: list[str | Path], image_size: int, resize_mode: str, *, read_workers: int = 8,
                progress: Any | None = None) -> None:
        """Preprocess unique source images concurrently into this cache."""
        if self.mode == "none":
            return
        unique_paths = list(dict.fromkeys(str(path) for path in paths))
        with ThreadPoolExecutor(max_workers=read_workers, thread_name_prefix="daxray-dicom") as executor:
            futures = [executor.submit(load_cxr_image, path, image_size, resize_mode, self) for path in unique_paths]
            for completed, future in enumerate(as_completed(futures), start=1):
                future.result()
                if progress is not None:
                    progress(completed, len(unique_paths))

    def load_many(self, paths: list[str | Path], image_size: int, resize_mode: str) -> list[np.ndarray]:
        """Load one batch concurrently while keeping cache ownership per process."""
        if len(paths) <= 1 or self.mode == "none":
            return [load_cxr_image(path, image_size, resize_mode, cache=self) for path in paths]
        with ThreadPoolExecutor(max_workers=min(self.read_workers, len(paths)),
                                thread_name_prefix="daxray-batch") as executor:
            return list(executor.map(
                lambda path: load_cxr_image(path, image_size, resize_mode, cache=self), paths
            ))


def _source_signature(path_text: str) -> tuple[Any, ...]:
    if path_text.startswith("gs://"):
        import fsspec
        fs, name = fsspec.core.url_to_fs(path_text)
        info = fs.info(name)
        return (info.get("generation"), info.get("etag"), info.get("size"), info.get("mtime"))
    stat = os.stat(path_text)
    return (stat.st_size, stat.st_mtime_ns)


def load_cxr_image(path: str | Path, image_size: int = 224, resize_mode: str = "pad",
                   cache: PreprocessingCache | None = None) -> np.ndarray:
    """Load a DICOM as normalized ``float32`` in ``(1, H, W)`` format."""
    if image_size <= 0 or resize_mode not in {"pad", "stretch"}:
        raise ValueError("image_size must be positive and resize_mode must be 'pad' or 'stretch'.")
    import pydicom
    from PIL import Image

    path_text = str(path)
    key = (path_text, _source_signature(path_text), image_size, resize_mode, PREPROCESSING_VERSION)
    cached = cache.get(key) if cache is not None else None
    if cached is not None:
        return cached
    read_started = time.perf_counter()
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
    if cache is not None:
        cache.stats.dicom_read_seconds += time.perf_counter() - read_started

    preprocess_started = time.perf_counter()
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
    result = np.asarray(image, dtype=np.float32)[None, :, :] / 255.0
    if cache is not None:
        cache.stats.preprocess_seconds += time.perf_counter() - preprocess_started
        cache.put(key, result)
    return result


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_cxr_image_for_detection(
    path: str | Path,
    target_size: tuple[int, int] = (512, 512),
    normalize: bool = True,
    cache: PreprocessingCache | None = None,
    patient_id: str | None = None,
) -> tuple[np.ndarray, TransformDetails, np.ndarray]:
    """Load a CXR image/DICOM prepared for Faster R-CNN detection.

    Applies rescale slope/intercept, MONOCHROME1 inversion, min-max intensity
    normalization, aspect-preserving resize with zero padding to target_size,
    grayscale-to-RGB conversion, and ImageNet normalization.

    Args:
        path: File path or gs:// URI.
        target_size: Target (width, height), default (512, 512).
        normalize: If True, applies ImageNet mean/std normalization.
        cache: Optional PreprocessingCache instance.
        patient_id: Optional patient identifier for error reporting.

    Returns:
        norm_tensor: (1, H, W, 3) float32 array ready for Faster R-CNN input.
        transform_details: TransformDetails instance recording geometry and box mappings.
        display_image: (H, W, 3) uint8 RGB array for visualization.
    """
    if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError(f"target_size must be positive (width, height), got {target_size}")

    import pydicom
    from PIL import Image

    path_text = str(path)
    read_started = time.perf_counter()
    try:
        if path_text.startswith("gs://"):
            import fsspec
            with fsspec.open(path_text, mode="rb") as handle:
                raw = handle.read()
        else:
            raw = Path(path_text).read_bytes()
    except Exception as exc:
        patient_suffix = f" for patient {patient_id}" if patient_id else ""
        raise ValueError(f"Unable to read DICOM image at {path_text}{patient_suffix}") from exc

    if cache is not None:
        cache.stats.dicom_read_seconds += time.perf_counter() - read_started

    preprocess_started = time.perf_counter()
    rescale_slope = 1.0
    rescale_intercept = 0.0
    photometric = "MONOCHROME2"

    try:
        ds = pydicom.dcmread(io.BytesIO(raw))
        pixels = ds.pixel_array.astype(np.float32)
        rescale_slope = float(getattr(ds, "RescaleSlope", 1.0))
        rescale_intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        pixels = pixels * rescale_slope + rescale_intercept
        photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
        if photometric == "MONOCHROME1":
            pixels = pixels.max() - pixels
        low, high = float(pixels.min()), float(pixels.max())
        pixels = (pixels - low) / (high - low) if high > low else np.zeros_like(pixels)
        orig_h, orig_w = pixels.shape[:2]
        base_image = Image.fromarray(np.asarray(pixels * 255.0, dtype=np.uint8)).convert("L")
    except Exception:
        # Fallback for standard non-DICOM fixtures (e.g. PNG / JPEG test data)
        try:
            base_image = Image.open(io.BytesIO(raw)).convert("L")
            orig_w, orig_h = base_image.size
        except Exception as exc:
            patient_suffix = f" for patient {patient_id}" if patient_id else ""
            raise ValueError(f"Unable to decode image at {path_text}{patient_suffix}") from exc

    target_w, target_h = target_size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top

    resized = base_image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), color=(0, 0, 0))
    canvas.paste(resized.convert("RGB"), (pad_left, pad_top))
    display_image = np.array(canvas, dtype=np.uint8)

    rgb_np = display_image.astype(np.float32) / 255.0
    if normalize:
        norm_tensor = (rgb_np - IMAGENET_MEAN) / IMAGENET_STD
    else:
        norm_tensor = rgb_np

    transform_details = TransformDetails(
        orig_width=orig_w,
        orig_height=orig_h,
        target_width=target_w,
        target_height=target_h,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        new_width=new_w,
        new_height=new_h,
        rescale_slope=rescale_slope,
        rescale_intercept=rescale_intercept,
        photometric_interpretation=photometric,
        preprocessing_version=DETECTION_PREPROCESSING_VERSION,
    )

    if cache is not None:
        cache.stats.preprocess_seconds += time.perf_counter() - preprocess_started

    return norm_tensor[None, ...], transform_details, display_image

