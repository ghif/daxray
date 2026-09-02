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
from typing import Any

import numpy as np

PREPROCESSING_VERSION = "1"


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
