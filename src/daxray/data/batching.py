"""Batch construction at the NumPy/JAX boundary."""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .dicom import PreprocessingCache, load_cxr_image


def augment_images(
    images: np.ndarray,
    *,
    seed: int,
    horizontal_flip_prob: float = 0.0,
    contrast_range: float = 0.0,
    brightness_range: float = 0.0,
) -> np.ndarray:
    """Apply deterministic, mild image augmentation to an NHWC batch."""
    if images.ndim != 4:
        raise ValueError("images must have shape (N, H, W, C).")
    if not 0 <= horizontal_flip_prob <= 1 or contrast_range < 0 or brightness_range < 0:
        raise ValueError("invalid augmentation range")
    rng = np.random.default_rng(seed)
    augmented = np.asarray(images, dtype=np.float32).copy()
    if horizontal_flip_prob:
        flipped = rng.random(len(augmented)) < horizontal_flip_prob
        augmented[flipped] = augmented[flipped, :, ::-1, :]
    if contrast_range:
        contrast = rng.uniform(1 - contrast_range, 1 + contrast_range, len(augmented)).astype(np.float32)
        augmented = (augmented - 0.5) * contrast[:, None, None, None] + 0.5
    if brightness_range:
        brightness = rng.uniform(-brightness_range, brightness_range, len(augmented)).astype(np.float32)
        augmented = augmented + brightness[:, None, None, None]
    return np.clip(augmented, 0.0, 1.0)


def iter_batches(
    records: Sequence[Mapping[str, Any]],
    patient_ids: Sequence[str],
    *,
    batch_size: int,
    image_size: int = 224,
    resize_mode: str = "pad",
    shuffle: bool = False,
    seed: int = 7,
    drop_remainder: bool = False,
    as_jax: bool = False,
    layout: str = "NCHW",
    cache: PreprocessingCache | None = None,
    augment: bool = False,
    horizontal_flip_prob: float = 0.0,
    contrast_range: float = 0.0,
    brightness_range: float = 0.0,
) -> Iterator[dict[str, Any]]:
    """Yield ``(N, 1, H, W)`` image batches and optional labels.

    Missing labels are represented by ``-1`` plus a boolean ``label_mask``;
    this supports unlabeled target-domain adaptation without object arrays.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if layout not in {"NCHW", "NHWC"}:
        raise ValueError("layout must be 'NCHW' or 'NHWC'.")
    by_id = {record["patient_id"]: record for record in records}
    try:
        selected = [by_id[patient_id] for patient_id in patient_ids]
    except KeyError as exc:
        raise KeyError(f"Patient ID {exc.args[0]!r} is absent from the records.") from exc
    order = np.arange(len(selected))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        batch_records = [selected[index] for index in order[start : start + batch_size]]
        if len(batch_records) < batch_size and drop_remainder:
            continue
        paths = [record["image_path"] for record in batch_records]
        if cache is not None:
            images = np.stack(cache.load_many(paths, image_size, resize_mode))
        else:
            images = np.stack([load_cxr_image(path, image_size, resize_mode) for path in paths])
        if layout == "NHWC":
            images = np.moveaxis(images, 1, -1)
        if augment:
            images = augment_images(
                images, seed=seed + start,
                horizontal_flip_prob=horizontal_flip_prob,
                contrast_range=contrast_range,
                brightness_range=brightness_range,
            )
        label_mask = np.asarray([record.get("label") is not None for record in batch_records], dtype=bool)
        labels = np.asarray([record.get("label", -1) if record.get("label") is not None else -1 for record in batch_records], dtype=np.int32)
        batch: dict[str, Any] = {
            "image": images,
            "label": labels,
            "label_mask": label_mask,
            "patient_id": [record["patient_id"] for record in batch_records],
        }
        if as_jax:
            import jax
            batch["image"] = jax.device_put(images)
            batch["label"] = jax.device_put(labels)
            batch["label_mask"] = jax.device_put(label_mask)
        yield batch
