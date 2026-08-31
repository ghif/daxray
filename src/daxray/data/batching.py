"""Batch construction at the NumPy/JAX boundary."""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from .dicom import load_cxr_image


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
        images = np.stack([load_cxr_image(record["image_path"], image_size, resize_mode) for record in batch_records])
        if layout == "NHWC":
            images = np.moveaxis(images, 1, -1)
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
