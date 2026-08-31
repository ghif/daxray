"""Visual diagnostics for image batches."""

from __future__ import annotations

import math
import os
from typing import Any, Mapping, Sequence

import numpy as np


def _image_for_display(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Expected a grayscale image, got shape {image.shape}.")
    return np.clip(image, 0.0, 1.0)


def save_batch_grid(
    batch: Mapping[str, Any],
    output_path: str,
    *,
    columns: int = 4,
    title: str = "DAXRay batch preview",
    show_axes: bool = False,
) -> str:
    """Save a batch as a labeled image grid and return the output path."""
    if columns <= 0:
        raise ValueError("columns must be positive.")
    images = np.asarray(batch["image"])
    if images.ndim != 4:
        raise ValueError(f"Expected a batch shaped (N, C, H, W), got {images.shape}.")
    count = images.shape[0]
    if count == 0:
        raise ValueError("Cannot visualize an empty batch.")

    labels = np.asarray(batch.get("label", np.full(count, -1)), dtype=np.int32)
    masks = np.asarray(batch.get("label_mask", labels >= 0), dtype=bool)
    patient_ids: Sequence[str] = batch.get("patient_id", [f"sample-{i}" for i in range(count)])
    if len(labels) != count or len(masks) != count or len(patient_ids) != count:
        raise ValueError("Batch image, label, mask, and patient_id lengths must match.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.0 * columns, 4.2 * rows), squeeze=False)
    axes_flat = axes.ravel()
    for index, axis in enumerate(axes_flat):
        if index >= count:
            axis.axis("off")
            continue
        axis.imshow(_image_for_display(images[index]), cmap="gray", vmin=0.0, vmax=1.0)
        if masks[index]:
            label_text = "TB positive" if labels[index] == 1 else "TB negative"
            color = "#b91c1c" if labels[index] == 1 else "#166534"
        else:
            label_text = "unlabeled"
            color = "#6b7280"
        axis.set_title(f"{patient_ids[index]}\n{label_text}", color=color, fontsize=10)
        if not show_axes:
            axis.axis("off")

    figure.suptitle(title, fontsize=14)
    figure.tight_layout()
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path
