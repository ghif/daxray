"""Visual diagnostics for image batches."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from PIL import Image, ImageDraw

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


def draw_cxr_detection_overlay(
    display_image: np.ndarray,
    predicted_boxes: Sequence[Mapping[str, Any]],
    ground_truth_boxes: Sequence[Mapping[str, Any]] | None = None,
    header_text: str | None = None,
) -> Image.Image:
    """Draw ground-truth and predicted detection bounding boxes onto a 512x512 display canvas."""
    if isinstance(display_image, np.ndarray):
        if display_image.dtype != np.uint8:
            img = Image.fromarray((np.clip(display_image, 0.0, 1.0) * 255.0).astype(np.uint8)).convert("RGB")
        else:
            img = Image.fromarray(display_image).convert("RGB")
    elif isinstance(display_image, Image.Image):
        img = display_image.convert("RGB")
    else:
        raise ValueError(f"Unsupported image format: {type(display_image)}")

    draw = ImageDraw.Draw(img)

    # 1. Draw ground truth boxes (Green)
    if ground_truth_boxes:
        for gb in ground_truth_boxes:
            box = gb.get("box_canvas") or gb.get("box") or [gb.get("x1", 0), gb.get("y1", 0), gb.get("x2", 0), gb.get("y2", 0)]
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            draw.rectangle([x1, y1, x2, y2], outline="#00FF00", width=3)
            c_name = gb.get("category_name") or ("Active TB" if gb.get("category_id") == 1 else "Latent TB")
            label_text = f"GT: {c_name}"
            draw.rectangle([x1, max(0, y1 - 16), x1 + len(label_text) * 7 + 4, max(0, y1)], fill="#00FF00")
            draw.text((x1 + 2, max(0, y1 - 15)), label_text, fill="black")

    # 2. Draw predicted boxes (Red for Active TB, Orange for Latent TB)
    for pb in predicted_boxes:
        box = pb.get("box_canvas") or pb.get("box") or [pb.get("x1", 0), pb.get("y1", 0), pb.get("x2", 0), pb.get("y2", 0)]
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        cat_id = int(pb.get("category_id", 1))
        color = "#FF3333" if cat_id == 1 else "#FFAA00"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        score = pb.get("score") or pb.get("confidence")
        score_str = f" ({float(score):.3f})" if score is not None else ""
        c_name = pb.get("category_name") or ("Active TB" if cat_id == 1 else "Latent TB")
        label_text = f"{c_name}{score_str}"
        ty = min(img.height - 16, int(y2))
        draw.rectangle([x1, ty, x1 + len(label_text) * 7 + 4, ty + 16], fill=color)
        draw.text((x1 + 2, ty + 1), label_text, fill="black")

    # 3. Header bar if specified
    if header_text:
        header_h = 24
        header_img = Image.new("RGB", (img.width, img.height + header_h), color=(30, 30, 30))
        h_draw = ImageDraw.Draw(header_img)
        h_draw.text((8, 5), header_text, fill="white")
        header_img.paste(img, (0, header_h))
        return header_img

    return img


def generate_cxr_rait_galleries(
    predictions: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    max_per_category: int = 5,
) -> dict[str, list[str]]:
    """Generate visual preview galleries stratified by label availability, site, and error type.

    Categories:
        - true_positives (GT=1, Pred=1)
        - false_positives (GT=0, Pred=1)
        - false_negatives (GT=1, Pred=0)
        - true_negatives (GT=0, Pred=0)
        - unlabeled (GT=None)
        - by_site (previews grouped by clinical site)
    """
    from daxray.data.dicom import load_cxr_image_for_detection

    base_out = Path(output_dir) / "galleries"
    categories = ["true_positives", "false_positives", "false_negatives", "true_negatives", "unlabeled"]
    for cat in categories:
        (base_out / cat).mkdir(parents=True, exist_ok=True)

    saved_gallery_paths: dict[str, list[str]] = {cat: [] for cat in categories}
    saved_gallery_paths["by_site"] = []

    site_counts: dict[str, int] = {}

    for p in predictions:
        gt_info = p.get("ground_truth", {})
        lbl = gt_info.get("label")
        pred_lbl = p.get("predicted_label")
        site = str(p.get("site") or "SITE_DEFAULT")
        pid = str(p.get("patient_id") or "patient")
        study_id = str(p.get("study_id") or pid)

        # Categorize
        target_cat: str | None = None
        if lbl == 1 and pred_lbl == 1:
            target_cat = "true_positives"
        elif lbl == 0 and pred_lbl == 1:
            target_cat = "false_positives"
        elif lbl == 1 and pred_lbl == 0:
            target_cat = "false_negatives"
        elif lbl == 0 and pred_lbl == 0:
            target_cat = "true_negatives"
        elif lbl is None:
            target_cat = "unlabeled"

        should_save_cat = target_cat and len(saved_gallery_paths[target_cat]) < max_per_category
        should_save_site = site_counts.get(site, 0) < max_per_category

        if not (should_save_cat or should_save_site):
            continue

        img_path = p.get("image_path")
        if not img_path:
            continue

        try:
            _, transform_details, display_img = load_cxr_image_for_detection(
                img_path,
                target_size=(512, 512),
                normalize=False,
                patient_id=pid,
            )
        except Exception:
            continue

        max_s = p.get("scores", {}).get("max_tb_score", 0.0)
        gt_str = "TB" if lbl == 1 else ("Non-TB" if lbl == 0 else "Unlabeled")
        pred_str = "TB" if pred_lbl == 1 else "Non-TB"
        header = f"Patient: {pid} | Site: {site} | GT: {gt_str} | Pred: {pred_str} | Score: {max_s:.3f}"

        filtered_boxes = p.get("filtered_detections", [])
        gt_boxes = gt_info.get("boxes", [])

        vis_img = draw_cxr_detection_overlay(
            display_image=display_img,
            predicted_boxes=filtered_boxes,
            ground_truth_boxes=gt_boxes,
            header_text=header,
        )

        filename = f"{pid}_{study_id}.png"

        if should_save_cat and target_cat:
            cat_p = base_out / target_cat / filename
            vis_img.save(cat_p)
            saved_gallery_paths[target_cat].append(str(cat_p))

        if should_save_site:
            site_dir = base_out / "by_site" / site
            site_dir.mkdir(parents=True, exist_ok=True)
            site_p = site_dir / filename
            vis_img.save(site_p)
            saved_gallery_paths["by_site"].append(str(site_p))
            site_counts[site] = site_counts.get(site, 0) + 1

    return saved_gallery_paths

