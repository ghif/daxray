"""TBX11K Faster R-CNN detector and visualization pipeline in JAX & Flax NNX."""

from pathlib import Path
from typing import Optional, Sequence, Union

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw

from daxray.data.tbx11k import BoundingBox, ID_TO_CLASS_NAME
from daxray.models.faster_rcnn import FasterRCNN
from daxray.models.weights import load_pretrained_faster_rcnn

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class TBXDetector:
    """End-to-end Faster R-CNN inference and visualization engine for TBX11K."""

    def __init__(
        self,
        model: Optional[FasterRCNN] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        target_size: tuple[int, int] = (512, 512),
        default_score_thresh: float = 0.001,
        default_nms_thresh: float = 0.5,
        *,
        rngs: Optional[nnx.Rngs] = None,
    ):
        if model is not None:
            self.model = model
        else:
            self.model = load_pretrained_faster_rcnn(
                checkpoint_path=checkpoint_path,
                rngs=rngs,
            )

        self.target_size = target_size
        self.default_score_thresh = default_score_thresh
        self.default_nms_thresh = default_nms_thresh

    def preprocess_image(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Prepares image into normalized (1, 512, 512, 3) tensor and returns original (width, height)."""
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype == np.uint8:
                img = Image.fromarray(image_input).convert("RGB")
            else:
                img = Image.fromarray((image_input * 255.0).astype(np.uint8)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        orig_w, orig_h = img.size
        img_resized = img.resize(self.target_size, Image.Resampling.BILINEAR)
        img_np = np.array(img_resized, dtype=np.float32) / 255.0

        norm_img = (img_np - IMAGENET_MEAN) / IMAGENET_STD
        return norm_img[None, ...], (orig_w, orig_h)

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        score_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
        rescale_to_original: bool = False,
    ) -> list[BoundingBox]:
        """Runs bounding box detection on an input image.

        Args:
            image_input: File path, PIL Image, or numpy array.
            score_thresh: Minimum confidence threshold.
            nms_thresh: IoU threshold for per-class NMS.
            rescale_to_original: If True, returns box coordinates mapped to original image dimensions.

        Returns:
            List of detected BoundingBox objects.
        """
        thresh = score_thresh if score_thresh is not None else self.default_score_thresh
        nms_th = nms_thresh if nms_thresh is not None else self.default_nms_thresh

        tensor, (orig_w, orig_h) = self.preprocess_image(image_input)
        raw_outputs = self.model(
            jnp.array(tensor),
            score_thresh=thresh,
            nms_thresh=nms_th,
        )

        dt = raw_outputs[0]
        boxes_np = np.array(dt["boxes"])
        labels_np = np.array(dt["labels"])
        scores_np = np.array(dt["scores"])

        scale_x = (float(orig_w) / float(self.target_size[0])) if rescale_to_original else 1.0
        scale_y = (float(orig_h) / float(self.target_size[1])) if rescale_to_original else 1.0

        boxes: list[BoundingBox] = []
        for b, label, score in zip(boxes_np, labels_np, scores_np):
            cat_id = int(label)
            boxes.append(
                BoundingBox(
                    x1=float(b[0]) * scale_x,
                    y1=float(b[1]) * scale_y,
                    x2=float(b[2]) * scale_x,
                    y2=float(b[3]) * scale_y,
                    category_id=cat_id,
                    category_name=ID_TO_CLASS_NAME.get(cat_id, "Lesion"),
                    confidence=float(score),
                )
            )

        return boxes

    def visualize(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        predicted_boxes: Sequence[BoundingBox],
        ground_truth_boxes: Optional[Sequence[BoundingBox]] = None,
        output_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
    ) -> Image.Image:
        """Draws ground-truth and predicted bounding boxes on the image and optionally saves preview.

        Args:
            image_input: Source image.
            predicted_boxes: Model predicted boxes in 512x512 coordinates.
            ground_truth_boxes: Ground truth boxes in 512x512 coordinates.
            output_path: Destination path to save visualization JPEG/PNG.
            title: Header title string.

        Returns:
            PIL Image with drawn bounding boxes and legend.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB").resize(self.target_size)
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype == np.uint8:
                img = Image.fromarray(image_input).convert("RGB").resize(self.target_size)
            else:
                img = Image.fromarray((image_input * 255.0).astype(np.uint8)).convert("RGB").resize(self.target_size)
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB").resize(self.target_size)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        draw = ImageDraw.Draw(img)

        # Color schemes:
        # Ground Truth: Green (#00FF00)
        # Active TB prediction: Red (#FF3333)
        # Latent TB prediction: Orange (#FFAA00)
        color_gt = "#00FF00"
        color_active = "#FF3333"
        color_latent = "#FFAA00"

        # 1. Draw Ground Truth boxes (dashed / thick green)
        if ground_truth_boxes:
            for b in ground_truth_boxes:
                draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=color_gt, width=3)
                label_text = f"GT: {b.category_name}"
                draw.rectangle([b.x1, max(0, b.y1 - 18), b.x1 + len(label_text) * 7 + 6, max(0, b.y1)], fill=color_gt)
                draw.text((b.x1 + 3, max(0, b.y1 - 16)), label_text, fill="black")

        # 2. Draw Predicted boxes
        for b in predicted_boxes:
            box_col = color_active if b.category_id == 1 else color_latent
            draw.rectangle([b.x1, b.y1, b.x2, b.y2], outline=box_col, width=2)
            score_str = f" ({b.confidence:.3f})" if b.confidence is not None else ""
            label_text = f"{b.category_name}{score_str}"
            text_y = min(img.height - 18, b.y2)
            draw.rectangle([b.x1, text_y, b.x1 + len(label_text) * 7 + 6, text_y + 16], fill=box_col)
            draw.text((b.x1 + 3, text_y + 1), label_text, fill="black")

        if output_path is not None:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_p)

        return img
