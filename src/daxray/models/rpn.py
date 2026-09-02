"""Region Proposal Network (RPN) in Flax NNX."""

import math
from typing import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np


def box_area(boxes: jax.Array) -> jax.Array:
    """Computes area for boxes of shape (N, 4) in [x1, y1, x2, y2]."""
    return jnp.maximum(boxes[:, 2] - boxes[:, 0], 0.0) * jnp.maximum(
        boxes[:, 3] - boxes[:, 1], 0.0
    )


def clip_boxes_to_image(boxes: jax.Array, image_shape: tuple[int, int]) -> jax.Array:
    """Clips boxes to image boundary (height, width).

    Args:
        boxes: Tensor of shape (N, 4) with coords [x1, y1, x2, y2].
        image_shape: (height, width).
    """
    height, width = image_shape
    x1 = jnp.clip(boxes[:, 0], 0, width)
    y1 = jnp.clip(boxes[:, 1], 0, height)
    x2 = jnp.clip(boxes[:, 2], 0, width)
    y2 = jnp.clip(boxes[:, 3], 0, height)
    return jnp.stack([x1, y1, x2, y2], axis=-1)


def decode_box_deltas(
    deltas: jax.Array,
    anchors: jax.Array,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    bbox_xform_clip: float = math.log(1000.0 / 16.0),
) -> jax.Array:
    """Decodes relative box deltas (dx, dy, dw, dh) into [x1, y1, x2, y2].

    Args:
        deltas: (N, 4) deltas or (N, K*4) class-specific deltas.
        anchors: (N, 4) reference boxes [x1, y1, x2, y2].
        weights: (wx, wy, ww, wh) scaling weights.
        bbox_xform_clip: Upper limit for dw and dh before exp.

    Returns:
        Decoded boxes of shape matching deltas.
    """
    widths = anchors[:, 2] - anchors[:, 0]
    heights = anchors[:, 3] - anchors[:, 1]
    ctr_x = anchors[:, 0] + 0.5 * widths
    ctr_y = anchors[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights

    if deltas.ndim == 2 and deltas.shape[1] > 4:
        # Multi-class deltas: shape (N, num_classes * 4)
        num_classes = deltas.shape[1] // 4
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        dw = jnp.clip(dw, max=bbox_xform_clip)
        dh = jnp.clip(dh, max=bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = jnp.exp(dw) * widths[:, None]
        pred_h = jnp.exp(dh) * heights[:, None]

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        # Stack into (N, num_classes, 4) -> (N, num_classes * 4)
        pred_boxes = jnp.stack([x1, y1, x2, y2], axis=-1).reshape(deltas.shape[0], num_classes, 4)
        return pred_boxes

    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = deltas[:, 2] / ww
    dh = deltas[:, 3] / wh

    dw = jnp.clip(dw, max=bbox_xform_clip)
    dh = jnp.clip(dh, max=bbox_xform_clip)

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = jnp.exp(dw) * widths
    pred_h = jnp.exp(dh) * heights

    x1 = pred_ctr_x - 0.5 * pred_w
    y1 = pred_ctr_y - 0.5 * pred_h
    x2 = pred_ctr_x + 0.5 * pred_w
    y2 = pred_ctr_y + 0.5 * pred_h

    return jnp.stack([x1, y1, x2, y2], axis=-1)


def nms_cpu(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.7) -> np.ndarray:
    """Non-Maximum Suppression on CPU.

    Args:
        boxes: (N, 4) [x1, y1, x2, y2].
        scores: (N,) confidence scores.
        iou_threshold: IoU overlap threshold.

    Returns:
        Indices of kept boxes.
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


class AnchorGenerator:
    """Generates anchors across multi-scale feature maps."""

    def __init__(
        self,
        sizes: Sequence[tuple[int, ...]] = ((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios: Sequence[tuple[float, ...]] = (
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
            (0.5, 1.0, 2.0),
        ),
    ):
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

    def _generate_base_anchors(
        self, scales: tuple[int, ...], ratios: tuple[float, ...]
    ) -> jax.Array:
        scales_arr = jnp.array(scales, dtype=jnp.float32)
        ratios_arr = jnp.array(ratios, dtype=jnp.float32)

        h_ratios = jnp.sqrt(ratios_arr)
        w_ratios = 1.0 / h_ratios

        ws = (w_ratios[:, None] * scales_arr[None, :]).reshape(-1)
        hs = (h_ratios[:, None] * scales_arr[None, :]).reshape(-1)

        base_anchors = jnp.stack([-ws, -hs, ws, hs], axis=1) / 2.0
        return jnp.round(base_anchors)

    def grid_anchors(
        self,
        grid_sizes: list[tuple[int, int]],
        strides: list[tuple[int, int]],
    ) -> list[jax.Array]:
        """Generates anchor coordinates for given grid sizes and strides."""
        anchors_per_level = []
        for (grid_h, grid_w), (stride_h, stride_w), size, ratio in zip(
            grid_sizes, strides, self.sizes, self.aspect_ratios
        ):
            base_anchors = self._generate_base_anchors(size, ratio)

            shifts_x = jnp.arange(0, grid_w, dtype=jnp.float32) * float(stride_w)
            shifts_y = jnp.arange(0, grid_h, dtype=jnp.float32) * float(stride_h)

            shift_y, shift_x = jnp.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            shifts = jnp.stack([shift_x, shift_y, shift_x, shift_y], axis=1)

            level_anchors = (shifts[:, None, :] + base_anchors[None, :, :]).reshape(-1, 4)
            anchors_per_level.append(level_anchors)

        return anchors_per_level

    def __call__(
        self,
        feature_maps: Sequence[jax.Array],
        image_shape: tuple[int, int],
    ) -> list[jax.Array]:
        img_h, img_w = image_shape
        grid_sizes = [(f.shape[1], f.shape[2]) if f.ndim == 4 else (f.shape[0], f.shape[1]) for f in feature_maps]
        strides = [(img_h // gh, img_w // gw) for (gh, gw) in grid_sizes]
        return self.grid_anchors(grid_sizes, strides)


class RPNHead(nnx.Module):
    """RPN head with 2 Conv3x3 layers followed by 1x1 cls and bbox heads."""

    def __init__(
        self,
        in_channels: int = 256,
        num_anchors: int = 3,
        *,
        rngs: nnx.Rngs,
    ):
        self.conv0 = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            use_bias=True,
            rngs=rngs,
        )
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            use_bias=True,
            rngs=rngs,
        )
        self.cls_logits = nnx.Conv(
            in_features=in_channels,
            out_features=num_anchors,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
        )
        self.bbox_pred = nnx.Conv(
            in_features=in_channels,
            out_features=num_anchors * 4,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Runs RPN head on single level feature map (B, H, W, C)."""
        t = self.conv0(x)
        t = nnx.relu(t)
        t = self.conv1(t)
        t = nnx.relu(t)

        cls_logits = self.cls_logits(t)  # (B, H, W, num_anchors)
        bbox_pred = self.bbox_pred(t)  # (B, H, W, num_anchors * 4)

        b, h, w, _ = cls_logits.shape
        cls_logits = cls_logits.reshape(b, h * w * 3, 1)
        bbox_pred = bbox_pred.reshape(b, h * w * 3, 4)

        return cls_logits, bbox_pred


class RegionProposalNetwork(nnx.Module):
    """Region Proposal Network orchestrating head, anchors, and proposal decoding."""

    def __init__(
        self,
        in_channels: int = 256,
        num_anchors: int = 3,
        pre_nms_top_n: int = 1000,
        post_nms_top_n: int = 1000,
        nms_thresh: float = 0.7,
        min_size: float = 1e-3,
        *,
        rngs: nnx.Rngs,
    ):
        self.head = RPNHead(in_channels=in_channels, num_anchors=num_anchors, rngs=rngs)
        self.anchor_generator = AnchorGenerator()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = min_size

    def __call__(
        self,
        features: dict[str, jax.Array],
        image_shape: tuple[int, int] = (512, 512),
    ) -> tuple[jax.Array, jax.Array]:
        """Generates candidate RoI proposals for a single image.

        Args:
            features: Dictionary containing feature maps ('0', '1', '2', '3', 'pool').
            image_shape: (height, width) of input image.

        Returns:
            Tuple of (proposals, proposal_scores).
        """
        ordered_keys = ["0", "1", "2", "3", "pool"]
        feat_list = [features[k] for k in ordered_keys if k in features]

        cls_logits_list = []
        bbox_pred_list = []
        for feat in feat_list:
            cls_l, bbox_p = self.head(feat)
            cls_logits_list.append(cls_l[0])  # take image index 0: shape (num_anchors_level, 1)
            bbox_pred_list.append(bbox_p[0])  # take image index 0: shape (num_anchors_level, 4)

        anchors_per_level = self.anchor_generator(feat_list, image_shape)

        # Per-level proposal decoding and filtering
        proposals_per_level = []
        scores_per_level = []

        for lvl_logits, lvl_deltas, lvl_anchors in zip(
            cls_logits_list, bbox_pred_list, anchors_per_level
        ):
            lvl_scores = jax.nn.sigmoid(lvl_logits[:, 0])

            # Select top pre_nms_top_n
            num_anchors = lvl_scores.shape[0]
            k = min(self.pre_nms_top_n, num_anchors)
            top_k_indices = jnp.argsort(lvl_scores)[::-1][:k]

            top_deltas = lvl_deltas[top_k_indices]
            top_anchors = lvl_anchors[top_k_indices]
            top_scores = lvl_scores[top_k_indices]

            # Decode box coordinates
            decoded_boxes = decode_box_deltas(top_deltas, top_anchors)
            decoded_boxes = clip_boxes_to_image(decoded_boxes, image_shape)

            # Filter small boxes
            ws = decoded_boxes[:, 2] - decoded_boxes[:, 0]
            hs = decoded_boxes[:, 3] - decoded_boxes[:, 1]
            keep_mask = (ws >= self.min_size) & (hs >= self.min_size)

            valid_boxes = np.array(decoded_boxes)[np.array(keep_mask)]
            valid_scores = np.array(top_scores)[np.array(keep_mask)]

            # Apply NMS per level
            keep_idx = nms_cpu(valid_boxes, valid_scores, iou_threshold=self.nms_thresh)
            keep_idx = keep_idx[: self.post_nms_top_n]

            if len(keep_idx) > 0:
                proposals_per_level.append(valid_boxes[keep_idx])
                scores_per_level.append(valid_scores[keep_idx])

        if len(proposals_per_level) == 0:
            return jnp.zeros((0, 4), dtype=jnp.float32), jnp.zeros((0,), dtype=jnp.float32)

        all_proposals = np.concatenate(proposals_per_level, axis=0)
        all_scores = np.concatenate(scores_per_level, axis=0)

        # Overall top post_nms_top_n across levels
        top_indices = np.argsort(all_scores)[::-1][: self.post_nms_top_n]
        final_proposals = jnp.array(all_proposals[top_indices], dtype=jnp.float32)
        final_scores = jnp.array(all_scores[top_indices], dtype=jnp.float32)

        return final_proposals, final_scores
