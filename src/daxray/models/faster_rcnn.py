"""Faster R-CNN (ResNet-50-FPN-V2) detection model in Flax NNX."""

from typing import Optional
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from daxray.models.resnet_fpn import BackboneWithFPN
from daxray.models.roi_align import multi_scale_roi_align
from daxray.models.rpn import RegionProposalNetwork, decode_box_deltas, nms_cpu


class FastRCNNConvFCHead(nnx.Module):
    """RoI box head: 4 Conv3x3-BN-ReLU layers followed by FC-1024-ReLU."""

    def __init__(
        self,
        in_channels: int = 256,
        fc_dim: int = 1024,
        resolution: int = 7,
        *,
        rngs: nnx.Rngs,
    ):
        self.conv0 = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            use_bias=False,
            rngs=rngs,
        )
        self.bn0 = nnx.BatchNorm(num_features=in_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)

        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            use_bias=False,
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(num_features=in_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)

        self.conv2 = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            use_bias=False,
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(num_features=in_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)

        self.conv3 = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=[(1, 1), (1, 1)],
            use_bias=False,
            rngs=rngs,
        )
        self.bn3 = nnx.BatchNorm(num_features=in_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)

        self.fc = nnx.Linear(
            in_features=in_channels * resolution * resolution,
            out_features=fc_dim,
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Processes pooled RoI features (N, 7, 7, 256)."""
        out = self.conv0(x)
        out = self.bn0(out)
        out = nnx.relu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = nnx.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nnx.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = nnx.relu(out)

        # Transpose from NHWC (N, 7, 7, 256) to NCHW (N, 256, 7, 7) before flattening
        # to match PyTorch torchvision flatten ordering
        n = out.shape[0]
        flattened = out.transpose(0, 3, 1, 2).reshape(n, -1)

        fc_out = self.fc(flattened)
        fc_out = nnx.relu(fc_out)
        return fc_out


class FastRCNNPredictor(nnx.Module):
    """Classification score and bounding box regression predictor."""

    def __init__(
        self,
        in_features: int = 1024,
        num_classes: int = 3,
        *,
        rngs: nnx.Rngs,
    ):
        self.cls_score = nnx.Linear(
            in_features=in_features,
            out_features=num_classes,
            use_bias=True,
            rngs=rngs,
        )
        self.bbox_pred = nnx.Linear(
            in_features=in_features,
            out_features=num_classes * 4,
            use_bias=True,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Returns (class_logits, box_regression) of shapes (N, num_classes) and (N, num_classes*4)."""
        cls_logits = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return cls_logits, bbox_deltas


class RoIHeads(nnx.Module):
    """RoI Heads combining RoIAlign pooling, box head, predictor, and postprocessing."""

    def __init__(
        self,
        num_classes: int = 3,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        box_detections_per_img: int = 100,
        bbox_reg_weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0),
        *,
        rngs: nnx.Rngs,
    ):
        self.box_head = FastRCNNConvFCHead(in_channels=256, fc_dim=1024, resolution=7, rngs=rngs)
        self.box_predictor = FastRCNNPredictor(in_features=1024, num_classes=num_classes, rngs=rngs)
        self.num_classes = num_classes
        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh
        self.box_detections_per_img = box_detections_per_img
        self.bbox_reg_weights = bbox_reg_weights

    def __call__(
        self,
        features: dict[str, jax.Array],
        proposals: jax.Array,
        image_shape: tuple[int, int] = (512, 512),
        score_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
    ) -> dict[str, jax.Array]:
        """Performs RoIAlign, head feature computation, and box post-processing."""
        thresh = score_thresh if score_thresh is not None else self.box_score_thresh
        nms_th = nms_thresh if nms_thresh is not None else self.box_nms_thresh

        if proposals.shape[0] == 0:
            return {
                "boxes": jnp.zeros((0, 4), dtype=jnp.float32),
                "labels": jnp.zeros((0,), dtype=jnp.int32),
                "scores": jnp.zeros((0,), dtype=jnp.float32),
            }

        # Multi-scale RoI Align
        pooled = multi_scale_roi_align(features, proposals, output_size=(7, 7))

        # Box head & predictor
        head_feats = self.box_head(pooled)
        class_logits, box_deltas = self.box_predictor(head_feats)

        # Softmax probabilities: shape (N, num_classes)
        probs = jax.nn.softmax(class_logits, axis=-1)

        # Decode boxes per class: shape (N, num_classes, 4)
        decoded_boxes = decode_box_deltas(
            box_deltas, proposals, weights=self.bbox_reg_weights
        )

        all_boxes_list = []
        all_scores_list = []
        all_labels_list = []

        decoded_boxes_np = np.array(decoded_boxes)
        probs_np = np.array(probs)

        # Filter boxes per foreground class (classes 1..num_classes-1)
        for cls_id in range(1, self.num_classes):
            cls_scores = probs_np[:, cls_id]
            cls_boxes = decoded_boxes_np[:, cls_id, :]

            # Clip to image boundary
            cls_boxes[:, 0] = np.clip(cls_boxes[:, 0], 0, image_shape[1])
            cls_boxes[:, 1] = np.clip(cls_boxes[:, 1], 0, image_shape[0])
            cls_boxes[:, 2] = np.clip(cls_boxes[:, 2], 0, image_shape[1])
            cls_boxes[:, 3] = np.clip(cls_boxes[:, 3], 0, image_shape[0])

            # Filter by score
            score_mask = cls_scores >= thresh
            cls_scores = cls_scores[score_mask]
            cls_boxes = cls_boxes[score_mask]

            # Filter small/empty boxes
            w = cls_boxes[:, 2] - cls_boxes[:, 0]
            h = cls_boxes[:, 3] - cls_boxes[:, 1]
            size_mask = (w >= 1e-2) & (h >= 1e-2)
            cls_scores = cls_scores[size_mask]
            cls_boxes = cls_boxes[size_mask]

            if len(cls_boxes) > 0:
                keep = nms_cpu(cls_boxes, cls_scores, iou_threshold=nms_th)
                all_boxes_list.append(cls_boxes[keep])
                all_scores_list.append(cls_scores[keep])
                all_labels_list.append(np.full((len(keep),), cls_id, dtype=np.int32))

        if len(all_boxes_list) == 0:
            return {
                "boxes": jnp.zeros((0, 4), dtype=jnp.float32),
                "labels": jnp.zeros((0,), dtype=jnp.int32),
                "scores": jnp.zeros((0,), dtype=jnp.float32),
            }

        final_boxes = np.concatenate(all_boxes_list, axis=0)
        final_scores = np.concatenate(all_scores_list, axis=0)
        final_labels = np.concatenate(all_labels_list, axis=0)

        # Keep top detections_per_img
        top_k_indices = np.argsort(final_scores)[::-1][: self.box_detections_per_img]

        return {
            "boxes": jnp.array(final_boxes[top_k_indices], dtype=jnp.float32),
            "labels": jnp.array(final_labels[top_k_indices], dtype=jnp.int32),
            "scores": jnp.array(final_scores[top_k_indices], dtype=jnp.float32),
        }


class FasterRCNN(nnx.Module):
    """Faster R-CNN (ResNet-50-FPN-V2) detection network in Flax NNX."""

    def __init__(
        self,
        num_classes: int = 3,
        box_score_thresh: float = 0.05,
        box_nms_thresh: float = 0.5,
        *,
        rngs: Optional[nnx.Rngs] = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.backbone = BackboneWithFPN(rngs=rngs)
        self.rpn = RegionProposalNetwork(in_channels=256, num_anchors=3, rngs=rngs)
        self.roi_heads = RoIHeads(
            num_classes=num_classes,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh,
            rngs=rngs,
        )

    def __call__(
        self,
        images: jax.Array,
        score_thresh: Optional[float] = None,
        nms_thresh: Optional[float] = None,
    ) -> list[dict[str, jax.Array]]:
        """Runs forward inference on a batch of images (B, H, W, 3).

        Args:
            images: Tensor of shape (B, H, W, 3) normalized in ImageNet range.
            score_thresh: Optional override for box confidence threshold.
            nms_thresh: Optional override for RoI NMS IoU threshold.

        Returns:
            List of detection dictionaries, one per image, each containing:
                'boxes': (K, 4) in [x1, y1, x2, y2] coords
                'labels': (K,) class IDs (1: Active TB, 2: Latent TB)
                'scores': (K,) confidence scores
        """
        batch_size, h, w, _ = images.shape
        image_shape = (h, w)

        results = []
        for b in range(batch_size):
            img_b = images[b : b + 1]
            features = self.backbone(img_b)
            proposals, proposal_scores = self.rpn(features, image_shape=image_shape)
            detections = self.roi_heads(
                features=features,
                proposals=proposals,
                image_shape=image_shape,
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
            )
            results.append(detections)

        return results
