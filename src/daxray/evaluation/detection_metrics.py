"""Evaluation metrics for TBX11K object detection and image-level classification."""

from dataclasses import dataclass
from typing import Sequence
import numpy as np


def compute_iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


@dataclass
class DetectionEvalResult:
    """Detection performance summary per class and overall."""

    ap_per_class: dict[int, float]
    map_50: float
    total_gt_per_class: dict[int, int]
    total_dt_per_class: dict[int, int]


def compute_ap_50_for_class(
    all_gt_boxes_per_img: list[list[list[float]]],
    all_pred_boxes_per_img: list[list[list[float]]],
    all_pred_scores_per_img: list[list[float]],
    iou_thresh: float = 0.5,
) -> tuple[float, int, int]:
    """Computes Average Precision (AP@0.5) using VOC continuous PR curve integration.

    Args:
        all_gt_boxes_per_img: List of ground-truth boxes per image for this class.
        all_pred_boxes_per_img: List of predicted boxes per image for this class.
        all_pred_scores_per_img: List of prediction confidence scores per image for this class.
        iou_thresh: IoU overlap threshold (default: 0.5).

    Returns:
        Tuple of (AP, total_gt_count, total_detections_count).
    """
    total_gt = sum(len(gts) for gts in all_gt_boxes_per_img)
    if total_gt == 0:
        return 0.0, 0, sum(len(dts) for dts in all_pred_boxes_per_img)

    # Flatten all predictions with image indices
    dt_records = []  # list of (score, img_idx, box)
    for img_idx, (boxes, scores) in enumerate(zip(all_pred_boxes_per_img, all_pred_scores_per_img)):
        for box, score in zip(boxes, scores):
            dt_records.append((float(score), img_idx, box))

    if len(dt_records) == 0:
        return 0.0, total_gt, 0

    # Sort descending by confidence score
    dt_records.sort(key=lambda x: x[0], reverse=True)

    gt_matched_per_img = [{i: False for i in range(len(gts))} for gts in all_gt_boxes_per_img]

    tp = np.zeros(len(dt_records), dtype=np.float64)
    fp = np.zeros(len(dt_records), dtype=np.float64)

    for i, (_, img_idx, dt_box) in enumerate(dt_records):
        gt_boxes = all_gt_boxes_per_img[img_idx]
        matched_dict = gt_matched_per_img[img_idx]

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = compute_iou(dt_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thresh and best_gt_idx >= 0 and not matched_dict[best_gt_idx]:
            tp[i] = 1.0
            matched_dict[best_gt_idx] = True
        else:
            fp[i] = 1.0

    # Cumulative TP and FP
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / float(total_gt)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

    # VOC 11-point or continuous PR curve integration (all points)
    # Continuous integration:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for j in range(len(mpre) - 2, -1, -1):
        mpre[j] = max(mpre[j], mpre[j + 1])

    i_indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i_indices + 1] - mrec[i_indices]) * mpre[i_indices + 1]))

    return ap, total_gt, len(dt_records)


def evaluate_detection_mAP(
    ground_truths: list[list[dict]],
    predictions: list[list[dict]],
    class_ids: Sequence[int] = (1, 2),
    iou_thresh: float = 0.5,
) -> DetectionEvalResult:
    """Evaluates multi-class detection mAP@0.5 on dataset.

    Args:
        ground_truths: Per-image list of ground truth dicts: {'box': [x1,y1,x2,y2], 'category_id': int}.
        predictions: Per-image list of predicted dicts: {'box': [x1,y1,x2,y2], 'category_id': int, 'score': float}.
        class_ids: Classes to evaluate (e.g. 1: Active TB, 2: Latent TB).
        iou_thresh: IoU overlap threshold.

    Returns:
        DetectionEvalResult with per-class AP and mAP@0.5.
    """
    ap_per_class = {}
    gt_counts = {}
    dt_counts = {}

    for cls_id in class_ids:
        cls_gts = []
        cls_dt_boxes = []
        cls_dt_scores = []

        for img_gts, img_dts in zip(ground_truths, predictions):
            img_cls_gts = [g["box"] for g in img_gts if g["category_id"] == cls_id]
            img_cls_dts = [d["box"] for d in img_dts if d["category_id"] == cls_id]
            img_cls_scores = [d["score"] for d in img_dts if d["category_id"] == cls_id]

            cls_gts.append(img_cls_gts)
            cls_dt_boxes.append(img_cls_dts)
            cls_dt_scores.append(img_cls_scores)

        ap, n_gt, n_dt = compute_ap_50_for_class(
            cls_gts, cls_dt_boxes, cls_dt_scores, iou_thresh=iou_thresh
        )
        ap_per_class[cls_id] = ap
        gt_counts[cls_id] = n_gt
        dt_counts[cls_id] = n_dt

    map_50 = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    return DetectionEvalResult(
        ap_per_class=ap_per_class,
        map_50=map_50,
        total_gt_per_class=gt_counts,
        total_dt_per_class=dt_counts,
    )


@dataclass
class ImageClassificationMetrics:
    """Image-level classification evaluation summary."""

    accuracy: float
    confusion_matrix: np.ndarray
    binary_sensitivity: float
    binary_specificity: float
    binary_auc: float
    threshold: float


def compute_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Computes Area Under ROC Curve using trapezoidal rule."""
    if len(np.unique(labels)) < 2:
        return 0.5

    # Sort descending
    desc_idx = np.argsort(scores)[::-1]
    y_true = labels[desc_idx]
    y_score = scores[desc_idx]

    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_indices = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_indices]
    fps = (1 + threshold_indices) - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] == 0 or tps[-1] == 0:
        return 0.5

    fpr = fps / fps[-1]
    tpr = tps / tps[-1]

    # Trapezoidal rule for ROC AUC
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) * 0.5))
    return auc


def evaluate_image_level_classification(
    true_labels: Sequence[int],
    predicted_active_scores: Sequence[float],
    predicted_latent_scores: Sequence[float],
    confidence_threshold: float = 0.001,
) -> ImageClassificationMetrics:
    """Evaluates 3-class and binary image-level TB classification metrics.

    Args:
        true_labels: True image labels (0: Normal/Sick non-TB, 1: Active TB, 2: Latent TB).
        predicted_active_scores: Maximum active TB lesion score per image.
        predicted_latent_scores: Maximum latent TB lesion score per image.
        confidence_threshold: Threshold to consider a prediction positive.

    Returns:
        ImageClassificationMetrics instance.
    """
    y_true = np.array(true_labels, dtype=np.int32)
    act_scores = np.array(predicted_active_scores, dtype=np.float32)
    lat_scores = np.array(predicted_latent_scores, dtype=np.float32)

    n_samples = len(y_true)
    y_pred = np.zeros(n_samples, dtype=np.int32)

    for i in range(n_samples):
        s_act = act_scores[i]
        s_lat = lat_scores[i]

        if s_act >= confidence_threshold or s_lat >= confidence_threshold:
            if s_act >= s_lat:
                y_pred[i] = 1
            else:
                y_pred[i] = 2
        else:
            y_pred[i] = 0

    # Map labels to [0, 1, 2] where any label > 2 is treated as non-TB 0
    y_true_3class = np.where(y_true > 2, 0, y_true)
    accuracy = float(np.mean(y_true_3class == y_pred))

    # Confusion matrix 3x3
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true_3class, y_pred):
        cm[t, p] += 1

    # Binary TB metrics: True TB is (1 or 2), Non-TB is 0
    binary_true = (y_true_3class > 0).astype(np.int32)
    binary_scores = np.maximum(act_scores, lat_scores)
    binary_pred = (binary_scores >= confidence_threshold).astype(np.int32)

    tp = np.sum((binary_true == 1) & (binary_pred == 1))
    fn = np.sum((binary_true == 1) & (binary_pred == 0))
    tn = np.sum((binary_true == 0) & (binary_pred == 0))
    fp = np.sum((binary_true == 0) & (binary_pred == 1))

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    auc = compute_roc_auc(binary_true, binary_scores)

    return ImageClassificationMetrics(
        accuracy=accuracy,
        confusion_matrix=cm,
        binary_sensitivity=sensitivity,
        binary_specificity=specificity,
        binary_auc=auc,
        threshold=confidence_threshold,
    )
