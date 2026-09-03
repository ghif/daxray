"""Evaluation metrics and patient-level bootstrap analysis for CXR-RAIT transfer experiments."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence
import numpy as np

from .detection_metrics import evaluate_detection_mAP
from .metrics import _auprc, _auroc


def compute_sensitivity_at_fixed_specificity(
    labels: np.ndarray,
    scores: np.ndarray,
    target_specificity: float = 0.90,
) -> float | None:
    """Compute maximum sensitivity achievable at or above a target specificity."""
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(np.sum(pos_mask))
    n_neg = int(np.sum(neg_mask))

    if n_pos == 0 or n_neg == 0:
        return None

    # Unique sorted candidate thresholds
    thresholds = np.unique(scores)
    thresholds = np.sort(thresholds)[::-1]
    # Add boundary thresholds
    thresholds = np.concatenate(([thresholds[0] + 1e-6], thresholds, [thresholds[-1] - 1e-6]))

    best_sens = None
    for th in thresholds:
        preds = (scores >= th).astype(np.int32)
        tn = int(np.sum((neg_mask) & (preds == 0)))
        spec = tn / n_neg
        if spec >= target_specificity:
            tp = int(np.sum((pos_mask) & (preds == 1)))
            sens = tp / n_pos
            if best_sens is None or sens > best_sens:
                best_sens = float(sens)

    return best_sens


def compute_binary_transfer_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.001,
) -> dict[str, Any]:
    """Compute image-level binary Tuberculosis vs Non-Tuberculosis classification metrics."""
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    if len(labels) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    count = len(labels)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))

    preds = (scores >= threshold).astype(np.int32)
    tp = int(np.sum((labels == 1) & (preds == 1)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))

    sensitivity = tp / positive_count if positive_count > 0 else float("nan")
    specificity = tn / negative_count if negative_count > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 and not math.isnan(sensitivity) else 0.0

    valid_rates = [r for r in [sensitivity, specificity] if not math.isnan(r)]
    balanced_acc = float(np.mean(valid_rates)) if valid_rates else float("nan")
    accuracy = (tp + tn) / count if count > 0 else float("nan")

    auroc = _auroc(labels, scores) if positive_count > 0 and negative_count > 0 else float("nan")
    auprc = _auprc(labels, scores) if positive_count > 0 else float("nan")

    return {
        "count": count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "threshold": float(threshold),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "npv": float(npv),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_acc),
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "sensitivity_at_fixed_specificity": {
            "at_specificity_0.80": compute_sensitivity_at_fixed_specificity(labels, scores, target_specificity=0.80),
            "at_specificity_0.90": compute_sensitivity_at_fixed_specificity(labels, scores, target_specificity=0.90),
            "at_specificity_0.95": compute_sensitivity_at_fixed_specificity(labels, scores, target_specificity=0.95),
        },
    }


def select_threshold_on_development_set(
    labels: np.ndarray,
    scores: np.ndarray,
    criterion: str = "max_f1",
    target_specificity: float = 0.90,
) -> tuple[float, dict[str, Any]]:
    """Select the optimal operating threshold on a development set.

    Args:
        labels: 1D array of binary labels (0: Non-TB, 1: TB).
        scores: 1D array of model confidence scores.
        criterion: One of 'max_f1', 'max_balanced_accuracy', or 'target_specificity'.
        target_specificity: Minimum specificity required if criterion is 'target_specificity'.

    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold).
    """
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    if len(labels) == 0:
        raise ValueError("Cannot select threshold on empty development set.")

    unique_scores = np.unique(scores)
    if len(unique_scores) == 0:
        return 0.001, compute_binary_transfer_metrics(labels, scores, threshold=0.001)

    candidate_thresholds = np.unique(
        np.concatenate([
            unique_scores,
            np.percentile(scores, np.linspace(0, 100, 101)),
            [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.5],
        ])
    )
    candidate_thresholds = np.sort(candidate_thresholds)

    best_th = 0.001
    best_score = -float("inf")
    best_metrics = None

    for th in candidate_thresholds:
        m = compute_binary_transfer_metrics(labels, scores, threshold=float(th))

        if criterion == "max_f1":
            val = m["f1"]
        elif criterion == "max_balanced_accuracy":
            val = m["balanced_accuracy"] if not math.isnan(m["balanced_accuracy"]) else -1.0
        elif criterion == "target_specificity":
            spec = m["specificity"]
            sens = m["sensitivity"]
            if not math.isnan(spec) and spec >= target_specificity:
                val = sens if not math.isnan(sens) else 0.0
            else:
                val = -1.0
        else:
            raise ValueError(f"Unknown threshold selection criterion: {criterion}")

        if val > best_score:
            best_score = val
            best_th = float(th)
            best_metrics = m

    if best_metrics is None:
        best_th = 0.001
        best_metrics = compute_binary_transfer_metrics(labels, scores, threshold=best_th)

    return best_th, best_metrics


def compute_patient_bootstrap_ci(
    patient_ids: Sequence[str],
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.001,
    n_bootstraps: int = 1000,
    seed: int = 42,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Compute patient-level clustered bootstrap confidence intervals for binary metrics."""
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    pids = np.asarray(patient_ids)

    point_metrics = compute_binary_transfer_metrics(labels, scores, threshold=threshold)

    if n_bootstraps <= 0:
        ci_summary = {}
        for k, v in point_metrics.items():
            if isinstance(v, (int, float)):
                ci_summary[k] = {
                    "value": v,
                    "ci_lower": v,
                    "ci_upper": v,
                    "bootstrap_samples": 0,
                }
            else:
                ci_summary[k] = v
        return ci_summary

    unique_patients = np.unique(pids)
    n_patients = len(unique_patients)

    # Index map: patient_id -> sample indices
    pid_to_indices = {pid: np.where(pids == pid)[0] for pid in unique_patients}

    rng = np.random.default_rng(seed)
    keys_to_bootstrap = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "npv",
        "f1",
        "auroc",
        "auprc",
    ]

    boot_results: dict[str, list[float]] = {k: [] for k in keys_to_bootstrap}

    for _ in range(n_bootstraps):
        sampled_pids = rng.choice(unique_patients, size=n_patients, replace=True)
        sampled_indices = np.concatenate([pid_to_indices[p] for p in sampled_pids])

        boot_labels = labels[sampled_indices]
        boot_scores = scores[sampled_indices]

        # Only compute if both classes exist
        if np.sum(boot_labels == 1) > 0 and np.sum(boot_labels == 0) > 0:
            bm = compute_binary_transfer_metrics(boot_labels, boot_scores, threshold=threshold)
            for k in keys_to_bootstrap:
                val = bm[k]
                if not math.isnan(val):
                    boot_results[k].append(val)

    alpha = 1.0 - ci_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    ci_summary = {}
    for k, v in point_metrics.items():
        if k in keys_to_bootstrap:
            arr = boot_results.get(k, [])
            if len(arr) >= 10:
                low = float(np.percentile(arr, lower_pct))
                high = float(np.percentile(arr, upper_pct))
                std_err = float(np.std(arr))
            else:
                low = v
                high = v
                std_err = 0.0
            ci_summary[k] = {
                "value": v,
                "ci_lower": low,
                "ci_upper": high,
                "std_error": std_err,
                "bootstrap_samples": len(arr),
            }
        else:
            ci_summary[k] = v

    return ci_summary


def evaluate_detection_localization(
    ground_truths: list[list[dict]],
    predictions: list[list[dict]],
    class_ids: Sequence[int] = (1, 2),
    iou_thresh: float = 0.5,
) -> dict[str, Any]:
    """Compute object detection mAP if ground-truth bounding boxes are present, or mark unavailable."""
    total_gt = sum(len(gts) for gts in ground_truths)
    if total_gt == 0:
        return {
            "available": False,
            "reason": "No ground-truth bounding box annotations present in CXR-RAIT dataset.",
            "mAP_50": None,
            "ap_per_class": {},
            "total_gt_boxes": 0,
            "total_predicted_boxes": sum(len(dts) for dts in predictions),
        }

    det_res = evaluate_detection_mAP(
        ground_truths=ground_truths,
        predictions=predictions,
        class_ids=class_ids,
        iou_thresh=iou_thresh,
    )

    return {
        "available": True,
        "mAP_50": det_res.map_50,
        "ap_per_class": {str(k): v for k, v in det_res.ap_per_class.items()},
        "total_gt_per_class": {str(k): v for k, v in det_res.total_gt_per_class.items()},
        "total_dt_per_class": {str(k): v for k, v in det_res.total_dt_per_class.items()},
        "total_gt_boxes": sum(det_res.total_gt_per_class.values()),
        "total_predicted_boxes": sum(det_res.total_dt_per_class.values()),
    }


def evaluate_cxr_rait_transfer(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    threshold: float = 0.001,
    threshold_provenance: dict[str, Any] | None = None,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Comprehensive zero-shot transfer evaluation on CXR-RAIT target pool."""
    records_list = list(records)
    preds_list = list(predictions)

    if len(records_list) != len(preds_list):
        raise ValueError("records and predictions lengths must match.")

    # Filter strictly binary supervised subset
    supervised_indices = [
        i for i, r in enumerate(records_list)
        if r.get("label") in (0, 1) and not r.get("is_excluded", False)
    ]

    has_supervised = len(supervised_indices) > 0
    supervised_metrics: dict[str, Any] | None = None

    if has_supervised:
        sup_pids = [records_list[i]["patient_id"] for i in supervised_indices]
        sup_labels = np.array([records_list[i]["label"] for i in supervised_indices], dtype=np.int32)
        sup_scores = np.array([preds_list[i]["scores"]["max_tb_score"] for i in supervised_indices], dtype=np.float64)

        supervised_metrics = compute_patient_bootstrap_ci(
            patient_ids=sup_pids,
            labels=sup_labels,
            scores=sup_scores,
            threshold=threshold,
            n_bootstraps=bootstrap_samples,
            seed=bootstrap_seed,
        )

    # Localization metrics
    all_gts = [r.get("boxes", []) for r in records_list]
    # Format GT boxes for detection metrics
    formatted_gts = []
    for gts in all_gts:
        img_gts = []
        for b in gts:
            if isinstance(b, dict):
                img_gts.append(b)
            elif hasattr(b, "to_list"):
                img_gts.append({"box": b.to_list(), "category_id": b.category_id})
        formatted_gts.append(img_gts)

    formatted_dts = []
    for p in preds_list:
        raw_dets = p.get("raw_detections", [])
        img_dts = [
            {
                "box": d.get("box_canvas") or d.get("box"),
                "category_id": int(d.get("category_id", 1)),
                "score": float(d.get("score", 0.0)),
            }
            for d in raw_dets
        ]
        formatted_dts.append(img_dts)

    localization_metrics = evaluate_detection_localization(formatted_gts, formatted_dts)

    # Subtype 3-class confusion matrix if 3-class ground truth exists
    has_subtype_gts = any(
        r.get("raw_label") in ("active", "active_tb", "latent", "latent_tb") or r.get("subtype") is not None
        for r in records_list
    )
    confusion_3class = None
    if has_subtype_gts:
        # Compute 3-class confusion matrix
        cm = np.zeros((3, 3), dtype=np.int64)
        for r, p in zip(records_list, preds_list):
            gt_lbl = r.get("label")
            if gt_lbl is None or r.get("is_excluded"):
                continue
            pred_sub = p.get("predicted_subtype")
            p_cat = 1 if pred_sub == "Active TB" else (2 if pred_sub == "Latent TB" else 0)
            g_cat = int(gt_lbl)
            cm[g_cat, p_cat] += 1
        confusion_3class = cm.tolist()

    # Per-site binary metrics breakdown
    site_breakdown: dict[str, Any] = {}
    if has_supervised:
        for idx in supervised_indices:
            r = records_list[idx]
            s = str(r.get("site") or "SITE_DEFAULT")
            site_breakdown.setdefault(s, {"labels": [], "scores": [], "patients": []})
            site_breakdown[s]["labels"].append(r["label"])
            site_breakdown[s]["scores"].append(preds_list[idx]["scores"]["max_tb_score"])
            site_breakdown[s]["patients"].append(r["patient_id"])

        site_metrics = {}
        for s, sdata in site_breakdown.items():
            s_labels = np.array(sdata["labels"], dtype=np.int32)
            s_scores = np.array(sdata["scores"], dtype=np.float64)
            s_pids = sdata["patients"]
            if len(s_labels) > 0:
                site_metrics[s] = compute_patient_bootstrap_ci(
                    patient_ids=s_pids,
                    labels=s_labels,
                    scores=s_scores,
                    threshold=threshold,
                    n_bootstraps=min(bootstrap_samples, 200),
                    seed=bootstrap_seed,
                )
    else:
        site_metrics = {}

    return {
        "binary_scope": "strictly_tuberculosis_vs_non_tuberculosis",
        "has_supervised_labels": has_supervised,
        "supervised_metrics": supervised_metrics,
        "localization_metrics": localization_metrics,
        "three_class_confusion_matrix": confusion_3class,
        "site_metrics": site_metrics,
        "threshold_provenance": threshold_provenance or {
            "threshold": threshold,
            "source": "cli_or_default",
            "tuning_subset": None,
            "locked_evaluation": True,
        },
    }
