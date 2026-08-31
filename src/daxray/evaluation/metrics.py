"""Numpy classification metrics for baseline experiments."""

from __future__ import annotations

import numpy as np


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int32)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must be matching 1-D arrays.")
    if len(labels) == 0 or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("labels must be non-empty and binary.")
    predictions = (probabilities >= threshold).astype(np.int32)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    positive_count = tp + fn
    negative_count = tn + fp
    sensitivity = tp / positive_count if positive_count else float("nan")
    specificity = tn / negative_count if negative_count else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = sensitivity if not np.isnan(sensitivity) else 0.0
    metrics: dict[str, float | int] = {
        "count": len(labels),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "auroc": _auroc(labels, probabilities),
        "auprc": _auprc(labels, probabilities),
        "threshold": threshold,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }
    return metrics


def _auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def _auprc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    positive_count = int(np.sum(labels == 1))
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-probabilities, kind="stable")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / positive_count
    previous_recall = np.concatenate(([0.0], recall[:-1]))
    return float(np.sum((recall - previous_recall) * precision))
