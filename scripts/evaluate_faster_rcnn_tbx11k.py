#!/usr/bin/env python3
"""Evaluates Faster R-CNN on TBX11K validation set.

Computes:
1. Lesion detection mAP@0.5 (Active TB, Latent TB, Mean AP@0.5).
2. Image-level 3-class classification accuracy & confusion matrix.
3. Binary TB classification sensitivity, specificity, and ROC AUC across thresholds.
4. Generates visual preview images with ground-truth and predicted boxes.
"""

import argparse
from pathlib import Path
import sys
import time

import jax.numpy as jnp
import numpy as np

from daxray.data.tbx11k import BoundingBox, TBX11KDataset
from daxray.evaluation.detection_metrics import (
    evaluate_detection_mAP,
    evaluate_image_level_classification,
)
from daxray.inference.detector import TBXDetector
from daxray.models.weights import load_pretrained_faster_rcnn


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN on TBX11K")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/Users/mghifary/Work/Code/AI/data/TBX11K",
        help="Path to TBX11K dataset directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="TBX11K_val.txt",
        help="Split list filename in lists/ directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to local PyTorch checkpoint (.pt)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of validation samples to evaluate (default: full split)",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.001,
        help="Box confidence score threshold for detection filtering",
    )
    parser.add_argument(
        "--save-previews",
        type=int,
        default=10,
        help="Number of detection preview images to save in output directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/detection_previews",
        help="Directory to save preview images and evaluation results",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("DAXRay - TBX11K Faster R-CNN Evaluation (Flax NNX / JAX)")
    print("=" * 70)
    print(f"Dataset directory : {args.data_dir}")
    print(f"Validation split  : {args.split}")
    print(f"Checkpoint path   : {args.checkpoint or 'Hugging Face Hub (best.pt)'}")
    print(f"Score threshold   : {args.score_thresh}")

    # 1. Load dataset
    dataset = TBX11KDataset(root_dir=args.data_dir, split_list=args.split)
    total_samples = len(dataset)
    if total_samples == 0:
        print(f"Error: No samples found in split '{args.split}'", file=sys.stderr)
        sys.exit(1)

    eval_count = min(args.max_samples, total_samples) if args.max_samples else total_samples
    print(f"Evaluating {eval_count} of {total_samples} samples...")

    # 2. Load model & detector
    print("Loading Faster R-CNN model weights into Flax NNX...")
    model = load_pretrained_faster_rcnn(checkpoint_path=args.checkpoint)
    detector = TBXDetector(model=model, default_score_thresh=args.score_thresh)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Iterate over samples
    ground_truths = []
    predictions = []
    true_labels = []
    pred_active_scores = []
    pred_latent_scores = []

    saved_preview_count = 0
    t0 = time.time()

    for idx in range(eval_count):
        sample = dataset[idx]
        img_np, gt_boxes = dataset.load_image(sample, target_size=(512, 512), normalize=True)

        # Run forward detection pass
        raw_preds = detector.model(
            jnp.array(img_np[None, ...]),
            score_thresh=args.score_thresh,
        )
        dt = raw_preds[0]
        dt_boxes_np = np.array(dt["boxes"])
        dt_labels_np = np.array(dt["labels"])
        dt_scores_np = np.array(dt["scores"])

        # Format GT and DT for mAP evaluation
        img_gt_dicts = [
            {"box": [b.x1, b.y1, b.x2, b.y2], "category_id": b.category_id}
            for b in gt_boxes
        ]
        img_dt_dicts = [
            {"box": dt_boxes_np[i].tolist(), "category_id": int(dt_labels_np[i]), "score": float(dt_scores_np[i])}
            for i in range(len(dt_boxes_np))
        ]

        ground_truths.append(img_gt_dicts)
        predictions.append(img_dt_dicts)
        true_labels.append(sample.image_label)

        # Image-level max scores per class
        active_scores = [d["score"] for d in img_dt_dicts if d["category_id"] == 1]
        latent_scores = [d["score"] for d in img_dt_dicts if d["category_id"] == 2]
        max_act = max(active_scores) if active_scores else 0.0
        max_lat = max(latent_scores) if latent_scores else 0.0
        pred_active_scores.append(max_act)
        pred_latent_scores.append(max_lat)

        # Save visual preview for sample TB images
        if saved_preview_count < args.save_previews and (len(gt_boxes) > 0 or len(dt_boxes_np) > 0):
            pred_bboxes = [
                BoundingBox(
                    x1=float(dt_boxes_np[i, 0]),
                    y1=float(dt_boxes_np[i, 1]),
                    x2=float(dt_boxes_np[i, 2]),
                    y2=float(dt_boxes_np[i, 3]),
                    category_id=int(dt_labels_np[i]),
                    category_name="Active TB" if int(dt_labels_np[i]) == 1 else "Latent TB",
                    confidence=float(dt_scores_np[i]),
                )
                for i in range(len(dt_boxes_np))
            ]
            preview_filename = f"val_preview_{sample.image_id}.png"
            detector.visualize(
                image_input=sample.image_path,
                predicted_boxes=pred_bboxes,
                ground_truth_boxes=gt_boxes,
                output_path=out_dir / preview_filename,
            )
            saved_preview_count += 1

        if (idx + 1) % 100 == 0 or (idx + 1) == eval_count:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(f"Processed {idx + 1:5d}/{eval_count} samples ({fps:.1f} imgs/s)")

    # 4. Compute metrics
    print("-" * 70)
    print("Computing Detection and Classification Metrics...")
    det_result = evaluate_detection_mAP(ground_truths, predictions, class_ids=(1, 2), iou_thresh=0.5)

    print("\n--- Lesion Detection Performance (IoU >= 0.5) ---")
    print(f"  Active TB AP@0.5 : {det_result.ap_per_class.get(1, 0.0):.4f} (GT count: {det_result.total_gt_per_class.get(1, 0)}, DT count: {det_result.total_dt_per_class.get(1, 0)})")
    print(f"  Latent TB AP@0.5 : {det_result.ap_per_class.get(2, 0.0):.4f} (GT count: {det_result.total_gt_per_class.get(2, 0)}, DT count: {det_result.total_dt_per_class.get(2, 0)})")
    print(f"  Overall mAP@0.5  : {det_result.map_50:.4f}")

    # Evaluate across confidence thresholds
    print("\n--- Image-Level Classification Across Confidence Thresholds ---")
    print(f"{'Threshold':<12} {'3-Class Acc':<14} {'Sensitivity':<14} {'Specificity':<14} {'ROC AUC':<10}")
    print("-" * 64)

    for th in [0.0005, 0.001, 0.002, 0.005, 0.01]:
        cls_metrics = evaluate_image_level_classification(
            true_labels=true_labels,
            predicted_active_scores=pred_active_scores,
            predicted_latent_scores=pred_latent_scores,
            confidence_threshold=th,
        )
        print(
            f"{th:<12.4f} "
            f"{cls_metrics.accuracy:<14.4f} "
            f"{cls_metrics.binary_sensitivity:<14.4f} "
            f"{cls_metrics.binary_specificity:<14.4f} "
            f"{cls_metrics.binary_auc:<10.4f}"
        )

    print("-" * 70)
    print(f"Saved {saved_preview_count} preview visualizations to: {out_dir}")
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
