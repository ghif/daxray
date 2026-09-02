#!/usr/bin/env python3
"""Zero-shot evaluation of Faster R-CNN (ResNet-50-FPN-V2) on CXR-RAIT target dataset.

Evaluates TBX11K-trained Faster R-CNN on CXR-RAIT with:
1. Strictly binary Tuberculosis vs. Non-Tuberculosis supervised evaluation scope.
2. Complete target pool evaluation via `--split all`, or split-specific evaluation.
3. Deterministic DICOM preprocessing and reversible box coordinate transforms.
4. Patient-level clustered bootstrap 95% confidence intervals (AUROC, AUPRC, sensitivity, specificity, NPV, F1, balanced accuracy).
5. Development-only threshold selection with locked evaluation mode enforcement.
6. Localization metrics reporting when ground-truth boxes exist, or explicit unavailability.
7. Visual preview galleries stratified by label availability, site, and error types.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax.numpy as jnp
import numpy as np

from daxray.data.cxr_rait import (
    audit_cxr_rait_manifest,
    filter_records_by_split,
    load_cxr_rait_records,
)
from daxray.data.dicom import (
    DETECTION_PREPROCESSING_VERSION,
    load_cxr_image_for_detection,
)
from daxray.evaluation.transfer_metrics import (
    evaluate_cxr_rait_transfer,
    select_threshold_on_development_set,
)
from daxray.evaluation.visualization import generate_cxr_rait_galleries
from daxray.inference.detector import TBXDetector
from daxray.models.weights import HF_DEFAULT_FILENAME, HF_REPO_ID, load_pretrained_faster_rcnn


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot CXR-RAIT transfer evaluation for Faster R-CNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="artifacts/cxr_rait/split_manifest_seed7.json",
        help="Path to CXR-RAIT manifest or dataset records (JSON, CSV, TSV)",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="Optional path to split manifest JSON containing train/val/test splits",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="Target split to evaluate: 'all', 'test', 'validation', or 'train'",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Optional base directory prefix for relative image paths",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to local Faster R-CNN checkpoint (.pt). If None, downloads from Hugging Face.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="Operating confidence threshold for binary TB classification",
    )
    parser.add_argument(
        "--threshold-source",
        type=str,
        default="registered_default",
        help="Description of threshold origin ('registered_default', 'cli', 'preset_tbx11k')",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Enable operating threshold tuning on a designated development split",
    )
    parser.add_argument(
        "--tune-split",
        type=str,
        default="validation",
        help="Development split to tune threshold on ('validation' or 'train'). Cannot be 'test' or 'all'.",
    )
    parser.add_argument(
        "--tuning-criterion",
        type=str,
        choices=["max_f1", "max_balanced_accuracy", "target_specificity"],
        default="max_f1",
        help="Objective metric for threshold selection on development split",
    )
    parser.add_argument(
        "--target-specificity",
        type=float,
        default=0.90,
        help="Target specificity required when tuning-criterion is 'target_specificity'",
    )
    parser.add_argument(
        "--locked",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce locked evaluation mode prohibiting threshold tuning on test or target evaluation data",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.001,
        help="Box confidence score threshold for raw detector inference",
    )
    parser.add_argument(
        "--nms-thresh",
        type=float,
        default=0.5,
        help="IoU threshold for Non-Maximum Suppression",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of patient-level bootstrap resamples for 95%% confidence intervals (0 to disable)",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling reproducibility",
    )
    parser.add_argument(
        "--save-galleries",
        type=int,
        default=5,
        help="Number of visual preview images to save per stratification category (0 to disable)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/cxr_rait_zero_shot",
        help="Directory to save machine-readable predictions, metrics, audit, and preview galleries",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of target samples to evaluate (for debugging or fast testing)",
    )
    parser.add_argument(
        "--dry-run",
        "--audit-only",
        action="store_true",
        dest="dry_run",
        help="Perform manifest audit and verification only, without loading model or running inference",
    )
    return parser.parse_args(args)


def run_inference_on_records(
    detector: TBXDetector,
    records: list[dict[str, Any]],
    operating_threshold: float,
    score_thresh: float = 0.001,
    nms_thresh: float = 0.5,
    max_samples: int | None = None,
    log_interval: int = 25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run forward Faster R-CNN inference over records and produce structured predictions.

    Returns:
        Tuple of (successful_predictions, valid_records).
    """
    eval_records = records[:max_samples] if max_samples else records
    predictions: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []

    total = len(eval_records)
    t0 = time.time()

    for idx, r in enumerate(eval_records):
        pid = r["patient_id"]
        study_id = r["study_id"]
        site = r["site"]
        img_path = r["image_path"]

        try:
            norm_tensor, transform_details, _ = load_cxr_image_for_detection(
                img_path,
                target_size=(512, 512),
                normalize=True,
                patient_id=pid,
            )
        except Exception as exc:
            print(f"[Warning] Failed to load image for patient {pid} ({img_path}): {exc}", file=sys.stderr)
            continue

        # Forward pass
        raw_outputs = detector.model(
            jnp.array(norm_tensor),
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
        )
        dt = raw_outputs[0]
        boxes_np = np.array(dt["boxes"])
        labels_np = np.array(dt["labels"])
        scores_np = np.array(dt["scores"])

        raw_detections: list[dict[str, Any]] = []
        filtered_detections: list[dict[str, Any]] = []

        active_scores: list[float] = []
        latent_scores: list[float] = []

        for b, lbl, s in zip(boxes_np, labels_np, scores_np):
            cat_id = int(lbl)
            cat_name = "Active TB" if cat_id == 1 else ("Latent TB" if cat_id == 2 else "Background")
            score_val = float(s)
            canv_box = [round(float(coord), 2) for coord in b]
            orig_box = transform_details.canvas_to_orig_box(canv_box)

            det_item = {
                "box_canvas": canv_box,
                "box_original": orig_box,
                "category_id": cat_id,
                "category_name": cat_name,
                "score": score_val,
            }
            raw_detections.append(det_item)

            if cat_id == 1:
                active_scores.append(score_val)
            elif cat_id == 2:
                latent_scores.append(score_val)

            if score_val >= operating_threshold:
                filtered_detections.append(det_item)

        max_active = max(active_scores) if active_scores else 0.0
        max_latent = max(latent_scores) if latent_scores else 0.0
        max_tb = max(max_active, max_latent)

        pred_label = 1 if max_tb >= operating_threshold else 0
        pred_diag = "TB" if pred_label == 1 else "Non-TB"
        if pred_label == 1:
            pred_sub = "Active TB" if max_active >= max_latent else "Latent TB"
        else:
            pred_sub = "Non-TB"

        pred_record = {
            "patient_id": pid,
            "study_id": study_id,
            "site": site,
            "image_path": img_path,
            "ground_truth": {
                "label": r["label"],
                "raw_label": r.get("raw_label"),
                "is_supervised": r["is_supervised"],
                "is_excluded": r["is_excluded"],
                "boxes": r.get("boxes", []),
            },
            "transform_details": transform_details.as_dict(),
            "raw_detections": raw_detections,
            "filtered_detections": filtered_detections,
            "scores": {
                "max_active_tb_score": max_active,
                "max_latent_tb_score": max_latent,
                "max_tb_score": max_tb,
            },
            "threshold": operating_threshold,
            "predicted_label": pred_label,
            "predicted_diagnosis": pred_diag,
            "predicted_subtype": pred_sub,
        }

        predictions.append(pred_record)
        valid_records.append(r)

        if (idx + 1) % log_interval == 0 or (idx + 1) == total:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed if elapsed > 0 else 0.0
            print(f"Processed {idx + 1:5d}/{total} samples ({fps:.1f} imgs/s)")

    return predictions, valid_records


def print_audit_summary(audit: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("CXR-RAIT TARGET POOL MANIFEST AUDIT")
    print("=" * 70)
    print(f"Total Records           : {audit['total_records']}")
    print(f"Unique Patients         : {audit['unique_patients']}")
    print(f"Unique Studies          : {audit['unique_studies']}")
    print(f"Dataset Fingerprint     : {audit['fingerprint']}")
    print("-" * 70)
    print("Strictly Binary Label Distribution (TB vs Non-TB):")
    lbls = audit["label_counts"]
    print(f"  Tuberculosis Positive : {lbls['tb_positive']:5d}")
    print(f"  Non-Tuberculosis      : {lbls['tb_negative']:5d}")
    print(f"  Unlabeled (Inference) : {lbls['unlabeled']:5d}")
    print(f"  Excluded (Non-binary) : {lbls['excluded']:5d}")
    print(f"  Supervised Target Pool: {audit['supervised_eval_count']:5d}")
    print("-" * 70)
    print("Clinical / Collection Sites:")
    for site, count in sorted(audit["sites"].items()):
        pat_c = audit["site_patient_counts"].get(site, 0)
        print(f"  Site {site:<16}: {count:5d} records ({pat_c:4d} patients)")
    print("-" * 70)
    print(f"Missing / Unreadable Imgs: {audit['missing_or_unreadable_images']}")
    print(f"Bounding Box Annotations : {'Available' if audit['has_bounding_boxes'] else 'Unavailable (Image-level labels only)'}")
    print("=" * 70 + "\n")


def print_metrics_summary(results: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("ZERO-SHOT TRANSFER EVALUATION RESULTS (CXR-RAIT)")
    print("=" * 70)
    prov = results.get("threshold_provenance", {})
    print(f"Operating Threshold     : {prov.get('threshold'):.4f} (Source: {prov.get('source')})")
    print("Supervised Scope        : Strictly Tuberculosis vs. Non-Tuberculosis")

    sup = results.get("supervised_metrics")
    if sup is None or not results.get("has_supervised_labels"):
        print("\nNo verified supervised binary labels found in evaluated split.")
        print("Inference-only predictions exported.")
    else:
        print("\n--- Image-Level Binary TB Classification Performance ---")
        print(f"{'Metric':<22} {'Value':<10} {'95% CI':<24} {'Std Error':<10}")
        print("-" * 68)
        metrics_to_show = [
            ("AUROC", "auroc"),
            ("AUPRC", "auprc"),
            ("Sensitivity (TPR)", "sensitivity"),
            ("Specificity (TNR)", "specificity"),
            ("Precision (PPV)", "precision"),
            ("NPV", "npv"),
            ("F1 Score", "f1"),
            ("Balanced Accuracy", "balanced_accuracy"),
            ("Overall Accuracy", "accuracy"),
        ]
        for name, key in metrics_to_show:
            m_info = sup.get(key)
            if isinstance(m_info, dict):
                val = m_info.get("value", float("nan"))
                low = m_info.get("ci_lower", float("nan"))
                high = m_info.get("ci_upper", float("nan"))
                se = m_info.get("std_error", 0.0)
                val_str = f"{val:.4f}" if not math.isnan(val) else "N/A"
                ci_str = f"[{low:.4f}, {high:.4f}]" if not (math.isnan(low) or math.isnan(high)) else "N/A"
                se_str = f"{se:.4f}"
                print(f"{name:<22} {val_str:<10} {ci_str:<24} {se_str:<10}")
            elif isinstance(m_info, (int, float)):
                val_str = f"{m_info:.4f}" if not math.isnan(m_info) else "N/A"
                print(f"{name:<22} {val_str:<10} {'N/A':<24} {'N/A':<10}")

        print("\n--- Sensitivity at Fixed Specificities ---")
        sens_fixed = sup.get("sensitivity_at_fixed_specificity", {})
        for spec_name, sens_val in sens_fixed.items():
            s_str = f"{sens_val:.4f}" if (sens_val is not None and not math.isnan(sens_val)) else "N/A"
            print(f"  Sensitivity {spec_name:<20}: {s_str}")

        print("\n--- Confusion Matrix (Operating Threshold) ---")
        tp = sup.get("true_positive", {}).get("value", 0) if isinstance(sup.get("true_positive"), dict) else sup.get("true_positive", 0)
        tn = sup.get("true_negative", {}).get("value", 0) if isinstance(sup.get("true_negative"), dict) else sup.get("true_negative", 0)
        fp = sup.get("false_positive", {}).get("value", 0) if isinstance(sup.get("false_positive"), dict) else sup.get("false_positive", 0)
        fn = sup.get("false_negative", {}).get("value", 0) if isinstance(sup.get("false_negative"), dict) else sup.get("false_negative", 0)
        print(f"  TP: {tp:5d}  |  FP: {fp:5d}")
        print(f"  FN: {fn:5d}  |  TN: {tn:5d}")

    # Localization
    loc = results.get("localization_metrics", {})
    print("\n--- Lesion Localization Performance ---")
    if not loc.get("available", False):
        print(f"  Status : UNAVAILABLE ({loc.get('reason', 'No box annotations present')})")
    else:
        print(f"  Status : AVAILABLE (mAP@0.5 = {loc.get('mAP_50', 0.0):.4f})")

    print("=" * 70 + "\n")


def main() -> None:
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DAXRay - Faster R-CNN Zero-Shot Evaluation on CXR-RAIT (Flax NNX / JAX)")
    print("=" * 70)
    print(f"Manifest path       : {args.manifest}")
    print(f"Target split        : {args.split}")
    print(f"Locked evaluation   : {args.locked}")
    print(f"Output directory    : {out_dir}")

    # 1. Load manifest and split manifest
    records = load_cxr_rait_records(manifest_path=args.manifest, base_dir=args.data_dir)
    split_manifest = None
    if args.split_manifest:
        split_manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    elif Path(args.manifest).is_file():
        try:
            raw_json = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
            if isinstance(raw_json, dict) and "splits" in raw_json:
                split_manifest = raw_json
        except Exception:
            pass

    # 2. Audit target pool
    audit = audit_cxr_rait_manifest(records)
    print_audit_summary(audit)

    audit_out_path = out_dir / "audit.json"
    audit_out_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    if args.dry_run:
        print("Dry run / audit-only mode complete. No model inference was executed.")
        return

    # 3. Filter target evaluation records
    eval_records = filter_records_by_split(records, split=args.split, split_manifest=split_manifest)
    print(f"Target evaluation records in split '{args.split}': {len(eval_records)}")
    if not eval_records:
        raise ValueError(f"No records found for split '{args.split}'")

    # 4. Threshold determination & Locked Check
    operating_threshold = args.threshold
    threshold_provenance: dict[str, Any] = {
        "threshold": args.threshold,
        "source": args.threshold_source,
        "tuning_subset": None,
        "locked_evaluation": args.locked,
    }

    # Prohibit threshold tuning on locked evaluation target
    if args.tune_threshold:
        tune_split_norm = args.tune_split.strip().lower()
        if tune_split_norm in {"test", "all"}:
            raise ValueError(
                f"Threshold tuning on locked evaluation split ('{args.tune_split}') is prohibited. "
                "Threshold tuning may only be performed on a development split (e.g., 'validation' or 'train')."
            )
        if args.locked and tune_split_norm == args.split.strip().lower():
            raise ValueError(
                f"Locked evaluation mode: cannot tune threshold on the evaluated target split ('{args.split}'). "
                "Use a separate development split for tuning."
            )

    # 5. Load model & detector
    print("\nLoading Faster R-CNN model weights into Flax NNX...")
    model = load_pretrained_faster_rcnn(checkpoint_path=args.checkpoint)
    detector = TBXDetector(
        model=model,
        default_score_thresh=args.score_thresh,
        default_nms_thresh=args.nms_thresh,
    )

    # 6. Execute threshold tuning on development set if requested
    if args.tune_threshold:
        print(f"\nTuning operating threshold on development split '{args.tune_split}' (criterion: {args.tuning_criterion})...")
        dev_records = filter_records_by_split(records, split=args.tune_split, split_manifest=split_manifest)
        if not dev_records:
            raise ValueError(f"No development records found for split '{args.tune_split}'")

        dev_preds, dev_valid_recs = run_inference_on_records(
            detector=detector,
            records=dev_records,
            operating_threshold=args.threshold,
            score_thresh=args.score_thresh,
            nms_thresh=args.nms_thresh,
        )
        dev_sup_indices = [
            i for i, r in enumerate(dev_valid_recs)
            if r.get("label") in (0, 1) and not r.get("is_excluded", False)
        ]
        if not dev_sup_indices:
            raise ValueError(f"No supervised labels found in development split '{args.tune_split}' for threshold tuning.")

        dev_labels = np.array([dev_valid_recs[i]["label"] for i in dev_sup_indices], dtype=np.int32)
        dev_scores = np.array([dev_preds[i]["scores"]["max_tb_score"] for i in dev_sup_indices], dtype=np.float64)

        best_th, best_dev_metrics = select_threshold_on_development_set(
            labels=dev_labels,
            scores=dev_scores,
            criterion=args.tuning_criterion,
            target_specificity=args.target_specificity,
        )
        operating_threshold = best_th
        threshold_provenance = {
            "threshold": operating_threshold,
            "source": f"development_tuning_{args.tune_split}",
            "tuning_subset": args.tune_split,
            "tuning_criterion": args.tuning_criterion,
            "tuning_sample_count": len(dev_labels),
            "tuning_dev_metrics": best_dev_metrics,
            "locked_evaluation": args.locked,
        }
        print(f"Selected optimal threshold: {operating_threshold:.4f} (F1: {best_dev_metrics['f1']:.4f}, BalAcc: {best_dev_metrics['balanced_accuracy']:.4f})")

    # 7. Run inference on target evaluation records
    print(f"\nRunning Faster R-CNN forward pass on {len(eval_records)} target samples...")
    predictions, valid_records = run_inference_on_records(
        detector=detector,
        records=eval_records,
        operating_threshold=operating_threshold,
        score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh,
        max_samples=args.max_samples,
    )

    # 8. Compute transfer metrics
    print("\nComputing supervised binary transfer metrics and patient bootstrap CIs...")
    eval_results = evaluate_cxr_rait_transfer(
        records=valid_records,
        predictions=predictions,
        threshold=operating_threshold,
        threshold_provenance=threshold_provenance,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )

    # Attach model metadata and audit
    model_metadata = {
        "architecture": "Faster R-CNN (ResNet-50-FPN-V2)",
        "checkpoint": args.checkpoint or f"hf://{HF_REPO_ID}/{HF_DEFAULT_FILENAME}",
        "hf_repo_id": HF_REPO_ID,
        "zero_shot": True,
        "preprocessing_version": DETECTION_PREPROCESSING_VERSION,
        "input_resolution": [512, 512],
        "classes": {0: "Background", 1: "Active TB", 2: "Latent TB"},
    }
    eval_results["model_metadata"] = model_metadata
    eval_results["audit"] = audit
    eval_results["split"] = args.split

    # 9. Generate visual galleries
    if args.save_galleries > 0:
        print(f"\nGenerating visual preview galleries (max {args.save_galleries} per category)...")
        gallery_paths = generate_cxr_rait_galleries(
            predictions=predictions,
            output_dir=out_dir,
            max_per_category=args.save_galleries,
        )
        eval_results["gallery_previews"] = gallery_paths
        print(f"Saved galleries to: {out_dir / 'galleries'}")

    # 10. Write JSON outputs
    preds_path = out_dir / "predictions.json"
    metrics_path = out_dir / "metrics.json"

    preds_path.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(eval_results, indent=2) + "\n", encoding="utf-8")

    # 11. Print metrics summary
    print_metrics_summary(eval_results)
    print(f"Evaluation complete. Machine-readable artifacts saved to: {out_dir}")
    print(f"  - Predictions: {preds_path}")
    print(f"  - Metrics    : {metrics_path}")
    print(f"  - Audit      : {audit_out_path}")


if __name__ == "__main__":
    main()
