# CXR-RAIT Zero-Shot Evaluation Plan: TBX11K Faster R-CNN in Flax NNX

**Date:** 2026-09-02  
**Framework:** JAX / Flax NNX  
**Source Domain:** TBX11K (Active TB, Latent TB, Healthy/Sick Non-TB)  
**Target Domain:** CXR-RAIT (`gs://cxr-rait/cxr-demography-data/`)  
**Evaluation Scope:** Strictly Binary Tuberculosis (TB) vs. Non-Tuberculosis (Non-TB)

---

## 1. Executive Summary

This experiment evaluates the zero-shot domain-transfer capability of a frozen Faster R-CNN (ResNet-50-FPN-V2) detection model, pre-trained on TBX11K, when applied directly to chest radiographs from the CXR-RAIT target pool.

Unlike downstream supervised fine-tuning or domain-adaptation training, this zero-shot protocol assesses the out-of-the-box transferability of features and lesion detectors learned on source radiographs without updating any model weights on the target domain.

---

## 2. Evaluation Scope & Strict Binary Target Definition

In accordance with clinical evaluation standards and dataset annotations:
- **Tuberculosis Positive (Class 1):** Verified TB cases (e.g., active or latent tuberculosis).
- **Non-Tuberculosis Negative (Class 0):** Verified healthy or non-TB individuals.
- **Unlabeled Samples:** Included for inference-only predictions, visual galleries, and unlabelled transfer auditing; explicitly excluded from supervised performance metrics.
- **Excluded / Indeterminate / Other Unknown Labels:** Non-binary, indeterminate, or unverified categories are strictly marked as `excluded` and never collapsed into either positive or negative classes.

---

## 3. Preprocessing & Reversible Coordinate Geometry

Deterministic CXR preprocessing for Faster R-CNN (`cxr-rait-detection-v1`):
1. **Rescale Slope & Intercept:** `pixels = raw * RescaleSlope + RescaleIntercept`.
2. **Photometric Inversion:** `MONOCHROME1` images are inverted so bone/lesion density is consistently bright.
3. **Intensity Normalization:** Min-max intensity scaling to `[0.0, 1.0]`.
4. **Aspect-Preserving Resize & Padding:** Scaled to fit within `512x512` canvas with centered zero-padding.
5. **Grayscale to RGB:** Replicated across 3 channels.
6. **ImageNet Normalization:** `(RGB / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]`.
7. **Reversible Coordinate Transforms:** Exact forward and inverse box coordinate transformations (`orig_to_canvas_box` and `canvas_to_orig_box`) preserve original pixel space alignment.

---

## 4. Evaluation CLI & Commands

The evaluation entry point is `scripts/evaluate_faster_rcnn_cxr_rait.py`.

### Dry-Run / Manifest Audit
Audits target pool counts, label availability, sites, and file accessibility without model inference:
```bash
conda run -n med-jax env PYTHONPATH=src \
  python scripts/evaluate_faster_rcnn_cxr_rait.py \
  --manifest artifacts/cxr_rait/split_manifest_seed7.json \
  --split all \
  --dry-run
```

### Full Target Pool Zero-Shot Evaluation
Evaluates all available CXR-RAIT samples using the registered default threshold:
```bash
conda run -n med-jax env PYTHONPATH=src \
  python scripts/evaluate_faster_rcnn_cxr_rait.py \
  --manifest artifacts/cxr_rait/split_manifest_seed7.json \
  --split all \
  --threshold 0.001 \
  --threshold-source registered_default \
  --output-dir artifacts/cxr_rait_zero_shot \
  --bootstrap-samples 1000 \
  --save-galleries 5
```

### Development-Tuned Threshold Evaluation (Locked Mode)
Tunes the operating threshold on the `validation` split (e.g., maximizing F1) and evaluates the locked `test` split:
```bash
conda run -n med-jax env PYTHONPATH=src \
  python scripts/evaluate_faster_rcnn_cxr_rait.py \
  --manifest artifacts/cxr_rait/split_manifest_seed7.json \
  --split test \
  --tune-threshold \
  --tune-split validation \
  --tuning-criterion max_f1 \
  --locked \
  --output-dir artifacts/cxr_rait_zero_shot_test
```

*Note: Attempting to tune thresholds directly on `--split test` or `--split all` while locked is prohibited and rejected with an explicit error.*

---

## 5. Generated Artifacts & Schema

Outputs are saved in the `--output-dir` (default: `artifacts/cxr_rait_zero_shot/`):
- `audit.json`: Manifest audit with total records, unique patients, TB+, Non-TB-, unlabeled, excluded, and site counts.
- `predictions.json`: Per-image predictions with patient/study IDs, site, transform details, raw detections, filtered detections, scores, threshold, and aggregated diagnosis.
- `metrics.json`: Supervised binary metrics with 95% patient-level bootstrap confidence intervals (AUROC, AUPRC, sensitivity, specificity, precision, NPV, F1, balanced accuracy, sensitivity at fixed specificities), localization metrics (or explicit unavailability reason), and threshold provenance.
- `galleries/`: Stratified visualization previews:
  - `true_positives/` (GT: TB, Pred: TB)
  - `false_positives/` (GT: Non-TB, Pred: TB)
  - `false_negatives/` (GT: TB, Pred: Non-TB)
  - `true_negatives/` (GT: Non-TB, Pred: Non-TB)
  - `unlabeled/` (Inference on unlabeled target samples)
  - `by_site/` (Stratified previews grouped by clinical collection site)

---

## 6. Distinctions & Limitations

1. **Zero-Shot vs. Fine-Tuning:**
   - Zero-shot evaluation applies the frozen TBX11K detector without adjusting any weights for CXR-RAIT scanner characteristics or patient population shifts.
   - Subsequent fine-tuning experiments (e.g. supervised domain adaptation or semi-supervised transfer) should be benchmarked against this zero-shot baseline.
2. **Localization Annotations:**
   - If CXR-RAIT target records lack ground-truth bounding boxes, lesion detection AP/mAP is reported as `unavailable`. Annotations are never synthesized or fabricated.
3. **Data Integrity:**
   - Missing or unreadable DICOM files are explicitly audited and logged; blank or synthetic surrogate images are never substituted silently.
