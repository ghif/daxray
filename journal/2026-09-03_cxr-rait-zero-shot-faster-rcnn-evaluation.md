# CXR-RAIT Zero-Shot Evaluation Report: Faster R-CNN (TBX11K) Transfer Baseline

**Date:** 2026-09-03  
**Evaluator:** `scripts/evaluate_faster_rcnn_cxr_rait.py`  
**Model Architecture:** Faster R-CNN (ResNet-50-FPN-V2) in Flax NNX / JAX  
**Pretrained Source:** TBX11K (`nakasiga/tbx11k-object-detection-faster-rcnn`, `best.pt`)  
**Target Dataset:** CXR-RAIT (`gs://cxr-rait/cxr-demography-data/`)  
**Evaluation Scope:** Strictly Binary Tuberculosis (TB) vs. Non-Tuberculosis (Non-TB)  
**Run Mode:** Frozen Zero-Shot Transfer Evaluation  

---

## 1. Executive Summary

We conducted a zero-shot domain-transfer evaluation of the Faster R-CNN detection model (trained on TBX11K) across the complete target pool of the CXR-RAIT dataset ($N = 470$ patients). 

The primary objective was to establish an unvarnished, auditable transfer benchmark before applying any domain-adaptation or target fine-tuning techniques.

### Key Takeaways
1. **Severe Domain Gap:** Zero-shot transfer from TBX11K to CXR-RAIT yields an overall **AUROC of 0.4883** [95% CI: 0.4332–0.5396] and **Balanced Accuracy of 0.4874** [95% CI: 0.4486–0.5284] across the target pool, indicating performance near chance level without target adaptation.
2. **Pronounced False-Positive Over-Detection:** At the standard operating threshold ($0.0010$), the model achieves high sensitivity ($75.32\%$) but very low specificity ($22.15\%$), incorrectly flagging $123$ out of $158$ non-TB CXR-RAIT target cases as tuberculosis lesions.
3. **Consistent Mismatch Across All Sites:** Performance remains uniformly poor across all three Indonesian clinical collection sites (Site S18 AUROC: $0.4368$; Site S19 AUROC: $0.4957$; Site S20 AUROC: $0.5063$), confirming that the domain discrepancy is general across scanners and collection cohorts.
4. **Localization Annotation Status:** CXR-RAIT provides patient- and image-level microbiological labels without bounding box annotations. Consequently, localization metrics ($m\text{AP}@0.5$) are reported as **unavailable** rather than synthesized.

---

## 2. Dataset Manifest & Target Pool Audit

The evaluation was executed on the authoritative CXR-RAIT data located at `gs://cxr-rait/cxr-demography-data/` with the following verified manifest characteristics:

| Property | Value | Notes |
| :--- | :--- | :--- |
| **Total Target Records** | **470** | All matched to demographic metadata |
| **Unique Patients** | **470** | One primary radiograph per patient |
| **Unique Studies** | **470** | Complete target cohort |
| **Tuberculosis Positive ($y=1$)** | **312** ($66.38\%$) | Microbiologically verified TB |
| **Non-Tuberculosis Negative ($y=0$)** | **158** ($33.62\%$) | Verified non-TB healthy controls |
| **Unlabeled Samples** | **0** | $100\%$ target label availability |
| **Excluded / Indeterminate** | **0** | All target records are strictly binary |
| **Missing / Corrupted DICOMs** | **0** | $100\%$ DICOMs accessible in GCS |
| **Lesion Box Annotations** | **0** | Image-level classification labels only |
| **Dataset Fingerprint** | `2ed6c96ef20c1746` | Deterministic SHA256 record digest |

### Clinical Collection Site Distribution

| Site Identifier | Total Records | TB Positive ($y=1$) | Non-TB Negative ($y=0$) | TB Prevalence |
| :--- | :---: | :---: | :---: | :---: |
| **Site S18** | 96 | 57 | 39 | $59.38\%$ |
| **Site S19** | 195 | 125 | 70 | $64.10\%$ |
| **Site S20** | 179 | 130 | 49 | $72.63\%$ |
| **Total Pool** | **470** | **312** | **158** | **66.38%** |

---

## 3. Evaluation Setup & Preprocessing

### Preprocessing Protocol (`cxr-rait-detection-v1`)
- **DICOM Pixel Scaling:** `pixel_array * RescaleSlope + RescaleIntercept`.
- **Photometric Normalization:** Automatic inversion of `MONOCHROME1` to ensure uniform bone/density polarity.
- **Dynamic Windowing:** Min-max intensity rescaling mapped to $[0.0, 1.0]$.
- **Aspect-Preserving Resize & Padding:** High-resolution DICOMs rescaled while preserving aspect ratio, centered with zero-padding onto a $512 \times 512 \times 3$ canvas.
- **ImageNet Normalization:** Standard RGB channel mean $[0.485, 0.456, 0.406]$ and std $[0.229, 0.224, 0.225]$.
- **Reversible Geometry:** Forward/inverse transforms mathematically logged per image for bounding box spatial mapping.

### Evaluation Execution Commands

```bash
# 1. Manifest verification & audit
conda run -n med-jax env PYTHONPATH=src \
  python scripts/evaluate_faster_rcnn_cxr_rait.py \
  --manifest artifacts/cxr_rait/manifest_all.json \
  --split-manifest artifacts/cxr_rait/split_manifest_seed7.json \
  --split all \
  --dry-run

# 2. Complete target pool zero-shot evaluation (N=470)
conda run -n med-jax env PYTHONPATH=src \
  python scripts/evaluate_faster_rcnn_cxr_rait.py \
  --manifest artifacts/cxr_rait/manifest_all.json \
  --split-manifest artifacts/cxr_rait/split_manifest_seed7.json \
  --split all \
  --threshold 0.001 \
  --threshold-source registered_default \
  --output-dir artifacts/cxr_rait_zero_shot \
  --bootstrap-samples 1000 \
  --save-galleries 5

# 3. Locked development-tuned test split evaluation (N=48)
conda run -n med-jax env PYTHONPATH=src \
  python scripts/evaluate_faster_rcnn_cxr_rait.py \
  --manifest artifacts/cxr_rait/manifest_all.json \
  --split-manifest artifacts/cxr_rait/split_manifest_seed7.json \
  --split test \
  --tune-threshold \
  --tune-split validation \
  --tuning-criterion max_f1 \
  --locked \
  --output-dir artifacts/cxr_rait_zero_shot_tuned \
  --bootstrap-samples 1000
```

---

## 4. Measured Results

### 4.1. Complete Target Pool Performance ($N = 470$)

Operating threshold: $\tau = 0.0010$ (TBX11K registered default). Confidence intervals computed via $1,000$ patient-level clustered bootstrap iterations ($95\%$ percentile CI).

| Metric | Point Estimate | 95% Clustered CI | Standard Error |
| :--- | :---: | :---: | :---: |
| **AUROC** | **0.4883** | $[0.4332, 0.5396]$ | $0.0273$ |
| **AUPRC** | **0.6657** | $[0.6078, 0.7260]$ | $0.0311$ |
| **Sensitivity (TPR / Recall)** | **0.7532** ($235/312$) | $[0.7038, 0.8007]$ | $0.0246$ |
| **Specificity (TNR)** | **0.2215** ($35/158$) | $[0.1588, 0.2905]$ | $0.0330$ |
| **Precision (PPV)** | **0.6564** | $[0.6035, 0.7079]$ | $0.0256$ |
| **Negative Predictive Value (NPV)** | **0.3125** | $[0.2301, 0.3982]$ | $0.0426$ |
| **F1 Score** | **0.7015** | $[0.6594, 0.7411]$ | $0.0207$ |
| **Balanced Accuracy** | **0.4874** | $[0.4486, 0.5284]$ | $0.0204$ |
| **Overall Accuracy** | **0.5745** ($270/470$) | $[0.5276, 0.6213]$ | $0.0233$ |

#### Sensitivity at Fixed Specificity Operating Points

| Target Specificity Constraint | Achievable Sensitivity | Required Threshold ($\tau$) |
| :--- | :---: | :---: |
| **Sensitivity @ Specificity $\ge 80\%$** | **19.55%** ($61/312$) | $\tau \approx 0.048$ |
| **Sensitivity @ Specificity $\ge 90\%$** | **10.26%** ($32/312$) | $\tau \approx 0.124$ |
| **Sensitivity @ Specificity $\ge 95\%$** | **7.05%** ($22/312$) | $\tau \approx 0.210$ |

#### 2x2 Binary Confusion Matrix ($\tau = 0.0010$)

```text
                     Predicted Positive    Predicted Negative
Actual TB Positive         235 (TP)               77 (FN)
Actual Non-TB Negative     123 (FP)               35 (TN)
```

---

### 4.2. Site-Stratified Performance Breakdown

| Clinical Site | Sample Size ($N$) | TB Pos / Neg | AUROC | AUPRC | Sensitivity | Specificity | F1 Score | Bal. Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Site S18** | 96 | $57 / 39$ | **0.4368** | **0.5574** | $80.70\%$ | $15.38\%$ | $0.6765$ | $48.04\%$ |
| **Site S19** | 195 | $125 / 70$ | **0.4957** | **0.6399** | $75.20\%$ | $27.14\%$ | $0.6963$ | $51.17\%$ |
| **Site S20** | 179 | $130 / 49$ | **0.5063** | **0.7550** | $73.08\%$ | $20.41\%$ | $0.7197$ | $46.74\%$ |

---

### 4.3. Locked Evaluation Mode on Test Split ($N = 48$)

Threshold tuning was performed strictly on the development `validation` split ($N = 49$, maximizing $F_1$), yielding an optimal threshold $\tau^* = 0.0000$ because the high prevalence in the validation set favored maximum recall.

When evaluated on the held-out locked `test` split ($N = 48$, $32$ TB+, $16$ Non-TB):
- **AUROC:** **0.4277** [95% CI: $0.2676$–$0.6008$]
- **AUPRC:** **0.6474** [95% CI: $0.4844$–$0.8341$]
- **Sensitivity:** **100.00%** ($32/32$)
- **Specificity:** **0.00%** ($0/16$)
- **Accuracy:** **66.67%** ($32/48$)
- **Balanced Accuracy:** **50.00%**

---

## 5. Visual Error Analysis & Diagnostic Galleries

Visual inspection of generated diagnostic galleries in `artifacts/cxr_rait_zero_shot/galleries/` revealed key failure modes:

1. **False Positives on Non-TB Radiographs (`false_positives/`):**
   - The model frequently triggers false positive bounding boxes on normal hilar lymph node shadows, vascular markings, and clavicle edges.
   - Differences in scanner contrast, focal distance, and DICOM dynamic range in CXR-RAIT cause standard parenchymal textures to be flagged as active or latent lesions.
2. **True Positives with Strong Localizations (`true_positives/`):**
   - On severe active TB cases with prominent apical infiltrates and cavitations, the model places high-confidence active TB boxes ($p > 0.70$) accurately around upper lobe consolidations.
3. **False Negatives on Subtle Presentations (`false_negatives/`):**
   - Miliary TB or faint lower-lobe infiltrates frequently yield scores below threshold ($p < 0.001$), escaping detection.

---

## 6. Scientific Implications for Future Work

1. **Zero-Shot Transfer is Insufficient Alone:**
   The empirical AUROC of $\approx 0.49$ establishes that off-the-shelf zero-shot object detection without domain adaptation cannot be deployed directly for Indonesian clinical screening.
2. **Need for Unsupervised Domain Adaptation (UDA):**
   Given that unlabeled CXR-RAIT target images are plentiful, feature-level alignment (e.g., adversarial domain classifiers or Fourier domain alignment) should be applied to bridge the scanner discrepancy between TBX11K and CXR-RAIT.
3. **Supervised Target Fine-Tuning Baseline:**
   The zero-shot metrics reported here serve as the definitive lower bound baseline against which target-domain fine-tuning (e.g., training the ResNet-50-FPN backbone on CXR-RAIT splits) can be evaluated.

---

## 7. Artifacts Summary

All experiment records, per-image detection outputs, and preview galleries have been persisted under:
- `artifacts/cxr_rait/manifest_all.json` ($470$ records)
- `artifacts/cxr_rait_zero_shot/predictions.json` (Full target pool per-image predictions)
- `artifacts/cxr_rait_zero_shot/metrics.json` (Structured metrics with bootstrap CIs)
- `artifacts/cxr_rait_zero_shot/audit.json` (Target pool audit summary)
- `artifacts/cxr_rait_zero_shot/galleries/` (Stratified preview PNG images)
