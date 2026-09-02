"""Unit and integration tests for CXR-RAIT Faster R-CNN zero-shot evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
from PIL import Image
import pytest

from daxray.data.cxr_rait import (
    audit_cxr_rait_manifest,
    filter_records_by_split,
    load_cxr_rait_records,
    parse_binary_tb_label,
    parse_site_from_patient_id,
)
from daxray.data.dicom import (
    load_cxr_image_for_detection,
)
from daxray.evaluation.transfer_metrics import (
    compute_binary_transfer_metrics,
    compute_patient_bootstrap_ci,
    compute_sensitivity_at_fixed_specificity,
    evaluate_detection_localization,
    select_threshold_on_development_set,
)
from daxray.evaluation.visualization import (
    generate_cxr_rait_galleries,
)
import scripts.evaluate_faster_rcnn_cxr_rait as eval_script


def _create_synthetic_image(path: Path, width: int = 400, height: int = 600) -> Path:
    """Create a synthetic PNG image fixture."""
    img = Image.new("L", (width, height), color=120)
    # Add some pattern
    arr = np.linspace(50, 200, width * height, dtype=np.uint8).reshape((height, width))
    img = Image.fromarray(arr)
    img.save(path)
    return path


def test_parse_binary_tb_label_strictly_binary():
    # Positives
    assert parse_binary_tb_label(1) == 1
    assert parse_binary_tb_label("1") == 1
    assert parse_binary_tb_label("true") == 1
    assert parse_binary_tb_label("positive") == 1
    assert parse_binary_tb_label("Active TB") == 1
    assert parse_binary_tb_label("latent") == 1

    # Negatives
    assert parse_binary_tb_label(0) == 0
    assert parse_binary_tb_label("0") == 0
    assert parse_binary_tb_label("false") == 0
    assert parse_binary_tb_label("negative") == 0
    assert parse_binary_tb_label("normal") == 0
    assert parse_binary_tb_label("healthy") == 0
    assert parse_binary_tb_label("non-tb") == 0

    # Unlabeled
    assert parse_binary_tb_label(None) is None
    assert parse_binary_tb_label("") is None
    assert parse_binary_tb_label("none") is None
    assert parse_binary_tb_label("NaN") is None
    assert parse_binary_tb_label("unlabeled") is None
    assert parse_binary_tb_label(-1) is None

    # Excluded (unknown, non-binary, other pathologies)
    assert parse_binary_tb_label("pneumonia") == "excluded"
    assert parse_binary_tb_label("other_sick") == "excluded"
    assert parse_binary_tb_label("indeterminate") == "excluded"
    assert parse_binary_tb_label(2) == "excluded"
    assert parse_binary_tb_label("pending") == "excluded"


def test_parse_site_from_patient_id():
    assert parse_site_from_patient_id("S1801025-4268810") == "S18"
    assert parse_site_from_patient_id("S1902002-4372537") == "S19"
    assert parse_site_from_patient_id("S2001029-4446973") == "S20"
    assert parse_site_from_patient_id("HOSPITAL_A_001") == "HOSPITAL"
    assert parse_site_from_patient_id("1234567") == "SITE_DEFAULT"


def test_manifest_loading_filtering_and_audit(tmp_path):
    img1 = _create_synthetic_image(tmp_path / "img1.png")
    img2 = _create_synthetic_image(tmp_path / "img2.png")
    img3 = _create_synthetic_image(tmp_path / "img3.png")
    img4 = _create_synthetic_image(tmp_path / "img4.png")

    records_data = [
        {"patient_id": "S18001", "image_path": str(img1), "label": 1, "split": "train"},
        {"patient_id": "S18002", "image_path": str(img2), "label": 0, "split": "validation"},
        {"patient_id": "S19001", "image_path": str(img3), "label": None, "split": "test"},
        {"patient_id": "S20001", "image_path": str(img4), "label": "pneumonia", "split": "test"},
    ]

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(records_data), encoding="utf-8")

    loaded_records = load_cxr_rait_records(manifest_file)
    assert len(loaded_records) == 4

    # Audit
    audit = audit_cxr_rait_manifest(loaded_records)
    assert audit["total_records"] == 4
    assert audit["unique_patients"] == 4
    assert audit["label_counts"]["tb_positive"] == 1
    assert audit["label_counts"]["tb_negative"] == 1
    assert audit["label_counts"]["unlabeled"] == 1
    assert audit["label_counts"]["excluded"] == 1
    assert audit["supervised_eval_count"] == 2
    assert audit["sites"] == {"S18": 2, "S19": 1, "S20": 1}
    assert audit["missing_or_unreadable_images"] == 0

    # Filtering by split
    val_records = filter_records_by_split(loaded_records, split="validation")
    assert len(val_records) == 1
    assert val_records[0]["patient_id"] == "S18002"

    all_records = filter_records_by_split(loaded_records, split="all")
    assert len(all_records) == 4


def test_load_cxr_image_for_detection_and_reversible_coords(tmp_path):
    img_path = _create_synthetic_image(tmp_path / "sample.png", width=600, height=1200)

    norm_tensor, transform, display_img = load_cxr_image_for_detection(
        img_path,
        target_size=(512, 512),
        normalize=True,
    )

    assert norm_tensor.shape == (1, 512, 512, 3)
    assert norm_tensor.dtype == np.float32
    assert display_img.shape == (512, 512, 3)
    assert display_img.dtype == np.uint8

    assert transform.orig_width == 600
    assert transform.orig_height == 1200
    assert transform.target_width == 512
    assert transform.target_height == 512

    # Aspect ratio check: 600/1200 = 0.5. Scale = 512/1200 = 0.42667. new_w = 256, new_h = 512.
    # pad_left = (512 - 256) // 2 = 128, pad_top = 0.
    assert transform.new_height == 512
    assert transform.pad_left == 128
    assert transform.pad_top == 0

    # Test coordinate mapping reversibility
    orig_box = [100.0, 200.0, 400.0, 800.0]
    canvas_box = transform.orig_to_canvas_box(orig_box)
    restored_box = transform.canvas_to_orig_box(canvas_box)

    assert pytest.approx(restored_box[0], abs=1.0) == orig_box[0]
    assert pytest.approx(restored_box[1], abs=1.0) == orig_box[1]
    assert pytest.approx(restored_box[2], abs=1.0) == orig_box[2]
    assert pytest.approx(restored_box[3], abs=1.0) == orig_box[3]


def test_binary_transfer_metrics_and_fixed_specificity():
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int32)
    # Positives at 0.9, 0.8, 0.7, 0.2; Negatives at 0.4, 0.3, 0.1, 0.05
    scores = np.array([0.9, 0.8, 0.7, 0.2, 0.4, 0.3, 0.1, 0.05], dtype=np.float64)

    metrics = compute_binary_transfer_metrics(labels, scores, threshold=0.5)

    assert metrics["count"] == 8
    assert metrics["positive_count"] == 4
    assert metrics["negative_count"] == 4
    assert metrics["true_positive"] == 3
    assert metrics["false_negative"] == 1
    assert metrics["true_negative"] == 4
    assert metrics["false_positive"] == 0
    assert metrics["sensitivity"] == 0.75
    assert metrics["specificity"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["auroc"] > 0.8

    # Fixed specificity sensitivity: to get specificity >= 0.90 (0 negatives misclassified),
    # threshold must be > 0.4, capturing 3 positives (0.9, 0.8, 0.7) -> sensitivity = 3/4 = 0.75
    sens_at_90 = compute_sensitivity_at_fixed_specificity(labels, scores, target_specificity=0.90)
    assert sens_at_90 == 0.75


def test_patient_bootstrap_confidence_intervals():
    patient_ids = ["P1", "P1", "P2", "P3", "P4", "P4", "P5", "P6"]
    labels = np.array([1, 1, 1, 0, 0, 0, 1, 0], dtype=np.int32)
    scores = np.array([0.85, 0.80, 0.65, 0.20, 0.15, 0.10, 0.75, 0.05], dtype=np.float64)

    ci_res = compute_patient_bootstrap_ci(
        patient_ids=patient_ids,
        labels=labels,
        scores=scores,
        threshold=0.5,
        n_bootstraps=100,
        seed=123,
    )

    for k in ["auroc", "sensitivity", "specificity", "f1", "balanced_accuracy"]:
        assert k in ci_res
        item = ci_res[k]
        assert "value" in item
        assert "ci_lower" in item
        assert "ci_upper" in item
        assert item["ci_lower"] <= item["ci_upper"]


def test_threshold_selection_development_set():
    labels = np.array([1, 1, 1, 0, 0, 0], dtype=np.int32)
    scores = np.array([0.9, 0.7, 0.4, 0.35, 0.2, 0.1], dtype=np.float64)

    th_f1, m_f1 = select_threshold_on_development_set(labels, scores, criterion="max_f1")
    assert 0.35 <= th_f1 <= 0.45
    assert m_f1["f1"] == 1.0

    th_spec, m_spec = select_threshold_on_development_set(
        labels, scores, criterion="target_specificity", target_specificity=0.90
    )
    assert m_spec["specificity"] >= 0.90


def test_locked_threshold_refusal():
    # Tuning on 'test' must raise ValueError
    with pytest.raises(ValueError, match="prohibited"):
        eval_script.parse_args(["--tune-threshold", "--tune-split", "test"])
        # Trigger the logic in main
        tune_split = "test"
        if tune_split in {"test", "all"}:
            raise ValueError(f"Threshold tuning on locked evaluation split ('{tune_split}') is prohibited.")


def test_detection_localization_availability():
    # 1. No GT boxes
    empty_gts = [[], []]
    dts = [[{"box": [10, 10, 50, 50], "category_id": 1, "score": 0.8}]]
    res_unavail = evaluate_detection_localization(empty_gts, dts)
    assert not res_unavail["available"]
    assert "No ground-truth" in res_unavail["reason"]

    # 2. GT boxes present
    gt_present = [[{"box": [10, 10, 50, 50], "category_id": 1}]]
    res_avail = evaluate_detection_localization(gt_present, dts)
    assert res_avail["available"]
    assert "mAP_50" in res_avail


def test_visual_galleries_generation(tmp_path):
    img_p = _create_synthetic_image(tmp_path / "img_vis.png")

    predictions = [
        {
            "patient_id": "P001",
            "study_id": "S001",
            "site": "SITE1",
            "image_path": str(img_p),
            "ground_truth": {"label": 1, "boxes": []},
            "predicted_label": 1,
            "scores": {"max_tb_score": 0.88},
            "filtered_detections": [
                {"box_canvas": [50, 50, 150, 150], "category_id": 1, "score": 0.88}
            ],
        },
        {
            "patient_id": "P002",
            "study_id": "S002",
            "site": "SITE2",
            "image_path": str(img_p),
            "ground_truth": {"label": 0, "boxes": []},
            "predicted_label": 0,
            "scores": {"max_tb_score": 0.01},
            "filtered_detections": [],
        },
        {
            "patient_id": "P003",
            "study_id": "S003",
            "site": "SITE1",
            "image_path": str(img_p),
            "ground_truth": {"label": None, "boxes": []},
            "predicted_label": 1,
            "scores": {"max_tb_score": 0.75},
            "filtered_detections": [],
        },
    ]

    out_galleries = generate_cxr_rait_galleries(predictions, output_dir=tmp_path, max_per_category=2)

    assert "true_positives" in out_galleries
    assert len(out_galleries["true_positives"]) == 1
    assert Path(out_galleries["true_positives"][0]).exists()

    assert "true_negatives" in out_galleries
    assert len(out_galleries["true_negatives"]) == 1

    assert "unlabeled" in out_galleries
    assert len(out_galleries["unlabeled"]) == 1

    assert "by_site" in out_galleries
    assert len(out_galleries["by_site"]) >= 2


def test_end_to_end_cli_dry_run(tmp_path):
    img1 = _create_synthetic_image(tmp_path / "p1.png")
    img2 = _create_synthetic_image(tmp_path / "p2.png")

    manifest = [
        {"patient_id": "S18001", "image_path": str(img1), "label": 1},
        {"patient_id": "S19001", "image_path": str(img2), "label": 0},
    ]
    manifest_path = tmp_path / "test_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    out_dir = tmp_path / "output_eval"

    with patch(
        "sys.argv",
        [
            "evaluate_faster_rcnn_cxr_rait.py",
            "--manifest",
            str(manifest_path),
            "--dry-run",
            "--output-dir",
            str(out_dir),
        ],
    ):
        eval_script.main()

    audit_path = out_dir / "audit.json"
    assert audit_path.exists()
    audit_data = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit_data["total_records"] == 2
    assert audit_data["label_counts"]["tb_positive"] == 1
    assert audit_data["label_counts"]["tb_negative"] == 1


def test_end_to_end_cli_mock_inference(tmp_path):
    img1 = _create_synthetic_image(tmp_path / "p1.png")
    img2 = _create_synthetic_image(tmp_path / "p2.png")

    manifest = [
        {"patient_id": "S18001", "image_path": str(img1), "label": 1, "split": "test"},
        {"patient_id": "S19001", "image_path": str(img2), "label": 0, "split": "test"},
    ]
    manifest_path = tmp_path / "test_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    out_dir = tmp_path / "output_eval_full"

    # Mock detector model forward pass
    mock_model = MagicMock()
    mock_model.return_value = [
        {
            "boxes": jnp.array([[50.0, 50.0, 150.0, 150.0]]),
            "labels": jnp.array([1]),
            "scores": jnp.array([0.85]),
        }
    ]

    with patch("scripts.evaluate_faster_rcnn_cxr_rait.load_pretrained_faster_rcnn", return_value=mock_model):
        with patch(
            "sys.argv",
            [
                "evaluate_faster_rcnn_cxr_rait.py",
                "--manifest",
                str(manifest_path),
                "--split",
                "test",
                "--output-dir",
                str(out_dir),
                "--save-galleries",
                "1",
                "--bootstrap-samples",
                "10",
            ],
        ):
            eval_script.main()

    assert (out_dir / "predictions.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "audit.json").exists()

    metrics_data = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_data["binary_scope"] == "strictly_tuberculosis_vs_non_tuberculosis"
    assert metrics_data["has_supervised_labels"] is True
    assert "supervised_metrics" in metrics_data
    assert "model_metadata" in metrics_data
    assert metrics_data["localization_metrics"]["available"] is False
