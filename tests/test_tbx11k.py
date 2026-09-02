"""Unit tests for TBX11K dataset discovery, parsers, metrics, and inference."""

from pathlib import Path
import tempfile

import pytest
from PIL import Image

from daxray.data.tbx11k import BoundingBox, parse_voc_xml
from daxray.evaluation.detection_metrics import (
    compute_iou,
    evaluate_detection_mAP,
    evaluate_image_level_classification,
)
from daxray.inference.detector import TBXDetector


def test_bounding_box_and_iou():
    b1 = BoundingBox(10.0, 10.0, 50.0, 50.0, category_id=1, category_name="Active TB")
    assert b1.width == 40.0
    assert b1.height == 40.0
    assert b1.area == 1600.0

    b1_scaled = b1.scale(2.0, 2.0)
    assert b1_scaled.x1 == 20.0
    assert b1_scaled.area == 6400.0

    # Perfect overlap
    iou_self = compute_iou([10, 10, 50, 50], [10, 10, 50, 50])
    assert pytest.approx(iou_self, abs=1e-5) == 1.0

    # Disjoint
    iou_disjoint = compute_iou([10, 10, 50, 50], [100, 100, 150, 150])
    assert iou_disjoint == 0.0

    # Partial overlap
    iou_half = compute_iou([0, 0, 10, 10], [5, 0, 15, 10])
    # Inter: 5*10 = 50, Union: 100 + 100 - 50 = 150 -> 50/150 = 1/3
    assert pytest.approx(iou_half, abs=1e-5) == 1.0 / 3.0


def test_voc_xml_parser():
    xml_content = """<annotation>
        <filename>test_tb001</filename>
        <size>
            <width>1000</width>
            <height>1000</height>
        </size>
        <object>
            <name>ActiveTuberculosis</name>
            <bndbox>
                <xmin>100</xmin>
                <ymin>150</ymin>
                <xmax>400</xmax>
                <ymax>450</ymax>
            </bndbox>
        </object>
        <object>
            <name>ObsoletePulmonaryTuberculosis</name>
            <bndbox>
                <xmin>500</xmin>
                <ymin>550</ymin>
                <xmax>800</xmax>
                <ymax>850</ymax>
            </bndbox>
        </object>
    </annotation>"""

    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml_content)
        temp_path = f.name

    try:
        boxes = parse_voc_xml(temp_path)
        assert len(boxes) == 2
        assert boxes[0].category_id == 1
        assert boxes[0].category_name == "Active TB"
        assert boxes[0].x1 == 100.0

        assert boxes[1].category_id == 2
        assert boxes[1].category_name == "Latent TB"
        assert boxes[1].x2 == 800.0
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_detection_metrics_calculation():
    # Synthetic ground truths and predictions
    gts = [
        [{"box": [10.0, 10.0, 50.0, 50.0], "category_id": 1}],
        [{"box": [20.0, 20.0, 60.0, 60.0], "category_id": 2}],
    ]
    preds = [
        [{"box": [10.0, 10.0, 50.0, 50.0], "category_id": 1, "score": 0.9}],  # TP
        [{"box": [20.0, 20.0, 60.0, 60.0], "category_id": 2, "score": 0.85}],  # TP
    ]

    res = evaluate_detection_mAP(gts, preds, class_ids=(1, 2), iou_thresh=0.5)
    assert pytest.approx(res.ap_per_class[1], abs=1e-3) == 1.0
    assert pytest.approx(res.ap_per_class[2], abs=1e-3) == 1.0
    assert pytest.approx(res.map_50, abs=1e-3) == 1.0


def test_image_level_classification_metrics():
    true_labels = [1, 2, 0, 0]
    pred_active = [0.9, 0.01, 0.0, 0.0001]
    pred_latent = [0.01, 0.85, 0.0, 0.0]

    metrics = evaluate_image_level_classification(
        true_labels, pred_active, pred_latent, confidence_threshold=0.01
    )

    assert metrics.accuracy == 1.0
    assert metrics.binary_sensitivity == 1.0
    assert metrics.binary_specificity == 1.0
    assert metrics.binary_auc == 1.0


def test_detector_inference_and_visualization():
    detector = TBXDetector(default_score_thresh=0.001)

    # Create dummy synthetic image
    dummy_img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    pred_boxes = detector.predict(dummy_img, rescale_to_original=False)
    assert isinstance(pred_boxes, list)

    # Test visualization export
    with tempfile.NamedTemporaryFile("w", suffix=".png", delete=False) as f:
        out_png = f.name

    try:
        gt_box = BoundingBox(50, 50, 150, 150, category_id=1, category_name="Active TB")
        vis_img = detector.visualize(
            dummy_img,
            predicted_boxes=pred_boxes,
            ground_truth_boxes=[gt_box],
            output_path=out_png,
        )
        assert isinstance(vis_img, Image.Image)
        assert Path(out_png).exists()
        assert Path(out_png).stat().st_size > 0
    finally:
        Path(out_png).unlink(missing_ok=True)
