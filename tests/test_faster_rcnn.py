"""Unit and parity tests for Faster R-CNN in Flax NNX."""

import flax.nnx as nnx
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from daxray.models.faster_rcnn import FasterRCNN, FastRCNNConvFCHead, FastRCNNPredictor
from daxray.models.resnet_fpn import BackboneWithFPN
from daxray.models.roi_align import multi_scale_roi_align, roi_align_single_feature_map
from daxray.models.rpn import (
    AnchorGenerator,
    clip_boxes_to_image,
    decode_box_deltas,
    nms_cpu,
)
from daxray.models.weights import set_eval_mode, transfer_state_dict_to_flax


def test_resnet50_and_fpn_shapes():
    rngs = nnx.Rngs(0)
    backbone = BackboneWithFPN(rngs=rngs)
    x = jnp.zeros((1, 512, 512, 3), dtype=jnp.float32)

    features = backbone(x)
    assert "0" in features
    assert "1" in features
    assert "2" in features
    assert "3" in features
    assert "pool" in features

    assert features["0"].shape == (1, 128, 128, 256)
    assert features["1"].shape == (1, 64, 64, 256)
    assert features["2"].shape == (1, 32, 32, 256)
    assert features["3"].shape == (1, 16, 16, 256)
    assert features["pool"].shape == (1, 8, 8, 256)


def test_anchor_generator():
    ag = AnchorGenerator()
    feat_maps = [
        jnp.zeros((1, 128, 128, 256)),
        jnp.zeros((1, 64, 64, 256)),
        jnp.zeros((1, 32, 32, 256)),
        jnp.zeros((1, 16, 16, 256)),
        jnp.zeros((1, 8, 8, 256)),
    ]
    anchors = ag(feat_maps, image_shape=(512, 512))
    assert len(anchors) == 5

    # Total anchors: (128*128 + 64*64 + 32*32 + 16*16 + 8*8) * 3 = 65472
    total_count = sum(a.shape[0] for a in anchors)
    assert total_count == 65472
    assert anchors[0].shape == (128 * 128 * 3, 4)


def test_box_operations():
    # Test clipping
    boxes = jnp.array([[-10.0, -5.0, 520.0, 530.0], [10.0, 20.0, 100.0, 200.0]])
    clipped = clip_boxes_to_image(boxes, (512, 512))
    assert clipped[0, 0] == 0.0
    assert clipped[0, 1] == 0.0
    assert clipped[0, 2] == 512.0
    assert clipped[0, 3] == 512.0
    assert clipped[1, 0] == 10.0

    # Test decode deltas
    anchors = jnp.array([[10.0, 10.0, 50.0, 50.0]])
    deltas = jnp.array([[0.0, 0.0, 0.0, 0.0]])  # zero delta -> exact anchor
    decoded = decode_box_deltas(deltas, anchors)
    assert np.allclose(np.array(decoded), np.array(anchors), atol=1e-5)

    # Test NMS
    nms_boxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [11.0, 11.0, 50.0, 50.0],  # High overlap with box 0
        [100.0, 100.0, 200.0, 200.0],  # Disjoint
    ])
    scores = np.array([0.9, 0.8, 0.95])
    keep = nms_cpu(nms_boxes, scores, iou_threshold=0.5)
    assert 2 in keep  # highest score
    assert 0 in keep  # kept
    assert 1 not in keep  # suppressed


def test_pure_jax_roi_align():
    feat = jnp.ones((32, 32, 64), dtype=jnp.float32)
    boxes = jnp.array([[0.0, 0.0, 128.0, 128.0]], dtype=jnp.float32)

    pooled = roi_align_single_feature_map(
        feature_map=feat,
        boxes=boxes,
        output_size=(7, 7),
        spatial_scale=0.25,
        sampling_ratio=2,
    )
    assert pooled.shape == (1, 7, 7, 64)
    assert np.allclose(np.array(pooled), 1.0, atol=1e-5)

    # Test multi-scale roi align
    feats = {
        "0": jnp.ones((1, 128, 128, 256)),
        "1": jnp.ones((1, 64, 64, 256)),
        "2": jnp.ones((1, 32, 32, 256)),
        "3": jnp.ones((1, 16, 16, 256)),
    }
    multi_pooled = multi_scale_roi_align(feats, boxes, output_size=(7, 7))
    assert multi_pooled.shape == (1, 7, 7, 256)


def test_fast_rcnn_head_and_predictor():
    rngs = nnx.Rngs(0)
    head = FastRCNNConvFCHead(in_channels=256, fc_dim=1024, resolution=7, rngs=rngs)
    predictor = FastRCNNPredictor(in_features=1024, num_classes=3, rngs=rngs)

    x = jnp.zeros((4, 7, 7, 256), dtype=jnp.float32)
    head_feats = head(x)
    assert head_feats.shape == (4, 1024)

    cls_logits, bbox_pred = predictor(head_feats)
    assert cls_logits.shape == (4, 3)
    assert bbox_pred.shape == (4, 12)


def test_full_faster_rcnn_forward_pass():
    rngs = nnx.Rngs(0)
    model = FasterRCNN(num_classes=3, rngs=rngs)
    set_eval_mode(model)

    dummy_input = jnp.zeros((1, 512, 512, 3), dtype=jnp.float32)
    results = model(dummy_input, score_thresh=0.0)

    assert len(results) == 1
    res = results[0]
    assert "boxes" in res
    assert "labels" in res
    assert "scores" in res
    assert res["boxes"].ndim == 2
    assert res["boxes"].shape[-1] == 4


def test_pretrained_weights_parity_with_torch():
    """Verifies that loaded Flax NNX model matches PyTorch torchvision model predictions."""
    try:
        import huggingface_hub

        ckpt_path = huggingface_hub.hf_hub_download(
            repo_id="nakasiga/tbx11k-object-detection-faster-rcnn",
            filename="best.pt",
        )
    except Exception as e:
        pytest.skip(f"Hugging Face hub unavailable: {e}")

    # 1. Load PyTorch model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt

    torch_model = fasterrcnn_resnet50_fpn_v2(num_classes=3)
    torch_model.load_state_dict(sd)
    torch_model.eval()

    # 2. Load Flax NNX model
    rngs = nnx.Rngs(0)
    flax_model = FasterRCNN(num_classes=3, rngs=rngs)
    transfer_state_dict_to_flax(flax_model, sd)
    set_eval_mode(flax_model)

    # 3. Backbone feature parity check
    np.random.seed(42)
    dummy_img = np.random.randn(1, 512, 512, 3).astype(np.float32)
    t_img = torch.from_numpy(dummy_img.transpose(0, 3, 1, 2))

    with torch.no_grad():
        t_fpn = torch_model.backbone(t_img)

    f_fpn = flax_model.backbone(jnp.array(dummy_img))

    for k in ["0", "1", "2", "3", "pool"]:
        t_out = t_fpn[k].numpy().transpose(0, 2, 3, 1)
        f_out = np.array(f_fpn[k])
        diff = np.max(np.abs(t_out - f_out))
        assert diff < 1e-4, f"FPN level {k} diff {diff} exceeds threshold"
