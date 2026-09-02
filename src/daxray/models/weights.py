"""Weight loading and checkpoint conversion for Faster R-CNN in Flax NNX."""

from pathlib import Path
from typing import Optional, Union

import flax.nnx as nnx
import jax.numpy as jnp
import torch

from daxray.models.faster_rcnn import FasterRCNN

HF_REPO_ID = "nakasiga/tbx11k-object-detection-faster-rcnn"
HF_DEFAULT_FILENAME = "best.pt"


def set_eval_mode(module: nnx.Module) -> None:
    """Recursively sets all BatchNorm layers in module to evaluation mode."""
    for _, submod in nnx.iter_modules(module):
        if isinstance(submod, nnx.BatchNorm):
            submod.use_running_average = True


def _load_conv(flax_conv: nnx.Conv, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
    """Transfers PyTorch Conv2d weight (out_c, in_c, h, w) -> Flax Conv kernel (h, w, in_c, out_c)."""
    flax_conv.kernel[...] = jnp.array(weight.detach().cpu().numpy().transpose(2, 3, 1, 0))
    if bias is not None and flax_conv.bias is not None:
        flax_conv.bias[...] = jnp.array(bias.detach().cpu().numpy())


def _load_bn(
    flax_bn: nnx.BatchNorm,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
) -> None:
    """Transfers PyTorch BatchNorm2d affine params and running stats to Flax BatchNorm."""
    flax_bn.scale[...] = jnp.array(weight.detach().cpu().numpy())
    flax_bn.bias[...] = jnp.array(bias.detach().cpu().numpy())
    flax_bn.mean[...] = jnp.array(mean.detach().cpu().numpy())
    flax_bn.var[...] = jnp.array(var.detach().cpu().numpy())


def _load_linear(
    flax_linear: nnx.Linear,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> None:
    """Transfers PyTorch Linear weight (out_f, in_f) -> Flax Linear kernel (in_f, out_f)."""
    flax_linear.kernel[...] = jnp.array(weight.detach().cpu().numpy().T)
    if bias is not None and flax_linear.bias is not None:
        flax_linear.bias[...] = jnp.array(bias.detach().cpu().numpy())


def transfer_state_dict_to_flax(
    flax_model: FasterRCNN,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Transfers Faster R-CNN state dict weights from PyTorch into Flax NNX model.

    Args:
        flax_model: FasterRCNN Flax NNX instance.
        state_dict: Dictionary of PyTorch tensors from torchvision Faster R-CNN V2 checkpoint.
    """
    sd = state_dict

    # 1. Backbone Stem
    _load_conv(flax_model.backbone.body.conv1, sd["backbone.body.conv1.weight"])
    _load_bn(
        flax_model.backbone.body.bn1,
        sd["backbone.body.bn1.weight"],
        sd["backbone.body.bn1.bias"],
        sd["backbone.body.bn1.running_mean"],
        sd["backbone.body.bn1.running_var"],
    )

    # 2. ResNet-50 Layers
    layer_configs = [
        ("layer1", 3, ["layer1_0", "layer1_1", "layer1_2"]),
        ("layer2", 4, ["layer2_0", "layer2_1", "layer2_2", "layer2_3"]),
        ("layer3", 6, ["layer3_0", "layer3_1", "layer3_2", "layer3_3", "layer3_4", "layer3_5"]),
        ("layer4", 3, ["layer4_0", "layer4_1", "layer4_2"]),
    ]

    for layer_name, num_blocks, flax_attr_names in layer_configs:
        for b_idx, flax_attr in enumerate(flax_attr_names):
            prefix = f"backbone.body.{layer_name}.{b_idx}"
            block = getattr(flax_model.backbone.body, flax_attr)
            _load_conv(block.conv1, sd[f"{prefix}.conv1.weight"])
            _load_bn(
                block.bn1,
                sd[f"{prefix}.bn1.weight"],
                sd[f"{prefix}.bn1.bias"],
                sd[f"{prefix}.bn1.running_mean"],
                sd[f"{prefix}.bn1.running_var"],
            )
            _load_conv(block.conv2, sd[f"{prefix}.conv2.weight"])
            _load_bn(
                block.bn2,
                sd[f"{prefix}.bn2.weight"],
                sd[f"{prefix}.bn2.bias"],
                sd[f"{prefix}.bn2.running_mean"],
                sd[f"{prefix}.bn2.running_var"],
            )
            _load_conv(block.conv3, sd[f"{prefix}.conv3.weight"])
            _load_bn(
                block.bn3,
                sd[f"{prefix}.bn3.weight"],
                sd[f"{prefix}.bn3.bias"],
                sd[f"{prefix}.bn3.running_mean"],
                sd[f"{prefix}.bn3.running_var"],
            )
            if f"{prefix}.downsample.0.weight" in sd:
                _load_conv(block.downsample_conv, sd[f"{prefix}.downsample.0.weight"])
                _load_bn(
                    block.downsample_bn,
                    sd[f"{prefix}.downsample.1.weight"],
                    sd[f"{prefix}.downsample.1.bias"],
                    sd[f"{prefix}.downsample.1.running_mean"],
                    sd[f"{prefix}.downsample.1.running_var"],
                )

    # 3. FPN Inner Blocks & Layer Blocks
    for i in range(4):
        in_prefix = f"backbone.fpn.inner_blocks.{i}"
        in_block = getattr(flax_model.backbone.fpn, f"inner_block_{i}")
        _load_conv(in_block.conv, sd[f"{in_prefix}.0.weight"])
        _load_bn(
            in_block.bn,
            sd[f"{in_prefix}.1.weight"],
            sd[f"{in_prefix}.1.bias"],
            sd[f"{in_prefix}.1.running_mean"],
            sd[f"{in_prefix}.1.running_var"],
        )

        out_prefix = f"backbone.fpn.layer_blocks.{i}"
        out_block = getattr(flax_model.backbone.fpn, f"layer_block_{i}")
        _load_conv(out_block.conv, sd[f"{out_prefix}.0.weight"])
        _load_bn(
            out_block.bn,
            sd[f"{out_prefix}.1.weight"],
            sd[f"{out_prefix}.1.bias"],
            sd[f"{out_prefix}.1.running_mean"],
            sd[f"{out_prefix}.1.running_var"],
        )

    # 4. RPN Head
    _load_conv(
        flax_model.rpn.head.conv0,
        sd["rpn.head.conv.0.0.weight"],
        sd["rpn.head.conv.0.0.bias"],
    )
    _load_conv(
        flax_model.rpn.head.conv1,
        sd["rpn.head.conv.1.0.weight"],
        sd["rpn.head.conv.1.0.bias"],
    )
    _load_conv(
        flax_model.rpn.head.cls_logits,
        sd["rpn.head.cls_logits.weight"],
        sd["rpn.head.cls_logits.bias"],
    )
    _load_conv(
        flax_model.rpn.head.bbox_pred,
        sd["rpn.head.bbox_pred.weight"],
        sd["rpn.head.bbox_pred.bias"],
    )

    # 5. RoI Box Head
    for i in range(4):
        box_conv = getattr(flax_model.roi_heads.box_head, f"conv{i}")
        box_bn = getattr(flax_model.roi_heads.box_head, f"bn{i}")
        _load_conv(box_conv, sd[f"roi_heads.box_head.{i}.0.weight"])
        _load_bn(
            box_bn,
            sd[f"roi_heads.box_head.{i}.1.weight"],
            sd[f"roi_heads.box_head.{i}.1.bias"],
            sd[f"roi_heads.box_head.{i}.1.running_mean"],
            sd[f"roi_heads.box_head.{i}.1.running_var"],
        )

    _load_linear(
        flax_model.roi_heads.box_head.fc,
        sd["roi_heads.box_head.5.weight"],
        sd["roi_heads.box_head.5.bias"],
    )

    # 6. RoI Box Predictor
    _load_linear(
        flax_model.roi_heads.box_predictor.cls_score,
        sd["roi_heads.box_predictor.cls_score.weight"],
        sd["roi_heads.box_predictor.cls_score.bias"],
    )
    _load_linear(
        flax_model.roi_heads.box_predictor.bbox_pred,
        sd["roi_heads.box_predictor.bbox_pred.weight"],
        sd["roi_heads.box_predictor.bbox_pred.bias"],
    )

    set_eval_mode(flax_model)


def load_pretrained_faster_rcnn(
    checkpoint_path: Optional[Union[str, Path]] = None,
    repo_id: str = HF_REPO_ID,
    filename: str = HF_DEFAULT_FILENAME,
    num_classes: int = 3,
    rngs: Optional[nnx.Rngs] = None,
) -> FasterRCNN:
    """Loads Faster R-CNN model with weights from local checkpoint or Hugging Face.

    Args:
        checkpoint_path: Path to PyTorch checkpoint (.pt). If None, downloads from HF Hub.
        repo_id: Hugging Face repository ID.
        filename: Checkpoint filename in HF repo.
        num_classes: Number of classes (default: 3 for background, active TB, latent TB).
        rngs: Optional random number generator state.

    Returns:
        Configured FasterRCNN Flax NNX model with loaded weights.
    """
    if checkpoint_path is None:
        try:
            import huggingface_hub

            resolved_path = huggingface_hub.hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )
        except Exception as err:
            raise RuntimeError(
                f"Failed to download checkpoint from Hugging Face repository '{repo_id}': {err}"
            ) from err
    else:
        resolved_path = Path(checkpoint_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at path: {resolved_path}")

    ckpt = torch.load(resolved_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = FasterRCNN(num_classes=num_classes, rngs=rngs)
    transfer_state_dict_to_flax(model, state_dict)
    return model
