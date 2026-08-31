"""Small image-only CXR classifier implemented with Flax NNX."""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx


def compute_dtype_for_precision(precision: str) -> jnp.dtype:
    """Resolve the supported activation dtype for a runtime precision name."""
    normalized = precision.lower()
    if normalized == "fp32":
        return jnp.float32
    if normalized == "bf16":
        return jnp.bfloat16
    raise ValueError("precision must be 'fp32' or 'bf16'.")


class ConvBlock(nnx.Module):
    """Convolutional feature block using LayerNorm instead of BatchNorm."""

    def __init__(self, in_channels: int, out_channels: int, *, dtype: jnp.dtype, rngs: nnx.Rngs):
        self.conv = nnx.Conv(
            in_channels, out_channels, kernel_size=(3, 3), padding="SAME",
            dtype=dtype, param_dtype=jnp.float32, rngs=rngs,
        )
        self.norm = nnx.LayerNorm(
            num_features=out_channels, dtype=dtype, param_dtype=jnp.float32, rngs=rngs,
        )

    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        outputs = self.conv(inputs)
        outputs = self.norm(outputs)
        outputs = nnx.relu(outputs)
        return nnx.avg_pool(outputs, window_shape=(2, 2), strides=(2, 2), padding="SAME")


class CxrSmallCNN(nnx.Module):
    """Compact grayscale CNN for the CXR-RAIT TB baseline.

    Inputs must be NHWC, for example ``(batch, 128, 128, 1)``. The output is a
    one-dimensional logit per image; callers should apply sigmoid for a
    probability.
    """

    def __init__(self, *, rngs: nnx.Rngs, dropout_rate: float = 0.2, dtype: jnp.dtype = jnp.float32):
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")
        self.compute_dtype = dtype
        self.block1 = ConvBlock(1, 16, dtype=dtype, rngs=rngs)
        self.block2 = ConvBlock(16, 32, dtype=dtype, rngs=rngs)
        self.block3 = ConvBlock(32, 64, dtype=dtype, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.classifier = nnx.Linear(64, 1, dtype=dtype, param_dtype=jnp.float32, rngs=rngs)

    def __call__(self, inputs: jnp.ndarray, *, training: bool = False) -> jnp.ndarray:
        inputs = jnp.asarray(inputs, dtype=self.compute_dtype)
        outputs = self.block1(inputs)
        outputs = self.block2(outputs)
        outputs = self.block3(outputs)
        outputs = jnp.mean(outputs, axis=(1, 2))
        outputs = self.dropout(outputs, deterministic=not training)
        return self.classifier(outputs).squeeze(-1)


def binary_cross_entropy_with_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable mean binary cross-entropy."""
    logits = jnp.asarray(logits, dtype=jnp.float32)
    labels = jnp.asarray(labels, dtype=jnp.float32)
    return jnp.mean(jnp.maximum(logits, 0.0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits))))


def model_parameter_count(model: CxrSmallCNN) -> int:
    """Return the number of scalar trainable parameters."""
    state = nnx.to_flat_state(nnx.state(model, nnx.Param))
    return sum(int(value.size) for value in state.leaves)
