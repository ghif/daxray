"""Small image-only CXR classifier implemented with Flax NNX."""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx


class ConvBlock(nnx.Module):
    """Convolutional feature block using LayerNorm instead of BatchNorm."""

    def __init__(self, in_channels: int, out_channels: int, *, rngs: nnx.Rngs):
        self.conv = nnx.Conv(in_channels, out_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.norm = nnx.LayerNorm(num_features=out_channels, rngs=rngs)

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

    def __init__(self, *, rngs: nnx.Rngs, dropout_rate: float = 0.2):
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1).")
        self.block1 = ConvBlock(1, 16, rngs=rngs)
        self.block2 = ConvBlock(16, 32, rngs=rngs)
        self.block3 = ConvBlock(32, 64, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.classifier = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, inputs: jnp.ndarray, *, training: bool = False) -> jnp.ndarray:
        outputs = self.block1(inputs)
        outputs = self.block2(outputs)
        outputs = self.block3(outputs)
        outputs = jnp.mean(outputs, axis=(1, 2))
        outputs = self.dropout(outputs, deterministic=not training)
        return self.classifier(outputs).squeeze(-1)


def binary_cross_entropy_with_logits(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable mean binary cross-entropy."""
    return jnp.mean(jnp.maximum(logits, 0.0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits))))


def model_parameter_count(model: CxrSmallCNN) -> int:
    """Return the number of scalar trainable parameters."""
    state = nnx.to_flat_state(nnx.state(model, nnx.Param))
    return sum(int(value.size) for value in state.leaves)

