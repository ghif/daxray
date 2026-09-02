"""ResNet-50 with Feature Pyramid Network (FPN V2) in Flax NNX."""

from collections.abc import Sequence

import flax.nnx as nnx
import jax


class Bottleneck(nnx.Module):
    """ResNet Bottleneck block."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=mid_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=False,
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(num_features=mid_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)

        self.conv2 = nnx.Conv(
            in_features=mid_channels,
            out_features=mid_channels,
            kernel_size=(3, 3),
            strides=(stride, stride),
            padding=[(1, 1), (1, 1)],
            use_bias=False,
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(num_features=mid_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)

        self.conv3 = nnx.Conv(
            in_features=mid_channels,
            out_features=out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            use_bias=False,
            rngs=rngs,
        )
        self.bn3 = nnx.BatchNorm(num_features=out_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)

        if downsample:
            self.downsample_conv = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                strides=(stride, stride),
                padding="VALID",
                use_bias=False,
                rngs=rngs,
            )
            self.downsample_bn = nnx.BatchNorm(
                num_features=out_channels,
                momentum=0.9,
                epsilon=1e-5,
                rngs=rngs,
            )
        else:
            self.downsample_conv = None
            self.downsample_bn = None

    def __call__(self, x: jax.Array) -> jax.Array:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = nnx.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = nnx.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample_conv is not None and self.downsample_bn is not None:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity)

        out = out + identity
        out = nnx.relu(out)
        return out


class ResNet50(nnx.Module):
    """ResNet-50 backbone extracting features at C2, C3, C4, C5."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            in_features=3,
            out_features=64,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=[(3, 3), (3, 3)],
            use_bias=False,
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(num_features=64, momentum=0.9, epsilon=1e-5, rngs=rngs)

        # Layer 1: 3 bottlenecks (64 -> 256)
        self.layer1_0 = Bottleneck(64, 64, 256, stride=1, downsample=True, rngs=rngs)
        self.layer1_1 = Bottleneck(256, 64, 256, stride=1, downsample=False, rngs=rngs)
        self.layer1_2 = Bottleneck(256, 64, 256, stride=1, downsample=False, rngs=rngs)

        # Layer 2: 4 bottlenecks (256 -> 512)
        self.layer2_0 = Bottleneck(256, 128, 512, stride=2, downsample=True, rngs=rngs)
        self.layer2_1 = Bottleneck(512, 128, 512, stride=1, downsample=False, rngs=rngs)
        self.layer2_2 = Bottleneck(512, 128, 512, stride=1, downsample=False, rngs=rngs)
        self.layer2_3 = Bottleneck(512, 128, 512, stride=1, downsample=False, rngs=rngs)

        # Layer 3: 6 bottlenecks (512 -> 1024)
        self.layer3_0 = Bottleneck(512, 256, 1024, stride=2, downsample=True, rngs=rngs)
        self.layer3_1 = Bottleneck(1024, 256, 1024, stride=1, downsample=False, rngs=rngs)
        self.layer3_2 = Bottleneck(1024, 256, 1024, stride=1, downsample=False, rngs=rngs)
        self.layer3_3 = Bottleneck(1024, 256, 1024, stride=1, downsample=False, rngs=rngs)
        self.layer3_4 = Bottleneck(1024, 256, 1024, stride=1, downsample=False, rngs=rngs)
        self.layer3_5 = Bottleneck(1024, 256, 1024, stride=1, downsample=False, rngs=rngs)

        # Layer 4: 3 bottlenecks (1024 -> 2048)
        self.layer4_0 = Bottleneck(1024, 512, 2048, stride=2, downsample=True, rngs=rngs)
        self.layer4_1 = Bottleneck(2048, 512, 2048, stride=1, downsample=False, rngs=rngs)
        self.layer4_2 = Bottleneck(2048, 512, 2048, stride=1, downsample=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Runs ResNet-50 stem and stages.

        Args:
            x: Input images of shape (B, H, W, 3).

        Returns:
            Tuple of feature maps (C2, C3, C4, C5).
        """
        out = self.conv1(x)
        out = self.bn1(out)
        out = nnx.relu(out)
        out = nnx.max_pool(out, window_shape=(3, 3), strides=(2, 2), padding=[(1, 1), (1, 1)])

        # Layer 1
        out = self.layer1_0(out)
        out = self.layer1_1(out)
        c2 = self.layer1_2(out)

        # Layer 2
        out = self.layer2_0(c2)
        out = self.layer2_1(out)
        out = self.layer2_2(out)
        c3 = self.layer2_3(out)

        # Layer 3
        out = self.layer3_0(c3)
        out = self.layer3_1(out)
        out = self.layer3_2(out)
        out = self.layer3_3(out)
        out = self.layer3_4(out)
        c4 = self.layer3_5(out)

        # Layer 4
        out = self.layer4_0(c4)
        out = self.layer4_1(out)
        c5 = self.layer4_2(out)

        return c2, c3, c4, c5


class ConvNorm(nnx.Module):
    """1x1 or 3x3 Conv + BatchNorm block for FPN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (1, 1),
        padding: str | Sequence[tuple[int, int]] = "VALID",
        *,
        rngs: nnx.Rngs,
    ):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=kernel_size,
            strides=(1, 1),
            padding=padding,
            use_bias=False,
            rngs=rngs,
        )
        self.bn = nnx.BatchNorm(num_features=out_channels, momentum=0.9, epsilon=1e-5, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        out = self.conv(x)
        out = self.bn(out)
        return out


class FeaturePyramidNetwork(nnx.Module):
    """FPN V2 with BatchNorm on lateral and output layers, plus P6 maxpool."""

    def __init__(
        self,
        in_channels_list: Sequence[int] = (256, 512, 1024, 2048),
        out_channels: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        # Lateral 1x1 convs (inner_blocks)
        self.inner_block_0 = ConvNorm(in_channels_list[0], out_channels, kernel_size=(1, 1), padding="VALID", rngs=rngs)
        self.inner_block_1 = ConvNorm(in_channels_list[1], out_channels, kernel_size=(1, 1), padding="VALID", rngs=rngs)
        self.inner_block_2 = ConvNorm(in_channels_list[2], out_channels, kernel_size=(1, 1), padding="VALID", rngs=rngs)
        self.inner_block_3 = ConvNorm(in_channels_list[3], out_channels, kernel_size=(1, 1), padding="VALID", rngs=rngs)

        # Output 3x3 convs (layer_blocks)
        self.layer_block_0 = ConvNorm(out_channels, out_channels, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], rngs=rngs)
        self.layer_block_1 = ConvNorm(out_channels, out_channels, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], rngs=rngs)
        self.layer_block_2 = ConvNorm(out_channels, out_channels, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], rngs=rngs)
        self.layer_block_3 = ConvNorm(out_channels, out_channels, kernel_size=(3, 3), padding=[(1, 1), (1, 1)], rngs=rngs)

    def __call__(
        self,
        c2: jax.Array,
        c3: jax.Array,
        c4: jax.Array,
        c5: jax.Array,
    ) -> dict[str, jax.Array]:
        """Constructs pyramid features P2, P3, P4, P5, pool (P6)."""
        # Top level C5 -> inner3
        inner3 = self.inner_block_3(c5)
        p5 = self.layer_block_3(inner3)

        # C4 + upsampled inner3 -> inner2
        inner3_up = jax.image.resize(
            inner3,
            shape=(inner3.shape[0], c4.shape[1], c4.shape[2], inner3.shape[3]),
            method="nearest",
        )
        inner2 = self.inner_block_2(c4) + inner3_up
        p4 = self.layer_block_2(inner2)

        # C3 + upsampled inner2 -> inner1
        inner2_up = jax.image.resize(
            inner2,
            shape=(inner2.shape[0], c3.shape[1], c3.shape[2], inner2.shape[3]),
            method="nearest",
        )
        inner1 = self.inner_block_1(c3) + inner2_up
        p3 = self.layer_block_1(inner1)

        # C2 + upsampled inner1 -> inner0
        inner1_up = jax.image.resize(
            inner1,
            shape=(inner1.shape[0], c2.shape[1], c2.shape[2], inner1.shape[3]),
            method="nearest",
        )
        inner0 = self.inner_block_0(c2) + inner1_up
        p2 = self.layer_block_0(inner0)

        # P6 (pool level) via 1x1 maxpool stride 2
        p6 = nnx.max_pool(p5, window_shape=(1, 1), strides=(2, 2))

        return {
            "0": p2,
            "1": p3,
            "2": p4,
            "3": p5,
            "pool": p6,
        }


class BackboneWithFPN(nnx.Module):
    """ResNet-50 backbone with FPN V2."""

    def __init__(self, *, rngs: nnx.Rngs):
        self.body = ResNet50(rngs=rngs)
        self.fpn = FeaturePyramidNetwork(rngs=rngs)

    def __call__(self, x: jax.Array) -> dict[str, jax.Array]:
        c2, c3, c4, c5 = self.body(x)
        return self.fpn(c2, c3, c4, c5)
