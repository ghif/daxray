"""Multi-scale RoIAlign implemented in pure JAX."""

from typing import Sequence
import jax
import jax.numpy as jnp


def roi_align_single_feature_map(
    feature_map: jax.Array,
    boxes: jax.Array,
    output_size: tuple[int, int] = (7, 7),
    spatial_scale: float = 0.25,
    sampling_ratio: int = 2,
) -> jax.Array:
    """Performs RoIAlign for a batch of boxes on a single feature map.

    Args:
        feature_map: Feature tensor of shape (H, W, C).
        boxes: Bounding boxes of shape (N, 4) in image coordinate space [x1, y1, x2, y2].
        output_size: Output spatial resolution (out_h, out_w), typically (7, 7).
        spatial_scale: Scale factor to map image coords to feature map coords (e.g. 1/4).
        sampling_ratio: Number of sampling points per bin along each axis (default: 2).

    Returns:
        Pooled features of shape (N, out_h, out_w, C).
    """
    out_h, out_w = output_size
    grid_h = sampling_ratio if sampling_ratio > 0 else 2
    grid_w = sampling_ratio if sampling_ratio > 0 else 2
    height, width, _ = feature_map.shape

    def _align_box(box: jax.Array) -> jax.Array:
        # Scale box coords to feature map space
        x1 = box[0] * spatial_scale
        y1 = box[1] * spatial_scale
        x2 = box[2] * spatial_scale
        y2 = box[3] * spatial_scale

        roi_w = jnp.maximum(x2 - x1, 1.0)
        roi_h = jnp.maximum(y2 - y1, 1.0)

        bin_size_h = roi_h / float(out_h)
        bin_size_w = roi_w / float(out_w)

        # Sampling grids along height and width
        ph = jnp.arange(out_h, dtype=jnp.float32)[:, None]
        iy = jnp.arange(grid_h, dtype=jnp.float32)[None, :]
        ys = y1 + ph * bin_size_h + (iy + 0.5) * (bin_size_h / float(grid_h))

        pw = jnp.arange(out_w, dtype=jnp.float32)[:, None]
        ix = jnp.arange(grid_w, dtype=jnp.float32)[None, :]
        xs = x1 + pw * bin_size_w + (ix + 0.5) * (bin_size_w / float(grid_w))

        # Broadcast to 4D grid: (out_h, out_w, grid_h, grid_w)
        ys_grid = jnp.broadcast_to(ys[:, None, :, None], (out_h, out_w, grid_h, grid_w))
        xs_grid = jnp.broadcast_to(xs[None, :, None, :], (out_h, out_w, grid_h, grid_w))

        # Continuous coordinates to 4 discrete neighbors
        y_low = jnp.floor(ys_grid).astype(jnp.int32)
        x_low = jnp.floor(xs_grid).astype(jnp.int32)
        y_high = y_low + 1
        x_high = x_low + 1

        ly = ys_grid - y_low
        lx = xs_grid - x_low
        hy = 1.0 - ly
        hx = 1.0 - lx

        w1 = hy * hx
        w2 = hy * lx
        w3 = ly * hx
        w4 = ly * lx

        # Validity mask for bounds
        valid_1 = (y_low >= 0) & (y_low < height) & (x_low >= 0) & (x_low < width)
        valid_2 = (y_low >= 0) & (y_low < height) & (x_high >= 0) & (x_high < width)
        valid_3 = (y_high >= 0) & (y_high < height) & (x_low >= 0) & (x_low < width)
        valid_4 = (y_high >= 0) & (y_high < height) & (x_high >= 0) & (x_high < width)

        y_low_c = jnp.clip(y_low, 0, height - 1)
        y_high_c = jnp.clip(y_high, 0, height - 1)
        x_low_c = jnp.clip(x_low, 0, width - 1)
        x_high_c = jnp.clip(x_high, 0, width - 1)

        v1 = feature_map[y_low_c, x_low_c] * (w1 * valid_1)[..., None]
        v2 = feature_map[y_low_c, x_high_c] * (w2 * valid_2)[..., None]
        v3 = feature_map[y_high_c, x_low_c] * (w3 * valid_3)[..., None]
        v4 = feature_map[y_high_c, x_high_c] * (w4 * valid_4)[..., None]

        sampled = v1 + v2 + v3 + v4  # (out_h, out_w, grid_h, grid_w, C)
        return jnp.mean(sampled, axis=(2, 3))

    if boxes.shape[0] == 0:
        c = feature_map.shape[-1]
        return jnp.zeros((0, out_h, out_w, c), dtype=feature_map.dtype)

    return jax.vmap(_align_box)(boxes)


def assign_boxes_to_levels(
    boxes: jax.Array,
    canonical_scale: float = 224.0,
    canonical_level: int = 4,
    min_level: int = 2,
    max_level: int = 5,
) -> jax.Array:
    """Computes target FPN pyramid level for each box according to FPN heuristic.

    Level heuristic:
        k = floor(k0 + log2(sqrt(w * h) / 224)) clamped to [min_level, max_level].

    Args:
        boxes: Bounding boxes of shape (N, 4) in [x1, y1, x2, y2].
        canonical_scale: Reference scale (s0 = 224.0).
        canonical_level: Reference level (lvl0 = 4).
        min_level: Minimum pyramid level (e.g. 2 for P2).
        max_level: Maximum pyramid level (e.g. 5 for P5).

    Returns:
        Level index for each box in range [0, max_level - min_level].
    """
    widths = jnp.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
    heights = jnp.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
    areas = widths * heights
    s = jnp.sqrt(areas)

    target_levels = jnp.floor(
        canonical_level + jnp.log2(s / canonical_scale + 1e-6)
    ).astype(jnp.int32)
    target_levels = jnp.clip(target_levels, min_level, max_level)
    return target_levels - min_level


def multi_scale_roi_align(
    features: dict[str, jax.Array],
    boxes: jax.Array,
    output_size: tuple[int, int] = (7, 7),
    sampling_ratio: int = 2,
    featmap_names: Sequence[str] = ("0", "1", "2", "3"),
    spatial_scales: Sequence[float] = (0.25, 0.125, 0.0625, 0.03125),
) -> jax.Array:
    """Performs multi-scale RoIAlign pooling across FPN feature levels.

    Args:
        features: Dictionary of FPN feature maps where keys are '0', '1', '2', '3',
            with shapes (1, H, W, C) or (H, W, C).
        boxes: RoI bounding boxes of shape (N, 4) in [x1, y1, x2, y2].
        output_size: (7, 7) output size per RoI.
        sampling_ratio: Sampling ratio per bin.
        featmap_names: Level keys to use ('0', '1', '2', '3').
        spatial_scales: Corresponding spatial scales (1/4, 1/8, 1/16, 1/32).

    Returns:
        Pooled features of shape (N, 7, 7, C).
    """
    n_boxes = boxes.shape[0]
    first_feat = features[featmap_names[0]]
    if first_feat.ndim == 4:
        first_feat = first_feat[0]
    c_channels = first_feat.shape[-1]
    dtype = first_feat.dtype

    if n_boxes == 0:
        return jnp.zeros((0, output_size[0], output_size[1], c_channels), dtype=dtype)

    level_indices = assign_boxes_to_levels(boxes)
    output = jnp.zeros((n_boxes, output_size[0], output_size[1], c_channels), dtype=dtype)

    for lvl_idx, (name, scale) in enumerate(zip(featmap_names, spatial_scales)):
        feat = features[name]
        if feat.ndim == 4:
            feat = feat[0]
        mask = (level_indices == lvl_idx)
        # Compute RoI align for all boxes on this feature map
        aligned = roi_align_single_feature_map(
            feature_map=feat,
            boxes=boxes,
            output_size=output_size,
            spatial_scale=scale,
            sampling_ratio=sampling_ratio,
        )
        # Selectively update where mask is True
        output = jnp.where(mask[:, None, None, None], aligned, output)

    return output
