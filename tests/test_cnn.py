import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from daxray.models import CxrSmallCNN, compute_dtype_for_precision, model_parameter_count
from daxray.training.cnn_baseline import create_cnn_optimizer, train_cnn
from daxray.training.cnn_baseline import _select_threshold


def test_cnn_outputs_one_logit_and_uses_layer_norm():
    model = CxrSmallCNN(rngs=nnx.Rngs(0), dropout_rate=0.2)
    outputs = model(jnp.zeros((2, 128, 128, 1)), training=False)

    assert outputs.shape == (2,)
    assert model.block1.norm is not None
    assert model.block2.norm is not None
    assert model.block3.norm is not None
    assert model_parameter_count(model) > 1000


def test_wider_cnn_has_more_parameters():
    small = CxrSmallCNN(rngs=nnx.Rngs(10), base_channels=16)
    wide = CxrSmallCNN(rngs=nnx.Rngs(11), base_channels=32)

    assert model_parameter_count(wide) > model_parameter_count(small)


def test_cnn_training_reduces_loss_on_simple_signal():
    model = CxrSmallCNN(rngs=nnx.Rngs(1), dropout_rate=0.0)
    optimizer = create_cnn_optimizer(model, learning_rate=1e-2, weight_decay=0.0)
    images = np.zeros((4, 128, 128, 1), dtype=np.float32)
    images[2:] = 1.0
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)
    batches = [{"image": images, "label": labels}]

    progress = []
    first_loss = train_cnn(model, optimizer, iter(batches), progress_callback=lambda *event: progress.append(event))
    for _ in range(10):
        last_loss = train_cnn(model, optimizer, iter(batches))

    assert last_loss < first_loss
    assert progress and progress[0][0] == 1
    assert progress[0][1] == first_loss


def test_cnn_training_masks_padded_labels_before_loss_reduction():
    model = CxrSmallCNN(rngs=nnx.Rngs(2), dropout_rate=0.0)
    optimizer = create_cnn_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    batch = {
        "image": np.zeros((2, 128, 128, 1), dtype=np.float32),
        "label": np.asarray([1, -1], dtype=np.int32),
        "label_mask": np.asarray([True, False]),
    }

    loss = train_cnn(model, optimizer, iter([batch]), batch_size=2)

    assert np.isfinite(loss)
    assert loss >= 0.0


def test_select_threshold_can_avoid_all_positive_predictions():
    labels = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.2, 0.4, 0.6, 0.8])

    threshold = _select_threshold(labels, probabilities)

    assert threshold == 0.5


def test_bfloat16_model_uses_bfloat16_compute_and_float32_parameters():
    model = CxrSmallCNN(rngs=nnx.Rngs(3), dtype=jnp.bfloat16)
    output = model(jnp.ones((2, 32, 32, 1), dtype=jnp.float32))

    assert output.dtype == jnp.bfloat16
    assert model.block1.conv.kernel.get_value().dtype == jnp.float32


@pytest.mark.parametrize("precision, expected", [("fp32", jnp.float32), ("bf16", jnp.bfloat16)])
def test_compute_dtype_for_precision(precision, expected):
    assert compute_dtype_for_precision(precision) == expected


def test_compute_dtype_for_precision_rejects_unknown_precision():
    with pytest.raises(ValueError, match="precision must be"):
        compute_dtype_for_precision("fp16")


def test_cosine_optimizer_requires_total_steps():
    model = CxrSmallCNN(rngs=nnx.Rngs(12))
    with pytest.raises(ValueError, match="total_steps"):
        create_cnn_optimizer(model, schedule="cosine")
