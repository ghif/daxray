import jax.numpy as jnp
import numpy as np
from flax import nnx

from daxray.models import CxrSmallCNN, model_parameter_count
from daxray.training.cnn_baseline import create_cnn_optimizer, train_cnn


def test_cnn_outputs_one_logit_and_uses_layer_norm():
    model = CxrSmallCNN(rngs=nnx.Rngs(0), dropout_rate=0.2)
    outputs = model(jnp.zeros((2, 128, 128, 1)), training=False)

    assert outputs.shape == (2,)
    assert model.block1.norm is not None
    assert model.block2.norm is not None
    assert model.block3.norm is not None
    assert model_parameter_count(model) > 1000


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
