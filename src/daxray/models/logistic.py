"""Small JAX logistic-regression baseline."""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np


class LogisticClassifier:
    """Binary logistic classifier with an explicit linear parameterization."""

    def __init__(self, weights: np.ndarray, bias: float):
        self.weights = np.asarray(weights, dtype=np.float32)
        self.bias = float(bias)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = np.asarray(features, dtype=np.float32) @ self.weights + self.bias
        return (1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"weights": self.weights.tolist(), "bias": self.bias}


def fit_logistic_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    learning_rate: float = 0.1,
    epochs: int = 500,
    l2_penalty: float = 1e-3,
) -> tuple[LogisticClassifier, list[float]]:
    """Fit a deterministic full-batch logistic model using JAX autodiff."""
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
        raise ValueError("features must be 2-D and labels must be a matching 1-D array.")
    if len(features) == 0:
        raise ValueError("Cannot fit a classifier on an empty dataset.")
    if epochs <= 0 or learning_rate <= 0.0 or l2_penalty < 0.0:
        raise ValueError("epochs and learning_rate must be positive; l2_penalty cannot be negative.")
    x = jnp.asarray(features, dtype=jnp.float32)
    y = jnp.asarray(labels, dtype=jnp.float32)
    params = {"weights": jnp.zeros(x.shape[1], dtype=jnp.float32), "bias": jnp.array(0.0, dtype=jnp.float32)}

    def loss_fn(current: Mapping[str, Any]) -> jnp.ndarray:
        logits = x @ current["weights"] + current["bias"]
        data_loss = jnp.mean(jnp.maximum(logits, 0.0) - logits * y + jnp.log1p(jnp.exp(-jnp.abs(logits))))
        return data_loss + 0.5 * l2_penalty * jnp.sum(current["weights"] ** 2)

    history: list[float] = []
    for _ in range(epochs):
        loss, gradients = jax.value_and_grad(loss_fn)(params)
        params = {name: value - learning_rate * gradients[name] for name, value in params.items()}
        history.append(float(loss))
    return LogisticClassifier(np.asarray(params["weights"]), float(params["bias"])), history
