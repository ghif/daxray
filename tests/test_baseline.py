import numpy as np

from daxray.evaluation.metrics import classification_metrics
from daxray.models import fit_logistic_classifier
from daxray.training.metadata_baseline import FEATURE_NAMES, metadata_features


def test_metadata_features_have_stable_shape_and_missingness_encoding():
    records = [
        {"age": 50, "gender": "M", "label": 1},
        {"age": None, "gender": None, "label": 0},
    ]

    features, labels = metadata_features(records)

    assert FEATURE_NAMES == ("age_normalized", "age_missing", "gender_male", "gender_female", "gender_missing")
    assert features.shape == (2, 5)
    assert features.dtype == np.float32
    assert labels.tolist() == [1, 0]
    np.testing.assert_allclose(features[1], [0.45, 1.0, 0.0, 0.0, 1.0])


def test_jax_logistic_baseline_learns_separable_data():
    features = np.asarray([[0.0], [0.1], [0.9], [1.0]], dtype=np.float32)
    labels = np.asarray([0, 0, 1, 1], dtype=np.int32)

    model, history = fit_logistic_classifier(features, labels, learning_rate=0.5, epochs=200)
    probabilities = model.predict_proba(features)
    metrics = classification_metrics(labels, probabilities)

    assert history[-1] < history[0]
    assert metrics["accuracy"] == 1.0
    assert metrics["auroc"] == 1.0


def test_classification_metrics_reject_non_binary_labels():
    try:
        classification_metrics(np.asarray([0, 2]), np.asarray([0.1, 0.9]))
    except ValueError as exc:
        assert "binary" in str(exc)
    else:
        raise AssertionError("Expected non-binary labels to fail validation.")
