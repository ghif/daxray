"""Feature preparation for the metadata-only TB baseline."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


FEATURE_NAMES = ("age_normalized", "age_missing", "gender_male", "gender_female", "gender_missing")


def metadata_features(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Encode age and gender without reading images."""
    features = []
    labels = []
    for record in records:
        age = record.get("age")
        age_missing = age is None
        age_value = float(age) if not age_missing else 45.0
        age_value = np.clip(age_value, 0.0, 100.0) / 100.0
        gender = str(record.get("gender") or "").strip().lower()
        male = gender in {"m", "male", "1", "1.0", "l", "laki-laki"}
        female = gender in {"f", "female", "0", "0.0", "p", "perempuan"}
        features.append([age_value, float(age_missing), float(male), float(female), float(not male and not female)])
        label = record.get("label")
        if label is None:
            raise ValueError("Metadata baseline requires a label for every record.")
        labels.append(int(label))
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32)
