"""Deterministic patient-level split manifests."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .manifest import dataset_fingerprint


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.80
    validation: float = 0.10
    test: float = 0.10
    seed: int = 7

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value <= 0.0 for value in values):
            raise ValueError("All split fractions must be greater than zero.")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-8):
            raise ValueError(f"Split fractions must sum to 1.0, got {sum(values):.8f}.")


def _age_bins(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    ages = np.asarray([record["age"] for record in records if record.get("age") is not None], dtype=np.float32)
    edges = np.unique(np.quantile(ages, [0.25, 0.50, 0.75])) if len(ages) else np.asarray([])
    return {
        record["patient_id"]: int(np.searchsorted(edges, record["age"], side="right"))
        if record.get("age") is not None
        else -1
        for record in records
    }


def _allocate_counts(size: int, fractions: Sequence[float]) -> list[int]:
    raw = np.asarray(fractions, dtype=np.float64) * size
    counts = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - counts), kind="stable")[: size - int(counts.sum())]:
        counts[int(index)] += 1
    if size >= len(counts):
        for index in range(len(counts)):
            if counts[index] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[index] += 1
    return counts.tolist()


def validate_split_manifest(manifest: Mapping[str, Any], expected_ids: Sequence[str] | None = None) -> None:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != {"train", "validation", "test"}:
        raise ValueError("Split manifest must contain train, validation, and test splits.")
    if any(not isinstance(values, list) for values in splits.values()):
        raise ValueError("Each split in the manifest must be a list of patient IDs.")
    assigned = [patient_id for values in splits.values() for patient_id in values]
    if len(assigned) != len(set(assigned)):
        raise ValueError("Patient leakage detected: an ID appears in more than one split.")
    if expected_ids is not None and set(assigned) != set(expected_ids):
        raise ValueError("Split manifest does not match the discovered patient IDs.")


def create_split_manifest(records: Sequence[Mapping[str, Any]], config: SplitConfig = SplitConfig()) -> dict[str, Any]:
    """Create a deterministic patient-level split stratified by label, sex, and age bin."""
    records = list(records)
    if not records:
        raise ValueError("Cannot split an empty manifest.")
    ids = [str(record["patient_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("A patient may occur only once in the split input.")

    age_bins = _age_bins(records)
    strata: dict[tuple[Any, ...], list[str]] = {}
    for record in records:
        key = (record.get("label"), record.get("gender"), age_bins[record["patient_id"]])
        strata.setdefault(key, []).append(record["patient_id"])

    rng = random.Random(config.seed)
    splits = {"train": [], "validation": [], "test": []}
    for key in sorted(strata, key=str):
        group = list(strata[key])
        rng.shuffle(group)
        counts = _allocate_counts(len(group), (config.train, config.validation, config.test))
        start = 0
        for name, count in zip(splits, counts):
            splits[name].extend(group[start : start + count])
            start += count
    for values in splits.values():
        values.sort()

    manifest = {
        "version": 1,
        "seed": config.seed,
        "fractions": {"train": config.train, "validation": config.validation, "test": config.test},
        "record_count": len(records),
        "dataset_fingerprint": dataset_fingerprint(records),
        "splits": splits,
    }
    validate_split_manifest(manifest, ids)
    return manifest


def save_split_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    path_text = str(path)
    if path_text.startswith("gs://cxr-rait"):
        raise ValueError("Do not write split manifests inside the read-only gs://cxr-rait/ dataset bucket.")
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split_manifest(manifest)
    return manifest
