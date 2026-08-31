"""Dataset manifest construction and audit summaries."""

from __future__ import annotations

import hashlib
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence

from .metadata import normalise_id, parse_age, parse_binary_label, read_metadata


def _discover_dicoms(root: str | Path) -> dict[str, str]:
    root_text = str(root)
    if root_text.startswith("gs://"):
        import fsspec

        fs, prefix = fsspec.core.url_to_fs(root_text)
        paths = sorted(fs.glob(prefix.rstrip("/") + "/**/*.dcm"))
        paths = [fs.unstrip_protocol(path) for path in paths]
    else:
        paths = sorted(str(path) for path in Path(root_text).rglob("*.dcm") if path.is_file())
    result: dict[str, str] = {}
    for path in paths:
        patient_id = PurePosixPath(str(path)).stem.strip()
        if patient_id in result:
            raise ValueError(f"Multiple DICOM files found for patient {patient_id!r}.")
        result[patient_id] = str(path)
    return result


def build_cxr_rait_manifest(
    root: str | Path,
    metadata_path: str | Path | None = None,
    *,
    patient_id_column: str = "Patient ID",
    label_columns: Sequence[str] = ("BTA.1", "BTA", "tb_status"),
    age_columns: Sequence[str] = ("Usia", "Age", "Birth", "Unnamed: 15"),
    gender_column: str = "Gender",
    domain: str = "cxr_rait",
    require_label: bool = True,
) -> list[dict[str, Any]]:
    """Join metadata to local DICOM files while preserving domain provenance."""
    root_text = str(root)
    if not root_text.startswith("gs://"):
        raise ValueError("CXR-RAIT must be loaded from its authoritative gs://cxr-rait/ location.")
    metadata_path = str(metadata_path) if metadata_path else root_text.rstrip("/") + "/data_demography.xlsx"
    records = read_metadata(metadata_path)
    dicoms = _discover_dicoms(root_text)
    manifest: list[dict[str, Any]] = []

    for row in records:
        raw_id = row.get(patient_id_column)
        if raw_id is None:
            continue
        patient_id = normalise_id(raw_id)
        if not patient_id or patient_id.lower() == "nan":
            continue
        label = None
        for column in label_columns:
            label = parse_binary_label(row.get(column))
            if label is not None:
                break
        if require_label and label is None:
            continue
        image_path = dicoms.get(patient_id)
        if image_path is None:
            continue
        gender = str(row.get(gender_column, "")).strip().lower() or None
        manifest.append(
            {
                "patient_id": patient_id,
                "study_id": patient_id,
                "image_path": image_path,
                "domain": domain,
                "label": label,
                "age": parse_age(row, age_columns),
                "gender": gender,
            }
        )

    if not manifest:
        raise ValueError("No CXR-RAIT DICOM records could be matched to metadata.")
    patient_ids = [record["patient_id"] for record in manifest]
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("Metadata contains duplicate patient IDs.")
    return sorted(manifest, key=lambda record: record["patient_id"])


def dataset_fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    value = "\n".join(
        f"{record['patient_id']}\t{record.get('study_id')}\t{record.get('domain')}\t"
        f"{record.get('image_path', '')}\t{record.get('label')}"
        for record in sorted(records, key=lambda item: str(item["patient_id"]))
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def audit_manifest(
    records: Sequence[Mapping[str, Any]],
    split_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a serialisable audit summary for a dataset manifest."""
    records = list(records)
    labels = [record.get("label") for record in records]
    summary: dict[str, Any] = {
        "records": len(records),
        "unique_patients": len({record.get("patient_id") for record in records}),
        "unique_studies": len({record.get("study_id") for record in records}),
        "domains": {},
        "labels": {"positive": sum(label == 1 for label in labels), "negative": sum(label == 0 for label in labels), "missing": sum(label is None for label in labels)},
        "missing_image_paths": sum(not _image_exists(record["image_path"]) for record in records),
        "missing_ages": sum(record.get("age") is None for record in records),
        "missing_genders": sum(record.get("gender") is None for record in records),
        "fingerprint": dataset_fingerprint(records),
    }
    for record in records:
        domain = record.get("domain", "unknown")
        summary["domains"][domain] = summary["domains"].get(domain, 0) + 1
    if split_manifest is not None:
        summary["split_counts"] = {name: len(values) for name, values in split_manifest["splits"].items()}
    return summary


def _image_exists(path: str) -> bool:
    if path.startswith("gs://"):
        import fsspec
        fs, name = fsspec.core.url_to_fs(path)
        return fs.exists(name)
    return Path(path).is_file()
