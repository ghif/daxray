"""CXR-RAIT dataset manifest loading, filtering, and auditing for binary TB evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .manifest import _image_exists, dataset_fingerprint
from .metadata import normalise_id, parse_age


def parse_site_from_patient_id(patient_id: str) -> str:
    """Extract clinical site or collection site identifier from patient ID."""
    pid = str(patient_id).strip()
    match = re.match(r"^([A-Za-z]+\d{2})", pid)
    if match:
        return match.group(1).upper()
    match_alpha = re.match(r"^([A-Za-z]+)", pid)
    if match_alpha:
        return match_alpha.group(1).upper()
    return "SITE_DEFAULT"


def parse_binary_tb_label(raw: Any) -> int | None | str:
    """Parse label strictly into binary Tuberculosis (1), Non-Tuberculosis (0), Unlabeled (None), or Excluded.

    Any unknown, indeterminate, or non-binary category is marked as 'excluded'
    so it is never collapsed into positive or negative supervised metrics.
    """
    if raw is None:
        return None

    if isinstance(raw, bool):
        return 1 if raw else 0

    if isinstance(raw, (int, float)):
        if isinstance(raw, float) and (raw != raw):  # NaN
            return None
        if int(raw) == 1:
            return 1
        if int(raw) == 0:
            return 0
        if int(raw) == -1:
            return None
        return "excluded"

    text = str(raw).strip().lower()
    if text in {"", "none", "nan", "null", "unlabeled", "missing", "-1"}:
        return None

    if (
        text in {"1", "true", "positive", "tb", "tb_positive", "tb positive", "active", "latent", "active_tb", "latent_tb", "active tb", "latent tb"}
        or text.startswith("active")
        or text.startswith("latent")
        or text.startswith("tb positive")
        or text.startswith("positive")
    ):
        return 1

    if (
        text in {"0", "false", "negative", "normal", "healthy", "non-tb", "non_tb", "tb_negative", "tb negative", "non tb"}
        or text.startswith("non-tb")
        or text.startswith("non_tb")
        or text.startswith("healthy")
        or text.startswith("normal")
        or text.startswith("negative")
    ):
        return 0

    return "excluded"


def load_cxr_rait_records(
    manifest_path: str | Path,
    base_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load CXR-RAIT manifest records from JSON or CSV.

    Standardizes records into a uniform schema with explicit binary label classification.
    """
    path = Path(manifest_path)
    if not path.exists() and not str(manifest_path).startswith("gs://"):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    raw_data: Any
    if str(manifest_path).startswith("gs://"):
        import fsspec

        with fsspec.open(str(manifest_path), mode="r", encoding="utf-8") as f:
            raw_data = json.load(f)
    elif path.suffix.lower() == ".json":
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            raw_data = list(reader)
    else:
        raise ValueError(f"Unsupported manifest file extension: {path.suffix}")

    raw_records: list[dict[str, Any]] = []
    if isinstance(raw_data, list):
        raw_records = raw_data
    elif isinstance(raw_data, dict):
        if "records" in raw_data and isinstance(raw_data["records"], list):
            raw_records = raw_data["records"]
        elif "manifest" in raw_data and isinstance(raw_data["manifest"], list):
            raw_records = raw_data["manifest"]
        elif "splits" in raw_data and isinstance(raw_data["splits"], dict):
            # Split manifest format: extract records with split assignments
            raw_records = []
            for split_name, pids in raw_data["splits"].items():
                for pid in pids:
                    raw_records.append({
                        "patient_id": pid,
                        "study_id": pid,
                        "split": split_name,
                    })
        else:
            raise ValueError(
                f"Invalid JSON manifest structure at {manifest_path}: expected list of records, 'records' key, or 'splits' dictionary."
            )

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_records):
        raw_id = row.get("patient_id") or row.get("Patient ID") or row.get("id") or f"P_{idx:05d}"
        patient_id = normalise_id(raw_id)
        if not patient_id or patient_id.lower() == "nan":
            continue

        study_id = str(row.get("study_id") or row.get("Study ID") or patient_id).strip()
        image_path = str(row.get("image_path") or row.get("file_path") or row.get("path") or "").strip()

        if base_dir and image_path and not image_path.startswith("gs://") and not Path(image_path).is_absolute():
            image_path = str(Path(base_dir) / image_path)

        raw_label = row.get("label")
        if raw_label is None:
            for cand_col in ("BTA.1", "BTA", "tb_status", "status", "diagnosis"):
                if cand_col in row:
                    raw_label = row[cand_col]
                    break

        label = parse_binary_tb_label(raw_label)
        site = str(row.get("site") or parse_site_from_patient_id(patient_id)).strip().upper()
        domain = str(row.get("domain") or "cxr_rait").strip()

        age = parse_age(row, ("Usia", "Age", "Birth", "age")) if isinstance(row, dict) else None
        gender = str(row.get("Gender") or row.get("gender") or "").strip().lower() or None

        boxes = row.get("boxes", [])

        records.append(
            {
                "patient_id": patient_id,
                "study_id": study_id,
                "image_path": image_path,
                "domain": domain,
                "site": site,
                "label": label,
                "raw_label": raw_label,
                "is_supervised": label in (0, 1),
                "is_excluded": label == "excluded",
                "boxes": boxes,
                "age": age,
                "gender": gender,
                "split": row.get("split"),
            }
        )

    return records


def filter_records_by_split(
    records: Sequence[dict[str, Any]],
    split: str = "all",
    split_manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Filter records by target split ('all', 'train', 'validation', 'test')."""
    split_norm = split.strip().lower()
    if split_norm == "all":
        return list(records)

    if split_manifest is not None and "splits" in split_manifest:
        splits = split_manifest["splits"]
        if split_norm not in splits:
            raise ValueError(
                f"Split '{split}' not found in split manifest. Available splits: {list(splits.keys())}"
            )
        allowed_pids = set(splits[split_norm])
        return [r for r in records if r["patient_id"] in allowed_pids]

    # If no split manifest, check if records have a 'split' field
    if any(r.get("split") is not None for r in records):
        return [r for r in records if str(r.get("split", "")).lower() == split_norm]

    raise ValueError(
        f"Cannot filter by split '{split}': no split manifest was provided and records do not have a 'split' field."
    )


def audit_cxr_rait_manifest(
    records: Sequence[Mapping[str, Any]],
    check_image_paths: bool = True,
) -> dict[str, Any]:
    """Audit CXR-RAIT target pool with strict binary label counting and site breakdown."""
    records_list = list(records)
    total_records = len(records_list)
    unique_patients = len({r["patient_id"] for r in records_list})
    unique_studies = len({r["study_id"] for r in records_list})

    tb_positive_count = 0
    tb_negative_count = 0
    unlabeled_count = 0
    excluded_count = 0

    sites: dict[str, int] = {}
    site_patients: dict[str, set[str]] = {}
    missing_images: list[dict[str, str]] = []
    total_boxes = 0

    for r in records_list:
        lbl = r.get("label")
        if lbl == 1:
            tb_positive_count += 1
        elif lbl == 0:
            tb_negative_count += 1
        elif lbl == "excluded":
            excluded_count += 1
        else:
            unlabeled_count += 1

        site = str(r.get("site") or "SITE_DEFAULT")
        sites[site] = sites.get(site, 0) + 1
        site_patients.setdefault(site, set()).add(r["patient_id"])

        boxes = r.get("boxes") or []
        total_boxes += len(boxes)

        img_p = str(r.get("image_path") or "")
        if check_image_paths:
            if not img_p or not _image_exists(img_p):
                missing_images.append({
                    "patient_id": str(r.get("patient_id")),
                    "image_path": img_p,
                })

    site_patient_counts = {site: len(pids) for site, pids in site_patients.items()}

    return {
        "total_records": total_records,
        "unique_patients": unique_patients,
        "unique_studies": unique_studies,
        "label_counts": {
            "tb_positive": tb_positive_count,
            "tb_negative": tb_negative_count,
            "unlabeled": unlabeled_count,
            "excluded": excluded_count,
        },
        "supervised_eval_count": tb_positive_count + tb_negative_count,
        "sites": sites,
        "site_patient_counts": site_patient_counts,
        "missing_or_unreadable_images": len(missing_images),
        "missing_image_details": missing_images[:50],  # Cap preview to first 50
        "has_bounding_boxes": total_boxes > 0,
        "total_bounding_boxes": total_boxes,
        "fingerprint": dataset_fingerprint(records_list),
        "binary_scope_enforced": True,
    }
