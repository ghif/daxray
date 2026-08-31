"""Metadata readers and CXR-RAIT field parsers."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def normalise_id(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def parse_binary_label(value: Any) -> int | None:
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "na", "n/a"}:
        return None
    if text in {"1", "1.0", "+", "positive", "pos", "yes", "true", "tb"}:
        return 1
    if text in {"0", "0.0", "-", "negative", "neg", "no", "false", "non-tb", "non_tb"}:
        return 0
    try:
        return int(float(text) > 0)
    except ValueError:
        return None


def parse_age(row: Mapping[str, Any], columns: Sequence[str]) -> float | None:
    for column in columns:
        try:
            age = float(row.get(column))
        except (TypeError, ValueError):
            continue
        if 0.0 < age < 120.0:
            return age
    return None


def read_metadata(path: str | Path) -> list[dict[str, Any]]:
    """Read an Excel workbook or CSV metadata export."""
    path_text = str(path)
    suffix = Path(path_text).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Reading Excel metadata requires pandas and openpyxl.") from exc
        if path_text.startswith("gs://"):
            import fsspec
            with fsspec.open(path_text, mode="rb") as handle:
                frame = pd.read_excel(handle, sheet_name=0)
        else:
            frame = pd.read_excel(Path(path_text), sheet_name=0)
        return frame.replace({np.nan: None}).to_dict(orient="records")
    if path_text.startswith("gs://"):
        import fsspec
        with fsspec.open(path_text, mode="rt", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    with Path(path_text).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))
