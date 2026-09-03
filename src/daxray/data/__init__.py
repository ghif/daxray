"""Data interfaces and loaders for DAXRay."""

from .batching import iter_batches
from .cxr_rait import (
    audit_cxr_rait_manifest,
    filter_records_by_split,
    load_cxr_rait_records,
    parse_binary_tb_label,
    parse_site_from_patient_id,
)
from .dicom import (
    DETECTION_PREPROCESSING_VERSION,
    CacheStats,
    PreprocessingCache,
    TransformDetails,
    load_cxr_image,
    load_cxr_image_for_detection,
)
from .manifest import audit_manifest, build_cxr_rait_manifest, dataset_fingerprint
from .splits import (
    SplitConfig,
    create_split_manifest,
    load_split_manifest,
    save_split_manifest,
    validate_split_manifest,
)
from .tbx11k import BoundingBox, TBX11KDataset, TBX11KSample, parse_coco_json, parse_voc_xml

__all__ = [
    "BoundingBox",
    "CacheStats",
    "DETECTION_PREPROCESSING_VERSION",
    "PreprocessingCache",
    "SplitConfig",
    "TBX11KDataset",
    "TBX11KSample",
    "TransformDetails",
    "audit_cxr_rait_manifest",
    "audit_manifest",
    "build_cxr_rait_manifest",
    "create_split_manifest",
    "dataset_fingerprint",
    "filter_records_by_split",
    "iter_batches",
    "load_cxr_image",
    "load_cxr_image_for_detection",
    "load_cxr_rait_records",
    "load_split_manifest",
    "parse_binary_tb_label",
    "parse_coco_json",
    "parse_site_from_patient_id",
    "parse_voc_xml",
    "save_split_manifest",
    "validate_split_manifest",
]
