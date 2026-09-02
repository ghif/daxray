"""Data interfaces and loaders for DAXRay."""

from .batching import iter_batches
from .dicom import CacheStats, PreprocessingCache, load_cxr_image
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
    "PreprocessingCache",
    "SplitConfig",
    "TBX11KDataset",
    "TBX11KSample",
    "audit_manifest",
    "build_cxr_rait_manifest",
    "create_split_manifest",
    "dataset_fingerprint",
    "iter_batches",
    "load_cxr_image",
    "load_split_manifest",
    "parse_coco_json",
    "parse_voc_xml",
    "save_split_manifest",
    "validate_split_manifest",
]
