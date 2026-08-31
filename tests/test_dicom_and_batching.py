import numpy as np
from pathlib import Path
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from daxray.data import PreprocessingCache, audit_manifest, iter_batches, load_cxr_image


def _write_dicom(path, values, photometric="MONOCHROME2"):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.Rows, dataset.Columns = values.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = photometric
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = 2
    dataset.RescaleIntercept = -10
    dataset.PixelData = values.astype(np.uint16).tobytes()
    dataset.save_as(path)


def test_dicom_preprocessing_shape_range_and_monochrome1(tmp_path):
    path = tmp_path / "P001.dcm"
    _write_dicom(path, np.array([[0, 1], [2, 3]], dtype=np.uint16), "MONOCHROME1")

    image = load_cxr_image(path, image_size=2, resize_mode="stretch")

    assert image.shape == (1, 2, 2)
    assert image.dtype == np.float32
    assert 0.0 <= image.min() <= image.max() <= 1.0
    assert image[0, 0, 0] == 1.0
    assert image[0, 1, 1] == 0.0


def test_batching_supports_unlabeled_records(tmp_path):
    path = tmp_path / "P001.dcm"
    _write_dicom(path, np.arange(4).reshape(2, 2))
    records = [{"patient_id": "P001", "image_path": str(path), "label": None}]

    batch = next(iter_batches(records, ["P001"], batch_size=1, image_size=2, resize_mode="stretch"))

    assert batch["image"].shape == (1, 1, 2, 2)
    assert batch["label"].tolist() == [-1]
    assert batch["label_mask"].tolist() == [False]


def test_memory_cache_reuses_preprocessed_image_and_respects_preprocessing_key(tmp_path):
    path = tmp_path / "P001.dcm"
    _write_dicom(path, np.arange(4).reshape(2, 2))
    cache = PreprocessingCache(mode="memory", max_bytes=1024)

    first = load_cxr_image(path, image_size=2, resize_mode="stretch", cache=cache)
    second = load_cxr_image(path, image_size=2, resize_mode="stretch", cache=cache)
    other_size = load_cxr_image(path, image_size=3, resize_mode="stretch", cache=cache)

    assert np.array_equal(first, second)
    assert other_size.shape == (1, 3, 3)
    assert cache.stats.hits == 1
    assert cache.stats.misses == 2
    cache.close()


def test_cache_eviction_and_ephemeral_cleanup(tmp_path):
    path = tmp_path / "P001.dcm"
    _write_dicom(path, np.arange(4).reshape(2, 2))
    cache = PreprocessingCache(mode="ephemeral", max_bytes=40)
    load_cxr_image(path, image_size=2, resize_mode="stretch", cache=cache)
    temporary_path = Path(cache._temporary.name)
    assert any(temporary_path.iterdir())
    load_cxr_image(path, image_size=3, resize_mode="stretch", cache=cache)
    assert cache.stats.evictions == 1
    cache.close()
    assert not temporary_path.exists()


def test_manifest_audit_reports_missing_data(tmp_path):
    records = [
        {"patient_id": "P001", "study_id": "S001", "domain": "source", "image_path": str(tmp_path / "missing.dcm"), "label": 1, "age": 40, "gender": "m"},
        {"patient_id": "P002", "study_id": "S002", "domain": "target", "image_path": str(tmp_path / "missing2.dcm"), "label": None, "age": None, "gender": None},
    ]

    summary = audit_manifest(records)

    assert summary["records"] == 2
    assert summary["unique_patients"] == 2
    assert summary["domains"] == {"source": 1, "target": 1}
    assert summary["labels"] == {"positive": 1, "negative": 0, "missing": 1}
    assert summary["missing_image_paths"] == 2
    assert summary["missing_ages"] == 1
    assert summary["missing_genders"] == 1
