import pytest

from daxray.data import SplitConfig, create_split_manifest, dataset_fingerprint, load_split_manifest, save_split_manifest, validate_split_manifest


def _records(n=60):
    return [
        {
            "patient_id": f"P{i:03d}",
            "image_path": f"/images/P{i:03d}.dcm",
            "label": i % 2,
            "gender": "m" if i % 2 else "f",
            "age": 20 + (i % 50),
        }
        for i in range(n)
    ]


def test_split_is_deterministic_and_patient_disjoint():
    records = _records()
    first = create_split_manifest(records, SplitConfig(seed=17))
    second = create_split_manifest(records, SplitConfig(seed=17))
    assert first == second
    validate_split_manifest(first, [record["patient_id"] for record in records])
    assert set(first["splits"]["train"]).isdisjoint(first["splits"]["validation"])
    assert set(first["splits"]["validation"]).isdisjoint(first["splits"]["test"])


def test_seed_changes_split():
    records = _records()
    first = create_split_manifest(records, SplitConfig(seed=1))
    second = create_split_manifest(records, SplitConfig(seed=2))
    assert first["splits"] != second["splits"]


def test_invalid_split_config():
    with pytest.raises(ValueError):
        SplitConfig(train=0.8, validation=0.2, test=0.2)


def test_fingerprint_changes_with_records():
    records = _records()
    assert dataset_fingerprint(records) != dataset_fingerprint(records[:-1])


def test_manifest_round_trip(tmp_path):
    records = _records()
    manifest = create_split_manifest(records)
    path = tmp_path / "split.json"
    save_split_manifest(manifest, path)
    assert load_split_manifest(path) == manifest


def test_manifest_rejects_non_list_split():
    manifest = create_split_manifest(_records())
    manifest["splits"]["test"] = tuple(manifest["splits"]["test"])
    with pytest.raises(ValueError, match="must be a list"):
        validate_split_manifest(manifest)


def test_manifest_cannot_be_saved_inside_cxr_rait_bucket(tmp_path):
    manifest = create_split_manifest(_records())
    with pytest.raises(ValueError, match="read-only"):
        save_split_manifest(manifest, "gs://cxr-rait/cxr-demography-data/split.json")
