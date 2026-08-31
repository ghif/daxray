from daxray.data import (
    SplitConfig,
    audit_manifest,
    build_cxr_rait_manifest,
    create_split_manifest,
    save_split_manifest,
)


# def main() -> None:
records = build_cxr_rait_manifest("gs://cxr-rait/cxr-demography-data")
split_manifest = create_split_manifest(records, SplitConfig(seed=7))
output_path = "artifacts/cxr_rait/split_manifest_seed7.json"
save_split_manifest(split_manifest, output_path)
summary = audit_manifest(records, split_manifest)
print(f"Matched records: {summary['records']}")
print(f"Unique patients: {summary['unique_patients']}")
print(f"Labels: {summary['labels']}")
print(f"Domains: {summary['domains']}")
print(f"Splits: {summary['split_counts']}")
print(f"Missing image paths: {summary['missing_image_paths']}")
print(f"Manifest: {output_path}")


# if __name__ == "__main__":
#     main()
