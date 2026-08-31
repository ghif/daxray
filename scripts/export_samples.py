"""Export a small labeled CXR-RAIT sample set for manual inspection."""

import argparse
import csv
import os

import numpy as np
from PIL import Image

from daxray.data import build_cxr_rait_manifest, iter_batches, load_split_manifest
from daxray.evaluation import save_batch_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count-per-class", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--output-dir", default="samples/cxr_rait")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.count_per_class <= 0:
        raise ValueError("--count-per-class must be positive.")

    records = build_cxr_rait_manifest("gs://cxr-rait/cxr-demography-data")
    split = load_split_manifest("artifacts/cxr_rait/split_manifest_seed7.json")
    train_ids = set(split["splits"]["train"])
    train_records = [record for record in records if record["patient_id"] in train_ids]
    selected = []
    for label, label_name in ((1, "tb_positive"), (0, "tb_negative")):
        candidates = [record for record in train_records if record["label"] == label]
        if len(candidates) < args.count_per_class:
            raise ValueError(f"Only {len(candidates)} records available for {label_name}.")
        selected.extend(candidates[: args.count_per_class])

    os.makedirs(args.output_dir, exist_ok=True)
    index_rows = []
    for index, record in enumerate(selected, start=1):
        batch = next(iter_batches(
            [record],
            [record["patient_id"]],
            batch_size=1,
            image_size=args.image_size,
            resize_mode="pad",
            as_jax=False,
        ))
        label_name = "tb_positive" if record["label"] == 1 else "tb_negative"
        sample_name = f"{label_name}_{index:02d}.png"
        image = np.asarray(batch["image"][0, 0] * 255.0, dtype=np.uint8)
        Image.fromarray(image, mode="L").save(os.path.join(args.output_dir, sample_name))
        index_rows.append({"sample": sample_name, "label": label_name, "age": record.get("age"), "gender": record.get("gender")})

    index_path = os.path.join(args.output_dir, "index.csv")
    with open(index_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", "label", "age", "gender"])
        writer.writeheader()
        writer.writerows(index_rows)

    grid_batch = {
        "image": np.stack([
            np.asarray(Image.open(os.path.join(args.output_dir, row["sample"])), dtype=np.float32)[None, :, :] / 255.0
            for row in index_rows
        ]),
        "label": np.asarray([1 if row["label"] == "tb_positive" else 0 for row in index_rows], dtype=np.int32),
        "label_mask": np.ones(len(index_rows), dtype=bool),
        "patient_id": [row["sample"].removesuffix(".png") for row in index_rows],
    }
    grid_path = os.path.join(args.output_dir, "grid.png")
    save_batch_grid(grid_batch, grid_path, columns=args.count_per_class, title="CXR-RAIT manual samples")
    print(f"Saved {len(index_rows)} images to {args.output_dir}")
    print(f"Index: {index_path}")
    print(f"Grid: {grid_path}")


if __name__ == "__main__":
    main()
