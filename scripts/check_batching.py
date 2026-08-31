import argparse

from daxray.data import build_cxr_rait_manifest, iter_batches, load_split_manifest
from daxray.evaluation import save_batch_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a labeled CXR-RAIT batch grid.")
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--resize-mode", choices=("pad", "stretch"), default="pad")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts/cxr_rait/batch_train_seed7.png")
    args = parser.parse_args()

    records = build_cxr_rait_manifest("gs://cxr-rait/cxr-demography-data")
    split = load_split_manifest("artifacts/cxr_rait/split_manifest_seed7.json")
    batch = next(iter_batches(
        records,
        split["splits"][args.split],
        batch_size=args.batch_size,
        image_size=args.image_size,
        resize_mode=args.resize_mode,
        shuffle=args.split == "train",
        seed=args.seed,
        as_jax=True,
    ))
    output_path = save_batch_grid(
        batch,
        args.output,
        columns=args.columns,
        title=f"DAXRay {args.split} batch",
    )
    print(f"Images: {batch['image'].shape}, dtype={batch['image'].dtype}")
    print(f"Labels: {batch['label'].shape}, labeled={int(batch['label_mask'].sum())}/{len(batch['label_mask'])}")
    print(f"Grid: {output_path}")


if __name__ == "__main__":
    main()
