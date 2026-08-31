"""Run the configurable CXR-RAIT CNN baseline."""

import argparse

from daxray.config import load_cnn_config
from daxray.runtime import configure_backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cxr_rait/cnn_baseline.yaml")
    parser.add_argument("--checkpoint-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Load one batch and execute one update without writing artifacts.")
    args = parser.parse_args()
    config = load_cnn_config(args.config, {"checkpoint_name": args.checkpoint_name})
    configure_backend(config.runtime.accelerator)
    from daxray.training.cnn_baseline import run_cnn_baseline

    result = run_cnn_baseline(config, resume=args.resume, dry_run=args.dry_run)
    print(f"Run directory: {result['run_directory']}")
    if args.dry_run:
        print("Dry run completed successfully.")
    else:
        print(f"Best checkpoint step: {result['best_checkpoint_step']}")


if __name__ == "__main__":
    main()
