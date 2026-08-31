"""Train and evaluate the metadata-only CXR-RAIT baseline."""

import argparse
import json
import os

from daxray.data import build_cxr_rait_manifest, load_split_manifest
from daxray.evaluation.metrics import classification_metrics
from daxray.models import fit_logistic_classifier
from daxray.training.metadata_baseline import FEATURE_NAMES, metadata_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2-penalty", type=float, default=1e-3)
    parser.add_argument("--output", default="artifacts/cxr_rait/metadata_baseline.json")
    args = parser.parse_args()

    records = build_cxr_rait_manifest("gs://cxr-rait/cxr-demography-data")
    split = load_split_manifest("artifacts/cxr_rait/split_manifest_seed7.json")
    by_id = {record["patient_id"]: record for record in records}

    split_records = {
        name: [by_id[patient_id] for patient_id in patient_ids]
        for name, patient_ids in split["splits"].items()
    }
    train_x, train_y = metadata_features(split_records["train"])
    model, history = fit_logistic_classifier(
        train_x,
        train_y,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        l2_penalty=args.l2_penalty,
    )
    metrics = {}
    for name, records_for_split in split_records.items():
        features, labels = metadata_features(records_for_split)
        metrics[name] = classification_metrics(labels, model.predict_proba(features))

    result = {
        "model": "metadata_logistic_regression",
        "features": list(FEATURE_NAMES),
        "dataset": "cxr_rait",
        "manifest": "artifacts/cxr_rait/split_manifest_seed7.json",
        "dataset_fingerprint": split.get("dataset_fingerprint"),
        "hyperparameters": {"epochs": args.epochs, "learning_rate": args.learning_rate, "l2_penalty": args.l2_penalty},
        "final_training_loss": history[-1],
        "metrics": metrics,
        "parameters": model.to_dict(),
    }
    parent = os.path.dirname(args.output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=True)
    for name, values in metrics.items():
        print(f"{name}: accuracy={values['accuracy']:.4f}, balanced_accuracy={values['balanced_accuracy']:.4f}, auroc={values['auroc']:.4f}, auprc={values['auprc']:.4f}")
    print(f"Results: {args.output}")


if __name__ == "__main__":
    main()
