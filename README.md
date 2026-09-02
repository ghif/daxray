# DAXRay

Domain-adaptive chest X-ray image classification for tuberculosis screening.

DAXRay is a research codebase for developing and evaluating robust CXR classifiers in the presence of domain mismatch. The project focuses on the CXR-RAIT image-classification task and is intended to support models that can generalize across hospitals, datasets, acquisition protocols, and population groups.

## Motivation

Chest X-ray datasets often differ substantially across collection sites. Differences in scanners, preprocessing, clinical workflows, patient populations, and labeling practices can reduce the performance of a model outside its training domain. DAXRay explores domain-adaptation and domain-generalization methods to make CXR-based TB classification more reliable in real-world settings, including Indonesian use cases.

## Project goals

- Establish a reproducible baseline for CXR TB classification.
- Measure performance under domain shift and domain mismatch.
- Evaluate domain-adaptation and domain-generalization approaches.
- Keep experiments modular so that models, datasets, and training backends can evolve over time.
- Report clinically relevant metrics alongside overall accuracy.

## Status

Early development. Training pipelines, datasets, model architectures, and evaluation protocols are still being defined.

## Planned components

- Dataset preparation and validation
- CXR preprocessing and augmentation
- Baseline image-classification models
- Domain-shift and cross-domain evaluation
- Domain-adaptation methods
- Experiment configuration and reproducibility utilities
- Metrics, visualizations, and error analysis

## Getting started

Install the package with its metadata and JAX extras:

```bash
conda run -n med-jax python -m pip install -r requirements.txt
```

For a Google Cloud TPU VM, use the TPU-specific requirements file instead:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements-tpu.txt
```

The file installs the Google TPU JAX extra and the matching `libtpu` wheel
source. For a multi-VM TPU slice, install it on every worker (for example with
`gcloud compute tpus tpu-vm ssh --worker=all`). Verify the installation before
training:

```bash
python3 -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

The expected backend is `tpu`. The standard `requirements.txt` remains the
local CPU/GPU development environment.

Build a patient-level manifest from the authoritative CXR-RAIT GCS location:

```python
from daxray.data import SplitConfig, build_cxr_rait_manifest, create_split_manifest, save_split_manifest

records = build_cxr_rait_manifest("gs://cxr-rait/cxr-demography-data")
split_manifest = create_split_manifest(records, SplitConfig(seed=7))
save_split_manifest(split_manifest, "artifacts/cxr_rait/split_manifest_seed7.json")
```

The split manifest is stratified by TB label, gender, and coarse age bin. All
images belonging to the same patient stay in one split. Batches can then be
created from a split’s patient IDs with `iter_batches`; the image tensor is
returned as `float32` in `(N, 1, H, W)` format and can optionally be placed on a
JAX device with `as_jax=True`.

## Manual data-processing run

The current data pipeline is library-driven and does not yet have a dedicated
command-line interface. Run the following commands from this directory. All
Python commands must use the `med-jax` environment and `PYTHONPATH=src`.

Install the package:

```bash
conda run -n med-jax python -m pip install -r requirements.txt
```

Create and save a split manifest from a local CXR-RAIT directory:

```bash
conda run -n med-jax env PYTHONPATH=src python - <<'PY'
from daxray.data import (
    SplitConfig,
    build_cxr_rait_manifest,
    create_split_manifest,
    save_split_manifest,
)

records = build_cxr_rait_manifest("gs://cxr-rait/cxr-demography-data")
manifest = create_split_manifest(records, SplitConfig(seed=7))
save_split_manifest(manifest, "artifacts/cxr_rait/split_manifest_seed7.json")
print({name: len(ids) for name, ids in manifest["splits"].items()})
PY
```

Audit the generated records and split:

```bash
conda run -n med-jax env PYTHONPATH=src python - <<'PY'
from daxray.data import audit_manifest, build_cxr_rait_manifest, load_split_manifest

records = build_cxr_rait_manifest("gs://cxr-rait/cxr-demography-data")
split = load_split_manifest("artifacts/cxr_rait/split_manifest_seed7.json")
print(audit_manifest(records, split))
PY
```

Load one JAX-ready batch:

```bash
conda run -n med-jax env PYTHONPATH=src python - <<'PY'
from daxray.data import build_cxr_rait_manifest, iter_batches, load_split_manifest

records = build_cxr_rait_manifest("/path/to/cxr-demography-data")
split = load_split_manifest("artifacts/cxr_rait/split_manifest_seed7.json")
batch = next(iter_batches(
    records,
    split["splits"]["train"],
    batch_size=4,
    image_size=224,
    resize_mode="pad",
    shuffle=True,
    seed=7,
    as_jax=True,
))
print(batch["image"].shape, batch["image"].dtype)
print(batch["label"].shape, batch["label_mask"].shape)
PY
```

Create a grid preview of a batch:

```bash
conda run -n med-jax env PYTHONPATH=src python -m scripts.check_batching \
  --split train \
  --batch-size 8 \
  --columns 4 \
  --output artifacts/cxr_rait/batch_train_seed7.png
```

The grid is saved locally and includes the patient ID, TB label, and label
availability for each image. The source images are always read from
`gs://cxr-rait/cxr-demography-data`; the output path must remain outside the
read-only dataset bucket.

Training uses a bounded per-process preprocessing cache by default. DICOM
bytes are read and decoded in memory, then the normalized resized tensor is
cached for reuse across epochs. No raw DICOM files or persistent cache files
are written to the local drive. Configure this under `dataset.cache` with
`mode: memory`, `mode: none`, or the cleaned-up `mode: ephemeral` fallback;
`max_bytes` limits cache usage and `read_workers` controls concurrent initial
GCS reads. Cache hit rate, read/preprocessing time, and evictions are printed
in the progress log and recorded in TensorBoard. Remote logs synchronize at
the configured workflow interval, while Orbax checkpoint uploads run
asynchronously and are awaited before final results are written.

Train the first learned sanity-check baseline using metadata only:

```bash
conda run -n med-jax env PYTHONPATH=src python -m scripts.train_metadata_baseline
```

This logistic-regression baseline uses age and gender, not image pixels. It
reports accuracy, balanced accuracy, AUROC, AUPRC, sensitivity, specificity,
and F1 for the train, validation, and test splits. Results are written to
`artifacts/cxr_rait/metadata_baseline.json`.

### Evaluate Faster R-CNN Zero-Shot Transfer on CXR-RAIT

To run the zero-shot evaluation of the TBX11K-trained Faster R-CNN (ResNet-50-FPN-V2) on CXR-RAIT:

```bash
conda run -n med-jax env PYTHONPATH=src \
  python scripts/evaluate_faster_rcnn_cxr_rait.py \
  --manifest artifacts/cxr_rait/split_manifest_seed7.json \
  --split all \
  --threshold 0.001 \
  --output-dir artifacts/cxr_rait_zero_shot \
  --bootstrap-samples 1000 \
  --save-galleries 5
```

For manifest auditing and verification without model inference:

```bash
conda run -n med-jax env PYTHONPATH=src \
  python scripts/evaluate_faster_rcnn_cxr_rait.py \
  --manifest artifacts/cxr_rait/split_manifest_seed7.json \
  --split all \
  --dry-run
```

For locked evaluation with threshold tuning on the validation split:

```bash
conda run -n med-jax env PYTHONPATH=src \
  python scripts/evaluate_faster_rcnn_cxr_rait.py \
  --manifest artifacts/cxr_rait/split_manifest_seed7.json \
  --split test \
  --tune-threshold \
  --tune-split validation \
  --tuning-criterion max_f1 \
  --locked \
  --output-dir artifacts/cxr_rait_zero_shot_test
```

Outputs are written to the target directory:
- `predictions.json`: Per-image predictions and bounding box detections.
- `metrics.json`: Binary TB transfer metrics with patient bootstrap 95% CIs.
- `audit.json`: Dataset manifest audit and site breakdowns.
- `galleries/`: Stratified visualization previews.

### Train the image classifier

Run the classifier from the `daxray` directory. The command must use the
`med-jax` Conda environment and `PYTHONPATH=src`:

```bash
cd /Users/mghifary/Work/Code/AI/medical-tpu/daxray
conda run -n med-jax env PYTHONPATH=src \
  python -m scripts.train_cnn_baseline \
  --config configs/cxr_rait/cnn_baseline.yaml \
  --checkpoint-name cnn_baseline_cpu_31-08-2026
```

The YAML file contains the dataset, model, optimizer, runtime, workflow, and
artifact settings. The checkpoint name identifies the run directory and must
be unique. To continue an existing run explicitly, add `--resume`:

```bash
conda run -n med-jax env PYTHONPATH=src \
  python -m scripts.train_cnn_baseline \
  --config configs/cxr_rait/cnn_baseline.yaml \
  --checkpoint-name cnn_baseline_cpu_31-08-2026 \
  --resume
```

The `runtime` section selects the JAX backend and execution mode:

```yaml
runtime:
  accelerator: cpu       # cpu, gpu, or tpu
  precision: fp32
  execution_mode: auto   # auto, single_device, or multi_device
```

`auto` uses one device when only one is available and all local devices when
multiple devices are available. Set `single_device` for one CPU, GPU, or TPU
core. Set `multi_device` to require data-parallel execution across all local
devices; the global batch is split across devices and padded safely when the
final batch is incomplete. The requested backend is validated at startup, so a
missing GPU or TPU runtime fails clearly instead of silently falling back to
CPU.

Before starting a full run, validate data access, model initialization, and
one training update without creating checkpoints or run artifacts:

```bash
conda run -n med-jax env PYTHONPATH=src \
  python -m scripts.train_cnn_baseline \
  --config configs/cxr_rait/cnn_baseline.yaml \
  --checkpoint-name dry-run-validation \
  --dry-run
```

The model is a compact three-block CNN. Each block uses a 3×3 convolution,
LayerNorm, ReLU, and average pooling, followed by global average pooling,
dropout, and a single TB logit. The model receives NHWC tensors at the Flax
NNX boundary. Each named run stores its resolved configuration, `trainlog.txt`,
TensorBoard events, `results.json`, and Orbax full-state checkpoints under
`gs://cxr-rait/checkpoints/classifier/<checkpoint-name>/`. Only the three
highest validation-AUROC checkpoints are retained. Existing run directories
fail by default; pass `--resume` to continue from the latest retained
checkpoint.

After training, open the TensorBoard curves with:

```bash
conda run -n med-jax tensorboard \
  --logdir gs://cxr-rait/checkpoints/classifier/cnn_baseline_cpu_31-08-2026/tensorboard
```

All durable artifacts are stored under the run directory:

```text
gs://cxr-rait/checkpoints/classifier/cnn_baseline_cpu_31-08-2026/
├── config.yaml
├── trainlog.txt
├── tensorboard/
├── results.json
└── orbax/                 # top 3 validation-AUROC checkpoints
```

The expected metadata file is `data_demography.xlsx`, and DICOM files are
discovered recursively by patient ID from their filenames. The
`gs://cxr-rait/cxr-demography-data/` is read-only source data. Checkpoints and
run artifacts belong under `gs://cxr-rait/checkpoints/classifier/`; do not
write split manifests, temporary files, caches, logs, or other derived
artifacts into the dataset prefix. Do not
commit the dataset, restricted metadata, split artifacts containing sensitive
identifiers, or preprocessing caches.

## Data

This repository does not include medical images or other restricted datasets. Users are responsible for obtaining authorized access to the datasets used in their experiments and for complying with the relevant data-use, privacy, and ethics requirements.

## Disclaimer

DAXRay is research software. It is not a medical device and must not be used to make clinical diagnoses or treatment decisions.

## License

License information will be added before the first public release.
