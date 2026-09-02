# Repository Guidelines

## Execution Environment

Any Python script, test, formatter, linter, build command, or notebook in this
repository must be executed inside the `med-jax` conda environment. The
repository dependency contract is `requirements.txt`; do not add or use
`pyproject.toml` for dependency management. Prefer explicit environment
selection for reproducibility:

```bash
conda run -n med-jax python -m pip install -r requirements.txt
conda run -n med-jax python <script.py>
conda run -n med-jax env PYTHONPATH=src python -m pytest -q
conda run -n med-jax ruff check src tests
```

Do not run repository Python commands with the system interpreter or another
conda environment. Before assuming a dependency is available, verify it with
`conda run -n med-jax python -c "import <package>"`.

## Project Structure

Use the following architecture as the default structure as the repository
grows:

```text
src/daxray/
├── data/          # metadata, manifests, splits, DICOM, and batching
├── models/        # model definitions and model-specific components
├── training/      # optimization and training workflows
├── inference/     # prediction workflows and inference utilities
├── evaluation/    # metrics, reports, and analysis
└── config.py      # shared configuration models
scripts/           # thin command-line entry points
tests/             # unit and integration tests
```

Keep reusable implementation code under `src/daxray/`. Dataset discovery,
metadata handling, preprocessing, split manifests, and batch iteration belong
in the single `data/` package, divided into focused modules as needed. Do not
create deeper data subdirectories unless the repository size makes the
boundary useful. Keep training, inference, evaluation, experiment scripts,
configuration files, and generated artifacts separate from data-processing
code.

## Clean Code Architecture

Follow clean-code architecture principles throughout the repository:

- Keep domain logic independent from infrastructure such as filesystems,
  cloud storage, JAX device placement, logging, and command-line interfaces.
- Separate dataset discovery, metadata parsing, split generation, image
  preprocessing, batching, training, and evaluation into cohesive modules.
- Keep dependencies flowing inward toward stable domain contracts; avoid
  importing training, experiment, or infrastructure code into low-level data
  structures.
- Prefer small functions with one clear responsibility and explicit inputs and
  outputs. Avoid hidden global state, implicit configuration, and duplicated
  preprocessing logic.
- Make side effects visible at boundaries. Pure transformations should remain
  deterministic and easy to unit test.
- Define stable interfaces for datasets, manifests, batches, and metrics so
  implementations can change without rewriting experiment code.
- Do not add abstractions speculatively. Introduce an interface when it
  protects a real boundary, supports a planned backend/domain, or improves
  testability.
- Keep research experiments reproducible and configurable without embedding
  experiment-specific decisions in reusable library modules.

## Data and Splitting

- The authoritative CXR-RAIT dataset location is Google Cloud Storage under
  `gs://cxr-rait/cxr-demography-data/`; do not replace it with a local staging
  path in configs, scripts, or documentation.
- Treat `gs://cxr-rait/cxr-demography-data/` as read-only source data. Store
  derived training artifacts under the explicitly named
  `gs://cxr-rait/checkpoints/classifier/` prefix, in a unique run directory.
- Split by patient, never by image or individual DICOM file.
- Keep all studies, views, and repeated examinations from one patient in the
  same split.
- Persist and validate the exact split manifest used by an experiment.
- Treat validation and test data as non-training data; do not use test labels
  for threshold selection, tuning, pseudo-labeling, or early stopping.
- Preserve dataset and domain provenance for future cross-domain experiments.
- Never commit medical images, restricted metadata, credentials, or generated
  caches.

## Coding Conventions

Use Python 3.10+ type hints, four-space indentation, `snake_case` for functions
and modules, and `PascalCase` for classes. Keep preprocessing deterministic
unless randomness is explicitly passed through a seed or random-number
generator. Prefer NumPy arrays at data-loading boundaries and make JAX device
placement explicit at batch boundaries.

Do not silently replace unreadable or missing medical images with synthetic
blank images. Raise an informative error that includes the patient ID and path.

## Testing and Validation

Run the relevant checks from the repository root using `med-jax`:

```bash
conda run -n med-jax env PYTHONPATH=src python -m pytest -q
conda run -n med-jax ruff check src tests
```

Changes to splitting must test determinism, patient disjointness, complete
patient coverage, and manifest validation. Changes to image preprocessing
should include tests for shape, dtype, intensity range, DICOM polarity, and
resize behavior where practical.

## Research Reproducibility

Record the random seed, dataset fingerprint, preprocessing settings, split
manifest path, model input size, and source/target domain roles for every
experiment. Keep the held-out test set locked until the final evaluation.
Report AUROC, AUPRC, sensitivity, specificity, calibration, and subgroup
metrics where labels and sample sizes permit; accuracy alone is insufficient
for imbalanced TB classification.

## Agent Workflow

Before editing, inspect the affected modules and tests. Make focused changes,
preserve unrelated user work, and run the narrowest relevant tests followed by
the full available suite. Summarize changed files, commands run, and any
dataset or environment limitation in the final handoff.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
