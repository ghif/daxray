"""Training loop for the small Flax NNX CXR classifier."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Iterator
import json
import time

import fsspec
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
import yaml

from daxray.config import CnnExperimentConfig
from daxray.data import build_cxr_rait_manifest, iter_batches, load_split_manifest
from daxray.evaluation.metrics import classification_metrics
from daxray.models.cnn import CxrSmallCNN, binary_cross_entropy_with_logits
from daxray.runtime import validate_backend
from .checkpointing import TopKCheckpointManager
from .logging import RunLogger


def train_cnn(
    model: CxrSmallCNN,
    optimizer: nnx.Optimizer,
    batches: Iterator[dict[str, Any]],
    *,
    devices: list[jax.Device] | None = None,
    progress_callback: Callable[[int, float, float], None] | None = None,
) -> float:
    """Run one epoch over NHWC batches and return mean training loss."""
    if devices and len(devices) > 1:
        return _train_cnn_multi_device(model, optimizer, batches, devices, progress_callback)
    losses = []
    for batch_index, batch in enumerate(batches, start=1):
        batch_started = time.perf_counter()
        images = jnp.asarray(batch["image"], dtype=jnp.float32)
        mask = jnp.asarray(batch.get("label_mask", np.ones(len(batch["label"]), dtype=bool)))
        labels = jnp.asarray(batch["label"], dtype=jnp.float32)
        if not bool(np.any(np.asarray(mask))):
            continue
        images = images[mask]
        labels = labels[mask]

        def loss_fn(current_model: CxrSmallCNN) -> jnp.ndarray:
            return binary_cross_entropy_with_logits(current_model(images, training=True), labels)

        loss, gradients = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, gradients)
        loss_value = float(loss)
        losses.append(loss_value)
        if progress_callback is not None:
            progress_callback(batch_index, loss_value, time.perf_counter() - batch_started)
    if not losses:
        raise ValueError("Cannot train on batches without labeled examples.")
    return float(np.mean(losses))


def _train_cnn_multi_device(
    model: CxrSmallCNN,
    optimizer: nnx.Optimizer,
    batches: Iterator[dict[str, Any]],
    devices: list[jax.Device],
    progress_callback: Callable[[int, float, float], None] | None = None,
) -> float:
    """Run replicated data-parallel updates across local devices."""
    device_count = len(devices)
    state_axes = nnx.StateAxes({nnx.Param: None, ...: None})

    @nnx.split_rngs(splits=device_count)
    @nnx.pmap(axis_name="devices", in_axes=(state_axes, None, 0, 0, 0), out_axes=0, devices=devices)
    def step(current_model, current_optimizer, images, labels, mask):
        def loss_fn(candidate_model):
            logits = candidate_model(images, training=True)
            losses = jnp.maximum(logits, 0.0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))
            return jnp.sum(losses * mask) / jnp.maximum(jnp.sum(mask), 1.0)

        loss, gradients = nnx.value_and_grad(loss_fn)(current_model)
        gradients = jax.lax.pmean(gradients, "devices")
        loss = jax.lax.pmean(loss, "devices")
        current_optimizer.update(current_model, gradients)
        return loss

    losses = []
    for batch_index, batch in enumerate(batches, start=1):
        batch_started = time.perf_counter()
        images = np.asarray(batch["image"], dtype=np.float32)
        labels = np.asarray(batch["label"], dtype=np.float32)
        mask = np.asarray(batch.get("label_mask", np.ones(len(labels), dtype=bool)), dtype=bool)
        if not np.any(mask):
            continue
        remainder = len(images) % device_count
        if remainder:
            padding = device_count - remainder
            images = np.pad(images, ((0, padding), (0, 0), (0, 0), (0, 0)))
            labels = np.pad(labels, (0, padding), constant_values=-1)
            mask = np.pad(mask, (0, padding), constant_values=False)
        per_device = len(images) // device_count
        value = step(
            model,
            optimizer,
            jnp.asarray(images).reshape(device_count, per_device, *images.shape[1:]),
            jnp.asarray(labels).reshape(device_count, per_device),
            jnp.asarray(mask, dtype=jnp.float32).reshape(device_count, per_device),
        )
        loss_value = float(np.asarray(value)[0])
        losses.append(loss_value)
        if progress_callback is not None:
            progress_callback(batch_index, loss_value, time.perf_counter() - batch_started)
    if not losses:
        raise ValueError("Cannot train on batches without labeled examples.")
    return float(np.mean(losses))


def create_cnn_optimizer(model: CxrSmallCNN, learning_rate: float = 1e-3, weight_decay: float = 1e-4) -> nnx.Optimizer:
    """Create the AdamW optimizer used by the baseline."""
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("learning_rate must be positive and weight_decay cannot be negative.")
    return nnx.Optimizer(model, optax.adamw(learning_rate, weight_decay=weight_decay), wrt=nnx.Param)


def _batches(records, patient_ids, config: CnnExperimentConfig, *, shuffle: bool, seed: int):
    return iter_batches(records, patient_ids, batch_size=config.batch_size, image_size=config.image_size,
                        resize_mode=config.resize_mode, shuffle=shuffle, seed=seed, layout="NHWC")


def _evaluate(model, records, patient_ids, config: CnnExperimentConfig) -> dict[str, float | int]:
    model.eval()
    probabilities, labels = [], []
    for batch in _batches(records, patient_ids, config, shuffle=False, seed=config.seed):
        mask = np.asarray(batch["label_mask"], dtype=bool)
        if not np.any(mask):
            continue
        logits = model(jnp.asarray(batch["image"]), training=False)
        probabilities.extend(np.asarray(jax.nn.sigmoid(logits))[mask])
        labels.extend(np.asarray(batch["label"])[mask])
    return classification_metrics(np.asarray(labels), np.asarray(probabilities))


def _write_text(path: str, content: str) -> None:
    fs, name = fsspec.core.url_to_fs(path)
    parent = name.rsplit("/", 1)[0] if "/" in name else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(name, "w", encoding="utf-8") as handle:
        handle.write(content)


def run_cnn_baseline(config: CnnExperimentConfig, *, resume: bool = False, dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        return _run_dry_run(config)
    fs, run_name = fsspec.core.url_to_fs(config.run_directory)
    if fs.exists(run_name) and not resume:
        raise FileExistsError(f"Checkpoint directory already exists: {config.run_directory}")
    fs.makedirs(run_name, exist_ok=True)
    _write_text(config.artifact_path("config.yaml"), yaml.safe_dump(config.to_dict(), sort_keys=False))
    logger = RunLogger(config.artifact_path("trainlog.txt"), config.artifact_path("tensorboard"))
    manager = TopKCheckpointManager(config.run_directory, keep_top_k=config.checkpoint_keep_top_k)
    try:
        runtime = validate_backend(config.runtime.accelerator)
        if config.runtime.execution_mode == "single_device":
            devices = runtime_devices = jax.local_devices()[:1]
        elif config.runtime.execution_mode == "multi_device":
            devices = runtime_devices = list(jax.local_devices())
        else:
            devices = runtime_devices = list(jax.local_devices())
        if not devices:
            raise RuntimeError("JAX reported no local devices.")
        logger.log({"event": "run_started", "run_directory": config.run_directory, "checkpoint_name": config.checkpoint_name,
                    "epochs": config.epochs, "batch_size": config.batch_size, "resume": resume,
                    "backend": runtime.backend, "local_device_count": runtime.local_device_count,
                    "global_device_count": runtime.global_device_count, "process_count": runtime.process_count,
                    "execution_device_count": len(runtime_devices)})
        records = build_cxr_rait_manifest(config.dataset_root)
        split = load_split_manifest(config.split_manifest)
        logger.log({"event": "dataset_loaded", "records": len(records),
                    "train_records": len(split["splits"]["train"]), "validation_records": len(split["splits"]["validation"]),
                    "test_records": len(split["splits"]["test"]), "dataset_fingerprint": split.get("dataset_fingerprint")})
        model = CxrSmallCNN(rngs=nnx.Rngs(config.seed), dropout_rate=config.dropout_rate)
        optimizer = create_cnn_optimizer(model, config.learning_rate, config.weight_decay)
        start_epoch = 0
        if resume:
            restored = manager.restore()
            if restored is not None:
                nnx.update(model, restored["model"])
                nnx.update(optimizer, restored["optimizer"])
                start_epoch = int(restored["epoch"])
        history = []
        for epoch in range(start_epoch + 1, config.epochs + 1):
            started = datetime.now(timezone.utc)
            samples_per_epoch = len(split["splits"]["train"])
            total_batches = (samples_per_epoch + config.batch_size - 1) // config.batch_size
            logger.log({"event": "epoch_started", "epoch": epoch, "total_epochs": config.epochs,
                        "total_batches": total_batches, "samples_per_epoch": samples_per_epoch,
                        "message": "loading first training batch"})
            model.train()
            epoch_t0 = time.perf_counter()
            global_batches_before = (epoch - 1) * total_batches

            def batch_progress(batch_index: int, batch_loss: float, batch_seconds: float) -> None:
                if batch_index != 1 and batch_index != total_batches and batch_index % config.workflow.speed_log_freq != 0:
                    return
                completed = global_batches_before + batch_index
                total_batches_all = config.epochs * total_batches
                samples_processed = min(batch_index * config.batch_size, samples_per_epoch)
                elapsed = time.perf_counter() - epoch_t0
                iter_per_sec = batch_index / max(elapsed, 1e-12)
                sample_per_sec = batch_index * config.batch_size / max(elapsed, 1e-12)
                epoch_samples_per_sec = samples_processed / max(elapsed, 1e-12)
                eta = max(0, total_batches_all - completed) / max(iter_per_sec, 1e-12)
                logger.log({"event": "train_batch_progress", "epoch": epoch, "total_epochs": config.epochs,
                            "batch": batch_index, "total_batches": total_batches, "global_batch": completed,
                            "total_train_batches": total_batches_all, "samples_per_epoch": samples_per_epoch,
                            "samples_processed": samples_processed, "batch_loss": batch_loss,
                            "batch_seconds": batch_seconds, "iter_per_sec": iter_per_sec,
                            "sample_per_sec": sample_per_sec, "epoch_samples_per_sec": epoch_samples_per_sec,
                            "epoch_elapsed_seconds": elapsed,
                            "eta_seconds": eta}, sync_remote=False)

            loss = train_cnn(
                model,
                optimizer,
                _batches(records, split["splits"]["train"], config, shuffle=True, seed=config.seed + epoch),
                devices=devices,
                progress_callback=batch_progress,
            )
            validation = _evaluate(model, records, split["splits"]["validation"], config)
            state = {"model": nnx.state(model), "optimizer": nnx.state(optimizer), "epoch": epoch, "rng": {"seed": config.seed}}
            saved = manager.save(epoch, state, float(validation["auroc"]))
            retained = manager.retained()
            event = {"timestamp": started.isoformat(), "timestamp_unix": started.timestamp(), "epoch": epoch,
                     "train_loss": loss, "validation_auroc": validation["auroc"], "validation_auprc": validation["auprc"],
                     "validation_accuracy": validation["accuracy"], "validation_f1": validation["f1"],
                     "validation_sensitivity": validation["sensitivity"], "validation_specificity": validation["specificity"],
                     "learning_rate": config.learning_rate, "checkpoint_saved": saved,
                     "samples_per_epoch": samples_per_epoch,
                     "retained_checkpoint_steps": [item.step for item in retained],
                     "retained_checkpoint_count": len(retained),
                     "best_validation_auroc": max((item.auroc for item in retained), default=float("nan")),
                     "epoch_seconds": (datetime.now(timezone.utc) - started).total_seconds()}
            event["epoch_samples_per_sec"] = samples_per_epoch / max(event["epoch_seconds"], 1e-12)
            logger.log(event)
            history.append({"epoch": epoch, "train_loss": loss, "validation": validation})
        best_step = manager.best_step
        if best_step is not None:
            best_state = manager.restore(best_step)
            nnx.update(model, best_state["model"])
        metrics = {name: _evaluate(model, records, ids, config) for name, ids in split["splits"].items()}
        retained = manager.retained()
        result = {"model": "cxr_small_cnn", "dataset": "cxr_rait", "checkpoint_name": config.checkpoint_name,
                  "run_directory": config.run_directory, "best_checkpoint_step": best_step,
                  "retained_checkpoints": [item.__dict__ for item in retained], "manifest": config.split_manifest,
                  "dataset_fingerprint": split.get("dataset_fingerprint"), "runtime": runtime.__dict__, "resolved_config": config.to_dict(),
                  "artifacts": {"checkpoint_root": "orbax", "training_log": "trainlog.txt", "tensorboard": "tensorboard", "config": "config.yaml", "results": "results.json"},
                  "metrics": metrics, "history": history}
        _write_text(config.artifact_path("results.json"), json.dumps(result, indent=2, allow_nan=True, default=str))
        logger.log({"event": "training_completed", "best_checkpoint_step": best_step,
                    "retained_checkpoint_steps": [item.step for item in retained], "results": config.artifact_path("results.json")})
        return result
    finally:
        manager.close()
        logger.close()


def _run_dry_run(config: CnnExperimentConfig) -> dict[str, Any]:
    runtime = validate_backend(config.runtime.accelerator)
    devices = list(jax.local_devices())
    if config.runtime.execution_mode == "single_device":
        devices = devices[:1]
    if not devices:
        raise RuntimeError("JAX reported no local devices.")
    print(f"dry_run_started backend={runtime.backend} devices={len(devices)}", flush=True)
    print("dry_run_loading_manifest", flush=True)
    records = build_cxr_rait_manifest(config.dataset_root)
    split = load_split_manifest(config.split_manifest)
    model = CxrSmallCNN(rngs=nnx.Rngs(config.seed), dropout_rate=config.dropout_rate)
    optimizer = create_cnn_optimizer(model, config.learning_rate, config.weight_decay)
    print("dry_run_loading_first_batch", flush=True)
    batches = _batches(records, split["splits"]["train"], config, shuffle=False, seed=config.seed)

    def dry_run_progress(batch_index: int, batch_loss: float, batch_seconds: float) -> None:
        iter_per_sec = 1.0 / max(batch_seconds, 1e-12)
        sample_per_sec = config.batch_size * iter_per_sec
        print(
            "dry_run_train_batch "
            f"epoch=1 total_epochs=1 batch={batch_index} total_batches=1 "
            f"samples_per_epoch={config.batch_size} samples_processed={config.batch_size} "
            f"batch_loss={batch_loss:.6g} batch_seconds={batch_seconds:.3f} "
            f"iter_per_sec={iter_per_sec:.3f} sample_per_sec={sample_per_sec:.3f} "
            f"epoch_samples_per_sec={sample_per_sec:.3f} "
            "eta_seconds=0",
            flush=True,
        )

    train_cnn(
        model,
        optimizer,
        iter([next(batches)]),
        devices=devices,
        progress_callback=dry_run_progress,
    )
    print("dry_run_completed", flush=True)
    return {"dry_run": True, "runtime": runtime.__dict__, "records": len(records),
            "train_records": len(split["splits"]["train"]), "run_directory": config.run_directory}
