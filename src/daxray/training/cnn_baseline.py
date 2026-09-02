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
from daxray.data import PreprocessingCache, build_cxr_rait_manifest, iter_batches, load_split_manifest
from daxray.evaluation.metrics import classification_metrics
from daxray.models.cnn import (
    CxrSmallCNN,
    binary_cross_entropy_with_logits_per_example,
    compute_dtype_for_precision,
)
from daxray.runtime import validate_backend
from .checkpointing import TopKCheckpointManager
from .logging import RunLogger


@nnx.jit
def _single_device_train_step(model: CxrSmallCNN, optimizer: nnx.Optimizer,
                              images: jnp.ndarray, labels: jnp.ndarray,
                              mask: jnp.ndarray, positive_weight: float,
                              negative_weight: float) -> jnp.ndarray:
    """JIT-compiled fixed-shape training step for CPU, GPU, and one TPU."""
    def loss_fn(current_model: CxrSmallCNN) -> jnp.ndarray:
        logits = current_model(images, training=True)
        # Reduce only after masking padded/unlabeled examples.  Reducing first
        # would include sentinel labels (-1) and can produce negative losses.
        losses = binary_cross_entropy_with_logits_per_example(logits, labels)
        weights = jnp.where(labels > 0.5, positive_weight, negative_weight)
        return jnp.sum(losses * weights * mask) / jnp.maximum(jnp.sum(weights * mask), 1.0)

    loss, gradients = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, gradients)
    return loss


@nnx.jit
def _evaluation_step(model: CxrSmallCNN, images: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled fixed-shape inference step."""
    return model(images, training=False)


def _pad_batch(images: np.ndarray, labels: np.ndarray, mask: np.ndarray, batch_size: int):
    padding = batch_size - len(images)
    if padding <= 0:
        return images, labels, mask
    return (
        np.pad(images, ((0, padding), (0, 0), (0, 0), (0, 0))),
        np.pad(labels, (0, padding), constant_values=-1),
        np.pad(mask, (0, padding), constant_values=False),
    )


def train_cnn(
    model: CxrSmallCNN,
    optimizer: nnx.Optimizer,
    batches: Iterator[dict[str, Any]],
    *,
    devices: list[jax.Device] | None = None,
    batch_size: int | None = None,
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
    progress_callback: Callable[[int, float, float], None] | None = None,
) -> float:
    """Run one epoch over NHWC batches and return mean training loss."""
    if devices and len(devices) > 1:
        return _train_cnn_multi_device(model, optimizer, batches, devices, progress_callback,
                                       positive_weight, negative_weight)
    losses = []
    for batch_index, batch in enumerate(batches, start=1):
        batch_started = time.perf_counter()
        images_np = np.asarray(batch["image"], dtype=np.float32)
        labels_np = np.asarray(batch["label"], dtype=np.float32)
        mask_np = np.asarray(batch.get("label_mask", np.ones(len(labels_np), dtype=bool)), dtype=bool)
        if not np.any(mask_np):
            continue
        images_np, labels_np, mask_np = _pad_batch(images_np, labels_np, mask_np, batch_size or len(images_np))
        loss = _single_device_train_step(
            model, optimizer, jnp.asarray(images_np), jnp.asarray(labels_np),
            jnp.asarray(mask_np, dtype=jnp.float32), positive_weight, negative_weight,
        )
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
    positive_weight: float = 1.0,
    negative_weight: float = 1.0,
) -> float:
    """Run replicated data-parallel updates across local devices."""
    device_count = len(devices)
    state_axes = nnx.StateAxes({nnx.Param: None, ...: None})

    @nnx.split_rngs(splits=device_count)
    @nnx.pmap(axis_name="devices", in_axes=(state_axes, None, 0, 0, 0), out_axes=0, devices=devices)
    def step(current_model, current_optimizer, images, labels, mask):
        def loss_fn(candidate_model):
            logits = candidate_model(images, training=True)
            losses = binary_cross_entropy_with_logits_per_example(logits, labels)
            weights = jnp.where(labels > 0.5, positive_weight, negative_weight)
            return jnp.sum(losses * weights * mask) / jnp.maximum(jnp.sum(weights * mask), 1.0)

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


def create_cnn_optimizer(model: CxrSmallCNN, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                         *, schedule: str = "constant", min_lr: float = 0.0,
                         total_steps: int | None = None, gradient_clip_norm: float = 0.0) -> nnx.Optimizer:
    """Create the AdamW optimizer used by the baseline."""
    if learning_rate <= 0.0 or weight_decay < 0.0 or min_lr < 0.0 or min_lr > learning_rate:
        raise ValueError("learning_rate must be positive and weight_decay cannot be negative.")
    if schedule == "cosine":
        if total_steps is None or total_steps <= 0:
            raise ValueError("total_steps must be positive for cosine scheduling.")
        learning_rate_fn = optax.cosine_decay_schedule(
            init_value=learning_rate, decay_steps=total_steps, alpha=min_lr / learning_rate
        )
    elif schedule == "constant":
        learning_rate_fn = learning_rate
    else:
        raise ValueError("schedule must be constant or cosine.")
    transforms = [optax.clip_by_global_norm(gradient_clip_norm)] if gradient_clip_norm > 0 else []
    transforms.append(optax.adamw(learning_rate_fn, weight_decay=weight_decay))
    return nnx.Optimizer(model, optax.chain(*transforms), wrt=nnx.Param)


def _batches(records, patient_ids, config: CnnExperimentConfig, *, shuffle: bool, seed: int,
             cache: PreprocessingCache | None = None):
    augmentation = config.dataset.augmentation
    return iter_batches(
        records, patient_ids, batch_size=config.batch_size, image_size=config.image_size,
        resize_mode=config.resize_mode, shuffle=shuffle, seed=seed, layout="NHWC", cache=cache,
        augment=shuffle and augmentation.enabled,
        horizontal_flip_prob=augmentation.horizontal_flip_prob,
        contrast_range=augmentation.contrast_range,
        brightness_range=augmentation.brightness_range,
    )


def _predict(model, records, patient_ids, config: CnnExperimentConfig,
             cache: PreprocessingCache | None = None) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels = [], []
    for batch in _batches(records, patient_ids, config, shuffle=False, seed=config.seed, cache=cache):
        mask = np.asarray(batch["label_mask"], dtype=bool)
        if not np.any(mask):
            continue
        batch_size = len(batch["image"])
        batch_labels = np.asarray(batch["label"])
        images = np.asarray(batch["image"], dtype=np.float32)
        images, _, _ = _pad_batch(images, batch_labels, mask, config.batch_size)
        logits = _evaluation_step(model, jnp.asarray(images))
        probabilities.extend(np.asarray(jax.nn.sigmoid(logits))[:batch_size][mask])
        labels.extend(batch_labels[mask])
    return np.asarray(labels), np.asarray(probabilities)


def _evaluate(model, records, patient_ids, config: CnnExperimentConfig,
              cache: PreprocessingCache | None = None, *, threshold: float = 0.5) -> dict[str, float | int]:
    labels, probabilities = _predict(model, records, patient_ids, config, cache)
    return classification_metrics(labels, probabilities, threshold=threshold)


def _select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Select a validation threshold maximizing balanced accuracy."""
    candidates = np.unique(np.concatenate((np.asarray([0.5]), probabilities)))
    scored = [(classification_metrics(labels, probabilities, float(threshold))["balanced_accuracy"],
               -abs(float(threshold) - 0.5), float(threshold)) for threshold in candidates]
    return max(scored)[2]


def _class_weights(records, patient_ids, config: CnnExperimentConfig) -> tuple[float, float]:
    if config.optimizer.class_weighting == "none":
        return 1.0, 1.0
    if config.optimizer.class_weighting == "custom":
        return config.optimizer.positive_class_weight, config.optimizer.negative_class_weight
    by_id = {record["patient_id"]: record for record in records}
    labels = np.asarray([by_id[patient_id]["label"] for patient_id in patient_ids], dtype=np.int32)
    positive_count = max(int(np.sum(labels == 1)), 1)
    negative_count = max(int(np.sum(labels == 0)), 1)
    total = positive_count + negative_count
    return total / (2.0 * positive_count), total / (2.0 * negative_count)


def _write_text(path: str, content: str) -> None:
    fs, name = fsspec.core.url_to_fs(path)
    parent = name.rsplit("/", 1)[0] if "/" in name else ""
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(name, "w", encoding="utf-8") as handle:
            handle.write(content)


def _checkpoint_target(model: CxrSmallCNN, optimizer: nnx.Optimizer, seed: int) -> dict[str, Any]:
    """Build the NNX-shaped target needed to restore Orbax checkpoint state."""
    return {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "epoch": 0,
        "rng": {"seed": seed},
    }


def run_cnn_baseline(config: CnnExperimentConfig, *, resume: bool = False, dry_run: bool = False) -> dict[str, Any]:
    if dry_run:
        return _run_dry_run(config)
    fs, run_name = fsspec.core.url_to_fs(config.run_directory)
    if fs.exists(run_name) and not resume:
        raise FileExistsError(f"Checkpoint directory already exists: {config.run_directory}")
    fs.makedirs(run_name, exist_ok=True)
    _write_text(config.artifact_path("config.yaml"), yaml.safe_dump(config.to_dict(), sort_keys=False))
    logger = RunLogger(config.artifact_path("trainlog.txt"), config.artifact_path("tensorboard"),
                       remote_sync_interval_epochs=config.workflow.remote_sync_interval_epochs)
    manager = TopKCheckpointManager(config.run_directory, keep_top_k=config.checkpoint_keep_top_k)
    cache = PreprocessingCache(config.dataset.cache.mode, config.dataset.cache.max_bytes,
                               config.dataset.cache.read_workers)
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
        if config.workflow.prewarm:
            cache_paths = [record["image_path"] for record in records if record.get("image_path") is not None]
            logger.log({"event": "data_prewarm_started", "sample_count": len(cache_paths),
                        "read_workers": config.dataset.cache.read_workers})
            def prewarm_progress(completed: int, total: int) -> None:
                if completed == 1 or completed == total or completed % max(1, config.dataset.cache.read_workers) == 0:
                    logger.log({"event": "data_prewarm_progress", "completed": completed, "total": total})
            cache.prewarm(cache_paths, config.image_size, config.resize_mode,
                          read_workers=config.dataset.cache.read_workers, progress=prewarm_progress)
            logger.log({"event": "data_prewarm_completed", **{f"data/{key}": value
                        for key, value in cache.stats.as_dict().items()}})
        else:
            logger.log({"event": "data_prewarm_skipped", "reason": "workflow.prewarm=false"})
        model = CxrSmallCNN(
            rngs=nnx.Rngs(config.seed),
            dropout_rate=config.dropout_rate,
            base_channels=config.model.base_channels,
            dtype=compute_dtype_for_precision(config.runtime.precision),
        )
        total_train_batches = config.epochs * ((len(split["splits"]["train"]) + config.batch_size - 1) // config.batch_size)
        optimizer = create_cnn_optimizer(
            model, config.learning_rate, config.weight_decay,
            schedule=config.optimizer.schedule, min_lr=config.optimizer.min_lr,
            total_steps=total_train_batches, gradient_clip_norm=config.optimizer.gradient_clip_norm,
        )
        positive_weight, negative_weight = _class_weights(records, split["splits"]["train"], config)
        logger.log({"event": "class_weights", "positive_class_weight": positive_weight,
                    "negative_class_weight": negative_weight})
        start_epoch = 0
        checkpoint_scores: dict[int, float] = {}
        if resume:
            restored = manager.restore(target=_checkpoint_target(model, optimizer, config.seed))
            if restored is not None:
                nnx.update(model, restored["model"])
                nnx.update(optimizer, restored["optimizer"])
                start_epoch = int(restored["epoch"])
                checkpoint_scores = {item.step: item.auroc for item in manager.retained()}
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
                _batches(records, split["splits"]["train"], config, shuffle=True, seed=config.seed + epoch, cache=cache),
                devices=devices,
                batch_size=config.batch_size,
                positive_weight=positive_weight,
                negative_weight=negative_weight,
                progress_callback=batch_progress,
            )
            train_seconds = time.perf_counter() - epoch_t0
            evaluation_started = time.perf_counter()
            training = _evaluate(model, records, split["splits"]["train"], config, cache)
            training_evaluation_seconds = time.perf_counter() - evaluation_started
            validation_started = time.perf_counter()
            validation = _evaluate(model, records, split["splits"]["validation"], config, cache)
            validation_seconds = time.perf_counter() - validation_started
            checkpoint_started = time.perf_counter()
            state = {"model": nnx.state(model), "optimizer": nnx.state(optimizer), "epoch": epoch, "rng": {"seed": config.seed}}
            saved = manager.save(epoch, state, float(validation["auroc"]))
            checkpoint_stage_seconds = time.perf_counter() - checkpoint_started
            checkpoint_scores[epoch] = float(validation["auroc"])
            retained_steps = [step for step, _ in sorted(
                checkpoint_scores.items(), key=lambda item: (item[1], item[0]), reverse=True
            )[:config.checkpoint_keep_top_k]]
            retained = [checkpoint_scores[step] for step in sorted(retained_steps)]
            event = {"timestamp": started.isoformat(), "timestamp_unix": started.timestamp(), "epoch": epoch,
                     "train_loss": loss, "training_accuracy": training["accuracy"],
                     "validation_auroc": validation["auroc"], "validation_auprc": validation["auprc"],
                     "validation_accuracy": validation["accuracy"], "validation_f1": validation["f1"],
                     "validation_sensitivity": validation["sensitivity"], "validation_specificity": validation["specificity"],
                     "learning_rate": config.learning_rate, "checkpoint_saved": saved,
                     "train_seconds": train_seconds, "training_evaluation_seconds": training_evaluation_seconds,
                     "validation_seconds": validation_seconds, "checkpoint_stage_seconds": checkpoint_stage_seconds,
                     "checkpoint_upload_pending": config.run_directory.startswith("gs://"),
                     "samples_per_epoch": samples_per_epoch,
                     "retained_checkpoint_steps": sorted(retained_steps),
                     "retained_checkpoint_count": len(retained_steps),
                     "best_validation_auroc": max(retained, default=float("nan")),
                     "epoch_seconds": (datetime.now(timezone.utc) - started).total_seconds()}
            event["epoch_samples_per_sec"] = samples_per_epoch / max(event["epoch_seconds"], 1e-12)
            event.update({f"data/{key}": value for key, value in cache.stats.as_dict().items()})
            logger.log(event)
            history.append({"epoch": epoch, "train_loss": loss, "training": training, "validation": validation})
        manager.wait_until_finished()
        best_step = manager.best_step
        if best_step is not None:
            best_state = manager.restore(
                best_step, target=_checkpoint_target(model, optimizer, config.seed)
            )
            nnx.update(model, best_state["model"])
        validation_labels, validation_probabilities = _predict(
            model, records, split["splits"]["validation"], config, cache
        )
        selected_threshold = _select_threshold(validation_labels, validation_probabilities)
        metrics = {name: _evaluate(model, records, ids, config, cache, threshold=selected_threshold)
                   for name, ids in split["splits"].items()}
        retained = manager.retained()
        result = {"model": "cxr_small_cnn", "dataset": "cxr_rait", "checkpoint_name": config.checkpoint_name,
                  "run_directory": config.run_directory, "best_checkpoint_step": best_step,
                  "retained_checkpoints": [item.__dict__ for item in retained], "manifest": config.split_manifest,
                  "dataset_fingerprint": split.get("dataset_fingerprint"), "runtime": runtime.__dict__, "resolved_config": config.to_dict(),
                  "selected_threshold": selected_threshold,
                  "artifacts": {"checkpoint_root": "orbax", "training_log": "trainlog.txt", "tensorboard": "tensorboard", "config": "config.yaml", "results": "results.json"},
                  "metrics": metrics, "history": history, "cache": cache.stats.as_dict()}
        _write_text(config.artifact_path("results.json"), json.dumps(result, indent=2, allow_nan=True, default=str))
        logger.log({"event": "training_completed", "best_checkpoint_step": best_step,
                    "retained_checkpoint_steps": [item.step for item in retained],
                    "selected_threshold": selected_threshold,
                    "positive_class_weight": positive_weight,
                    "negative_class_weight": negative_weight,
                    "training_accuracy": metrics["train"]["accuracy"],
                    "validation_accuracy": metrics["validation"]["accuracy"],
                    "test_accuracy": metrics["test"]["accuracy"],
                    "results": config.artifact_path("results.json")})
        return result
    finally:
        manager.close()
        logger.close()
        cache.close()


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
    model = CxrSmallCNN(
        rngs=nnx.Rngs(config.seed),
        dropout_rate=config.dropout_rate,
        base_channels=config.model.base_channels,
        dtype=compute_dtype_for_precision(config.runtime.precision),
    )
    optimizer = create_cnn_optimizer(model, config.learning_rate, config.weight_decay)
    cache = PreprocessingCache(config.dataset.cache.mode, config.dataset.cache.max_bytes,
                               config.dataset.cache.read_workers)
    print("dry_run_loading_first_batch", flush=True)
    batches = _batches(records, split["splits"]["train"], config, shuffle=False, seed=config.seed, cache=cache)

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

    try:
        train_cnn(
            model,
            optimizer,
            iter([next(batches)]),
            devices=devices,
            progress_callback=dry_run_progress,
        )
        print("dry_run_cache " + " ".join(f"{key}={value}" for key, value in cache.stats.as_dict().items()), flush=True)
        print("dry_run_completed", flush=True)
        return {"dry_run": True, "runtime": runtime.__dict__, "records": len(records),
                "train_records": len(split["splits"]["train"]), "run_directory": config.run_directory,
                "cache": cache.stats.as_dict()}
    finally:
        cache.close()
