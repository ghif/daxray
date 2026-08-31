"""Typed configuration and artifact paths for DAXRay experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class DatasetCacheConfig:
    mode: str = "memory"
    max_bytes: int = 1_073_741_824


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "cxr_rait"
    root: str = "gs://cxr-rait/cxr-demography-data"
    split_manifest: str = "artifacts/cxr_rait/split_manifest_seed7.json"
    input_res: int = 128
    resize_mode: str = "pad"
    cache: DatasetCacheConfig = DatasetCacheConfig()


@dataclass(frozen=True)
class ModelConfig:
    name: str = "cxr_small_cnn"
    dropout_rate: float = 0.2


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16


@dataclass(frozen=True)
class RuntimeConfig:
    accelerator: str = "cpu"
    precision: str = "fp32"
    execution_mode: str = "auto"


@dataclass(frozen=True)
class ArtifactsConfig:
    root: str = "checkpoints"
    run_name: str = "cnn_baseline_cpu_31-08-2026"
    remote_root: str = "gs://cxr-rait/checkpoints/classifier"
    checkpoint_keep_top_k: int = 3
    checkpoint_metric: str = "validation.auroc"


@dataclass(frozen=True)
class WorkflowConfig:
    type: str = "train-classifier"
    predictor_model: str = "cxr_rait_image_classifier"
    epochs: int = 30
    checkpoint_freq: int = 1
    speed_log_freq: int = 10


@dataclass(frozen=True)
class CnnExperimentConfig:
    version: str = "1"
    seed: int = 7
    dataset: DatasetConfig = DatasetConfig(cache=DatasetCacheConfig())
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()
    workflow: WorkflowConfig = WorkflowConfig()

    def __post_init__(self) -> None:
        if self.dataset.name != "cxr_rait":
            raise ValueError("dataset.name must be 'cxr_rait'.")
        if self.dataset.input_res <= 0 or self.optimizer.batch_size <= 0:
            raise ValueError("dataset.input_res and optimizer.batch_size must be positive.")
        if self.workflow.epochs <= 0 or self.workflow.checkpoint_freq <= 0 or self.workflow.speed_log_freq <= 0:
            raise ValueError("workflow.epochs, checkpoint_freq, and speed_log_freq must be positive.")
        if self.optimizer.lr <= 0 or self.optimizer.weight_decay < 0:
            raise ValueError("optimizer.lr must be positive and weight_decay non-negative.")
        if not 0 <= self.model.dropout_rate < 1:
            raise ValueError("model.dropout_rate must be in [0, 1).")
        if self.dataset.resize_mode not in {"pad", "stretch"}:
            raise ValueError("dataset.resize_mode must be 'pad' or 'stretch'.")
        if self.dataset.cache.mode not in {"none", "memory", "ephemeral"}:
            raise ValueError("dataset.cache.mode must be none, memory, or ephemeral.")
        if self.dataset.cache.max_bytes <= 0:
            raise ValueError("dataset.cache.max_bytes must be positive.")
        if self.runtime.accelerator not in {"cpu", "gpu", "cuda", "tpu"}:
            raise ValueError("runtime.accelerator must be cpu, gpu, or tpu.")
        if self.runtime.precision not in {"fp32", "bf16"}:
            raise ValueError("runtime.precision must be fp32 or bf16.")
        if self.runtime.execution_mode not in {"auto", "single_device", "multi_device"}:
            raise ValueError("runtime.execution_mode must be auto, single_device, or multi_device.")
        if self.artifacts.checkpoint_keep_top_k <= 0:
            raise ValueError("artifacts.checkpoint_keep_top_k must be positive.")
        if self.artifacts.checkpoint_metric != "validation.auroc":
            raise ValueError("artifacts.checkpoint_metric must be 'validation.auroc'.")
        name = self.artifacts.run_name
        if not name or name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("artifacts.run_name must be a single non-empty path component.")

    @property
    def dataset_root(self) -> str:
        return self.dataset.root

    @property
    def split_manifest(self) -> str:
        return self.dataset.split_manifest

    @property
    def image_size(self) -> int:
        return self.dataset.input_res

    @property
    def resize_mode(self) -> str:
        return self.dataset.resize_mode

    @property
    def batch_size(self) -> int:
        return self.optimizer.batch_size

    @property
    def learning_rate(self) -> float:
        return self.optimizer.lr

    @property
    def weight_decay(self) -> float:
        return self.optimizer.weight_decay

    @property
    def dropout_rate(self) -> float:
        return self.model.dropout_rate

    @property
    def epochs(self) -> int:
        return self.workflow.epochs

    @property
    def checkpoint_name(self) -> str:
        return self.artifacts.run_name

    @property
    def checkpoint_root(self) -> str:
        return self.artifacts.remote_root

    @property
    def checkpoint_keep_top_k(self) -> int:
        return self.artifacts.checkpoint_keep_top_k

    @property
    def checkpoint_metric(self) -> str:
        return self.artifacts.checkpoint_metric

    @property
    def run_directory(self) -> str:
        return f"{self.checkpoint_root.rstrip('/')}/{self.checkpoint_name}"

    def artifact_path(self, name: str) -> str:
        return f"{self.run_directory.rstrip('/')}/{name.lstrip('/')}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_cnn_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> CnnExperimentConfig:
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError("Configuration YAML must contain a mapping.")
    if overrides and overrides.get("checkpoint_name") is not None:
        artifacts = dict(values.get("artifacts") or {})
        artifacts["run_name"] = overrides["checkpoint_name"]
        values["artifacts"] = artifacts
    sections = {name: dict(values.get(name) or {}) for name in ("dataset", "model", "optimizer", "runtime", "artifacts", "workflow")}
    dataset_cache = DatasetCacheConfig(**dict(sections["dataset"].pop("cache", {}) or {}))
    return CnnExperimentConfig(
        version=str(values.get("version", "1")),
        seed=int(values.get("seed", 7)),
        dataset=DatasetConfig(cache=dataset_cache, **sections["dataset"]),
        model=ModelConfig(**sections["model"]),
        optimizer=OptimizerConfig(**sections["optimizer"]),
        runtime=RuntimeConfig(**sections["runtime"]),
        artifacts=ArtifactsConfig(**sections["artifacts"]),
        workflow=WorkflowConfig(**sections["workflow"]),
    )
