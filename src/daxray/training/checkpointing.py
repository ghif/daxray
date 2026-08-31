"""Orbax checkpoint management for resumable DAXRay training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import fsspec
import orbax.checkpoint as ocp


@dataclass(frozen=True)
class RetainedCheckpoint:
    step: int
    auroc: float


class TopKCheckpointManager:
    """Save full training states and retain the top K validation scores."""

    def __init__(self, directory: str, *, keep_top_k: int = 3, resume: bool = False):
        self.directory = directory
        orbax_directory = f"{directory.rstrip('/')}/orbax"
        fs, orbax_name = fsspec.core.url_to_fs(orbax_directory)
        fs.makedirs(orbax_name, exist_ok=True)
        fs.makedirs(f"{orbax_name.rstrip('/')}/metadata", exist_ok=True)
        self._manager = ocp.CheckpointManager(
            orbax_directory,
            handler_registry=ocp.DefaultCheckpointHandlerRegistry(),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=keep_top_k,
                best_fn=lambda metrics: float(metrics["selection_score"]),
                best_mode="max",
                keep_checkpoints_without_metrics=False,
                enable_async_checkpointing=False,
            ),
        )
        self.resume = resume

    @property
    def latest_step(self) -> int | None:
        return self._manager.latest_step()

    @property
    def best_step(self) -> int | None:
        return self._manager.best_step()

    def save(self, step: int, state: Mapping[str, Any], auroc: float) -> bool:
        return bool(self._manager.save(
            step,
            args=ocp.args.StandardSave(dict(state)),
            metrics={"validation_auroc": float(auroc), "selection_score": float(auroc) + step * 1e-12},
        ))

    def restore(self, step: int | None = None, *, target: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        selected = self.latest_step if step is None else step
        if selected is None:
            return None
        restore_args = ocp.args.StandardRestore(target) if target is not None else ocp.args.StandardRestore()
        return dict(self._manager.restore(selected, args=restore_args))

    def retained(self) -> list[RetainedCheckpoint]:
        result = []
        for step in self._manager.all_steps():
            metadata = self._manager.metadata(step)
            metrics = getattr(metadata, "metrics", {}) or {}
            result.append(RetainedCheckpoint(step=int(step), auroc=float(metrics.get("validation_auroc", float("nan")))))
        return sorted(result, key=lambda item: item.step)

    def close(self) -> None:
        self._manager.close()
