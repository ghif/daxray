"""Orbax checkpoint management for resumable DAXRay training."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
        self._remote = directory.startswith("gs://")
        self._temporary = tempfile.TemporaryDirectory(prefix="daxray-checkpoint-") if self._remote else None
        if self._remote:
            local_root = Path(self._temporary.name)
            orbax_directory = str(local_root / "orbax")
            self._remote_orbax_directory = f"{directory.rstrip('/')}/orbax"
            self._download_remote() if resume else None
            self._uploader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="daxray-checkpoint-upload")
            self._snapshot_root = local_root / "snapshots"
            self._snapshot_root.mkdir()
        else:
            orbax_directory = f"{directory.rstrip('/')}/orbax"
            self._remote_orbax_directory = None
            self._uploader = None
            self._snapshot_root = None
        self._local_orbax_directory = Path(orbax_directory)
        Path(orbax_directory).mkdir(parents=True, exist_ok=True)
        Path(orbax_directory, "metadata").mkdir(parents=True, exist_ok=True)
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
        self._upload_futures = []
        self.resume = resume

    @property
    def latest_step(self) -> int | None:
        return self._manager.latest_step()

    @property
    def best_step(self) -> int | None:
        return self._manager.best_step()

    def save(self, step: int, state: Mapping[str, Any], auroc: float) -> bool:
        saved = bool(self._manager.save(
            step,
            args=ocp.args.StandardSave(dict(state)),
            metrics={"validation_auroc": float(auroc), "selection_score": float(auroc) + step * 1e-12},
        ))
        if saved and self._remote:
            # Orbax may finalize even synchronous local saves on a worker
            # thread. Snapshot only after the local directory is complete.
            self._manager.wait_until_finished()
            retained_steps = [int(item) for item in self._manager.all_steps()]
            snapshot = self._snapshot_root / str(step)
            shutil.copytree(self._local_orbax_directory / str(step), snapshot)
            self._upload_futures.append(self._uploader.submit(self._upload_snapshot, step, snapshot, retained_steps))
        return saved

    def restore(self, step: int | None = None, *, target: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        selected = self.latest_step if step is None else step
        if selected is None:
            return None
        restore_args = ocp.args.StandardRestore(target) if target is not None else ocp.args.StandardRestore()
        return dict(self._manager.restore(selected, args=restore_args))

    def retained(self) -> list[RetainedCheckpoint]:
        self.wait_until_finished()
        result = []
        for step in self._manager.all_steps():
            metadata = self._manager.metadata(step)
            metrics = getattr(metadata, "metrics", {}) or {}
            result.append(RetainedCheckpoint(step=int(step), auroc=float(metrics.get("validation_auroc", float("nan")))))
        return sorted(result, key=lambda item: item.step)

    def close(self) -> None:
        self._manager.close()
        if self._uploader is not None:
            self._uploader.shutdown(wait=True)
            for future in self._upload_futures:
                future.result()
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def wait_until_finished(self) -> None:
        """Wait for pending asynchronous checkpoint uploads."""
        self._manager.wait_until_finished()

    def _download_remote(self) -> None:
        fs, remote_root = fsspec.core.url_to_fs(self._remote_orbax_directory)
        if not fs.exists(remote_root):
            return
        local_root = Path(self._temporary.name) / "orbax"
        for remote_path in fs.find(remote_root):
            relative = remote_path[len(remote_root):].lstrip("/")
            local_path = local_root / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with fs.open(remote_path, "rb") as source, local_path.open("wb") as target:
                shutil.copyfileobj(source, target)

    def _upload_snapshot(self, step: int, snapshot: Path, retained_steps: list[int]) -> None:
        fs, remote_root = fsspec.core.url_to_fs(self._remote_orbax_directory)
        fs.makedirs(remote_root, exist_ok=True)
        remote_step = f"{remote_root.rstrip('/')}/{step}"
        for local_path in snapshot.rglob("*"):
            if local_path.is_file():
                relative = local_path.relative_to(snapshot).as_posix()
                remote_path = f"{remote_step}/{relative}"
                parent = remote_path.rsplit("/", 1)[0]
                fs.makedirs(parent, exist_ok=True)
                with local_path.open("rb") as source, fs.open(remote_path, "wb") as target:
                    shutil.copyfileobj(source, target)
        for remote_step_path in fs.glob(f"{remote_root.rstrip('/')}/*"):
            name = remote_step_path.rstrip("/").rsplit("/", 1)[-1]
            if name.isdigit() and int(name) not in retained_steps:
                fs.rm(remote_step_path, recursive=True)
        shutil.rmtree(snapshot, ignore_errors=True)
