"""Human-readable and TensorBoard logging for training runs."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import fsspec
from tensorboard.compat.proto import summary_pb2
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class RunLogger:
    def __init__(self, trainlog_path: str, tensorboard_path: str):
        self.trainlog_path = trainlog_path
        self.tensorboard_path = tensorboard_path.rstrip("/")
        self._remote = trainlog_path.startswith("gs://")
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        if self._remote:
            self._temporary = tempfile.TemporaryDirectory(prefix="daxray-tensorboard-")
            self._local_log = Path(self._temporary.name) / "trainlog.txt"
            self._local_tb = Path(self._temporary.name) / "tensorboard"
        else:
            self._local_log = Path(trainlog_path)
            self._local_tb = Path(tensorboard_path)
            self._local_log.parent.mkdir(parents=True, exist_ok=True)
        self._local_tb.mkdir(parents=True, exist_ok=True)
        self._handle = self._local_log.open("a", encoding="utf-8")
        self._writer = EventFileWriter(str(self._local_tb))

    def log(self, values: Mapping[str, Any], *, sync_remote: bool = True) -> None:
        event = dict(values)
        event.setdefault("timestamp_unix", time.time())
        event.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(event["timestamp_unix"])))
        message = self._format(event)
        print(message, flush=True)
        self._handle.write(message + "\n")
        self._handle.flush()
        step = int(event.get("epoch", 0))
        summary = summary_pb2.Summary()
        tensorboard_names = {
            "train_loss": "train/loss",
            "batch_loss": "train/batch_loss",
            "validation_accuracy": "validation/accuracy",
            "validation_auroc": "validation/auroc",
            "validation_auprc": "validation/auprc",
            "validation_sensitivity": "validation/sensitivity",
            "validation_specificity": "validation/specificity",
            "learning_rate": "learning_rate",
            "epoch_seconds": "epoch_duration_seconds",
            "epoch_samples_per_sec": "epoch_samples_per_sec",
            "retained_checkpoint_count": "retained_checkpoint_count",
            "best_validation_auroc": "best_validation_auroc",
        }
        for key, value in event.items():
            if isinstance(value, (int, float)) and value == value:
                summary.value.add(tag=tensorboard_names.get(str(key), str(key)), simple_value=float(value))
        self._writer.add_event(event_pb2.Event(wall_time=float(event["timestamp_unix"]), step=step, summary=summary))
        self._writer.flush()
        if self._remote and sync_remote:
            self._sync()

    @staticmethod
    def _format(values: Mapping[str, Any]) -> str:
        timestamp = values.get("timestamp", "")
        fields = []
        for key, value in values.items():
            if key in {"timestamp", "timestamp_unix"}:
                continue
            if isinstance(value, float):
                fields.append(f"{key}={value:.6g}")
            else:
                fields.append(f"{key}={value}")
        return f"{timestamp} INFO " + " ".join(fields)

    def _sync(self) -> None:
        fs, root = fsspec.core.url_to_fs(self.trainlog_path)
        fs.makedirs(os.path.dirname(root), exist_ok=True)
        with self._local_log.open("rb") as source, fs.open(root, "wb") as target:
            target.write(source.read())
        tb_fs, tb_root = fsspec.core.url_to_fs(self.tensorboard_path)
        tb_fs.makedirs(tb_root, exist_ok=True)
        for path in self._local_tb.iterdir():
            with path.open("rb") as source, tb_fs.open(f"{tb_root.rstrip('/')}/{path.name}", "wb") as target:
                target.write(source.read())

    def close(self) -> None:
        try:
            self._writer.close()
            self._handle.close()
            if self._remote:
                self._sync()
        finally:
            if self._temporary is not None:
                self._temporary.cleanup()
