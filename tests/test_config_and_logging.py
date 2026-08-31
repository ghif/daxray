import numpy as np
import jax.numpy as jnp
from flax import nnx

from daxray.config import ArtifactsConfig, CnnExperimentConfig, load_cnn_config
from daxray.training.checkpointing import TopKCheckpointManager
from daxray.training.logging import RunLogger
from daxray.training import cnn_baseline
from daxray.runtime import RuntimeSummary
from daxray.models import CxrSmallCNN


def test_config_loads_and_resolves_named_run(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("version: '1'\nartifacts:\n  run_name: from_file\nworkflow:\n  epochs: 4\n", encoding="utf-8")
    config = load_cnn_config(path, {"checkpoint_name": "from_override"})
    assert config.checkpoint_name == "from_override"
    assert config.run_directory.endswith("/from_override")
    assert config.artifact_path("results.json").endswith("/from_override/results.json")


def test_config_rejects_nested_checkpoint_name():
    try:
        CnnExperimentConfig(artifacts=ArtifactsConfig(run_name="bad/name"))
    except ValueError as exc:
        assert "single non-empty path component" in str(exc)
    else:
        raise AssertionError("Expected nested checkpoint name to fail.")


def test_checkpoint_manager_retains_top_three_and_restores_best(tmp_path):
    manager = TopKCheckpointManager(str(tmp_path), keep_top_k=3)
    for step, score in enumerate((0.1, 0.5, 0.2, 0.4), start=1):
        manager.save(step, {"epoch": step, "value": np.asarray([step])}, score)
    assert [item.step for item in manager.retained()] == [2, 3, 4]
    assert manager.best_step == 2
    assert manager.restore(manager.best_step)["epoch"] == 2
    manager.close()


def test_checkpoint_manager_restores_nnx_state_with_target(tmp_path):
    model = CxrSmallCNN(rngs=nnx.Rngs(0))
    manager = TopKCheckpointManager(str(tmp_path), keep_top_k=1)
    manager.save(1, {"model": nnx.state(model), "epoch": 1, "rng": {"seed": 0}}, 0.5)

    restored = manager.restore(target={"model": nnx.state(model), "epoch": 0, "rng": {"seed": 0}})
    restored_model = CxrSmallCNN(rngs=nnx.Rngs(1))
    nnx.update(restored_model, restored["model"])

    assert restored_model(jnp.ones((1, 32, 32, 1))).shape == (1,)
    manager.close()


def test_run_logger_prints_and_writes_text_and_tensorboard_event(tmp_path, capsys):
    logger = RunLogger(str(tmp_path / "trainlog.txt"), str(tmp_path / "tensorboard"))
    logger.log({"epoch": 1, "train_loss": 0.25, "timestamp_unix": 1.0})
    logger.close()
    assert "epoch=1" in capsys.readouterr().out
    line = (tmp_path / "trainlog.txt").read_text(encoding="utf-8").splitlines()[0]
    assert "epoch=1" in line
    assert "train_loss=0.25" in line
    assert list((tmp_path / "tensorboard").iterdir())


def test_dry_run_loads_one_batch_and_reports_completion(monkeypatch, capsys):
    records = [{"patient_id": "P1", "image_path": "P1.dcm", "label": 1}]
    monkeypatch.setattr(cnn_baseline, "build_cxr_rait_manifest", lambda _: records)
    monkeypatch.setattr(cnn_baseline, "load_split_manifest", lambda _: {"splits": {"train": ["P1"], "validation": ["P1"], "test": ["P1"]}})
    monkeypatch.setattr(cnn_baseline, "_batches", lambda *_args, **_kwargs: iter([{
        "image": np.ones((1, 128, 128, 1), dtype=np.float32),
        "label": np.asarray([1], dtype=np.int32),
        "label_mask": np.asarray([True]),
    }]))
    monkeypatch.setattr(cnn_baseline, "validate_backend", lambda _: RuntimeSummary("cpu", "cpu", 1, 1, 1, 0, "cpu", "test"))
    monkeypatch.setattr(cnn_baseline.jax, "local_devices", lambda: [cnn_baseline.jax.devices("cpu")[0]])
    config = CnnExperimentConfig()
    result = cnn_baseline.run_cnn_baseline(config, dry_run=True)
    output = capsys.readouterr().out
    assert result["dry_run"] is True
    assert "dry_run_loading_first_batch" in output
    assert "dry_run_train_batch" in output
    assert "batch_loss=" in output
    assert "iter_per_sec=" in output
    assert "samples_per_epoch=" in output
    assert "epoch_samples_per_sec=" in output
    assert "eta_seconds=0" in output
    assert "dry_run_completed" in output
