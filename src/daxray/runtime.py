"""JAX backend selection and topology validation."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeSummary:
    requested_accelerator: str
    backend: str
    local_device_count: int
    global_device_count: int
    process_count: int
    process_index: int
    device_kind: str
    jax_version: str


def configure_backend(accelerator: str) -> str:
    requested = accelerator.lower()
    if requested in {"gpu", "cuda"}:
        os.environ["JAX_PLATFORMS"] = "cuda"
    elif requested in {"cpu", "tpu"}:
        os.environ["JAX_PLATFORMS"] = requested
    else:
        raise ValueError(f"Unsupported accelerator {accelerator!r}.")
    os.environ.pop("JAX_PLATFORM_NAME", None)
    return requested


def validate_backend(accelerator: str) -> RuntimeSummary:
    import jax

    requested = accelerator.lower()
    expected = "gpu" if requested in {"gpu", "cuda"} else requested
    actual = jax.default_backend()
    if actual != expected:
        hint = "Install the matching JAX accelerator package."
        if requested == "tpu":
            hint = "Install jax[tpu] and libtpu on the TPU VM."
        raise RuntimeError(f"Requested accelerator={requested!r}, but JAX selected backend={actual!r}. {hint}")
    devices = jax.local_devices()
    return RuntimeSummary(
        requested_accelerator=requested,
        backend=actual,
        local_device_count=jax.local_device_count(),
        global_device_count=jax.device_count(),
        process_count=jax.process_count(),
        process_index=jax.process_index(),
        device_kind=str(getattr(devices[0], "device_kind", "unknown")) if devices else "none",
        jax_version=jax.__version__,
    )
