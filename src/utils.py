"""Shared utilities for paths, logging, runtime metadata, and reproducibility."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class ProjectError(RuntimeError):
    """Base exception for recoverable project-level failures."""


def resolve_project_path(path: str | os.PathLike[str], root: Path | None = None) -> Path:
    """Resolve a path relative to the project root unless it is already absolute."""

    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    return (root or PROJECT_ROOT).joinpath(value).resolve()


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create a directory if needed and return the resolved path."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp suitable for logs and summaries."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_logging(
    log_dir: str | os.PathLike[str] | None = None,
    *,
    level: int = logging.INFO,
    logger_name: str = "smartphone_3dgs",
) -> logging.Logger:
    """Configure console logging and, optionally, a timestamped file log."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        directory = ensure_dir(log_dir)
        log_path = directory / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info("Logging to %s", log_path)

    return logger


def discover_images(root: str | os.PathLike[str]) -> list[Path]:
    """Return image files under a directory, sorted for reproducibility."""

    directory = Path(root)
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_json(path: str | os.PathLike[str], data: dict[str, Any]) -> None:
    """Write pretty JSON with stable key ordering."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def get_runtime_info() -> dict[str, Any]:
    """Collect runtime metadata for reproducibility logs."""

    info: dict[str, Any] = {
        "timestamp_utc": utc_timestamp(),
        "platform": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
    }

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional install state
        info["torch_available"] = False
        info["torch_error"] = str(exc)
        return info

    info["torch_available"] = True
    info["torch_version"] = getattr(torch, "__version__", None)
    cuda_available = bool(torch.cuda.is_available())
    info["cuda_available"] = cuda_available
    if cuda_available:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return info


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when available."""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def format_bytes(size_bytes: int) -> str:
    """Format a byte count for command-line messages."""

    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size_bytes} B"
