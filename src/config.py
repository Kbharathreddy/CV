"""Configuration loading and validation."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from src.utils import PROJECT_ROOT

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - exercised only when PyYAML is missing
    yaml = None


class ConfigError(ValueError):
    """Raised when a configuration file is missing or invalid."""


REQUIRED_KEYS: dict[str, set[str]] = {
    "paths": {"data_root", "output_root", "colmap_executable", "gaussian_splatting_path"},
    "dataset": {"name", "scene"},
    "video": {"frame_interval", "target_frames"},
    "quality": {
        "blur_threshold",
        "min_brightness",
        "max_brightness",
        "dark_pixel_threshold",
        "bright_pixel_threshold",
    },
    "frame_selection": {"method", "redundancy_threshold", "minimum_matches"},
    "preprocessing": {"max_width", "max_height"},
    "colmap": {"matcher", "camera_model"},
    "gaussian": {"iterations", "lambda_ssim", "device", "checkpoint_interval"},
    "evaluation": {"psnr", "ssim", "lpips"},
    "reproducibility": {"random_seed"},
}


def _strip_inline_comment(value: str) -> str:
    if "#" not in value:
        return value
    if value.lstrip().startswith("#"):
        return ""
    return value.split(" #", 1)[0].rstrip()


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    if value in {"true", "True", "TRUE"}:
        return True
    if value in {"false", "False", "FALSE"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?", value):
        return float(value)
    return value


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Parse the small config subset used by config.example.yaml.

    PyYAML is the supported parser. This fallback keeps config validation and
    lightweight tests usable in minimal environments.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if "\t" in raw_line:
            raise ConfigError(f"Tabs are not supported in YAML fallback at line {line_number}.")
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = _strip_inline_comment(raw_line.strip())
        if not line:
            continue
        if ":" not in line:
            raise ConfigError(f"Invalid YAML line {line_number}: {raw_line!r}")

        key, raw_value = line.split(":", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"Empty YAML key at line {line_number}.")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ConfigError(f"Invalid indentation at line {line_number}.")
        parent = stack[-1][1]

        if raw_value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(raw_value)

    return root


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = _simple_yaml_load(text)
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration must be a mapping: {config_path}")
    return data


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate required sections, keys, and critical value ranges."""

    missing: list[str] = []
    for section, keys in REQUIRED_KEYS.items():
        value = config.get(section)
        if not isinstance(value, dict):
            missing.append(section)
            continue
        for key in sorted(keys):
            if key not in value:
                missing.append(f"{section}.{key}")

    if missing:
        joined = ", ".join(missing)
        raise ConfigError(f"Missing required configuration keys: {joined}")

    if int(config["video"]["frame_interval"]) <= 0:
        raise ConfigError("video.frame_interval must be positive.")
    if int(config["video"]["target_frames"]) <= 0:
        raise ConfigError("video.target_frames must be positive.")
    if int(config["preprocessing"]["max_width"]) <= 0:
        raise ConfigError("preprocessing.max_width must be positive.")
    if int(config["preprocessing"]["max_height"]) <= 0:
        raise ConfigError("preprocessing.max_height must be positive.")
    if int(config["gaussian"]["iterations"]) <= 0:
        raise ConfigError("gaussian.iterations must be positive.")
    if int(config["gaussian"]["checkpoint_interval"]) <= 0:
        raise ConfigError("gaussian.checkpoint_interval must be positive.")

    return config


def load_config(
    path: str | Path = "config.yaml",
    *,
    allow_example_fallback: bool = False,
) -> dict[str, Any]:
    """Load and validate project configuration.

    Args:
        path: Config path. Relative paths are resolved from the project root.
        allow_example_fallback: If true, use config.example.yaml when path is
            missing. Runtime entry points should generally require config.yaml.
    """

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists() and allow_example_fallback:
        config_path = PROJECT_ROOT / "config.example.yaml"

    return validate_config(load_yaml(config_path))


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two configuration mappings and return a new dictionary."""

    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
