from __future__ import annotations

import copy

import pytest

from src.config import ConfigError, load_config, validate_config


def test_example_config_loads_and_validates() -> None:
    config = load_config("config.example.yaml")

    assert config["dataset"]["name"] == "mipnerf360"
    assert config["dataset"]["scene"] == "bonsai"
    assert config["reproducibility"]["random_seed"] == 42


def test_validation_reports_missing_required_key() -> None:
    config = load_config("config.example.yaml")
    invalid = copy.deepcopy(config)
    del invalid["paths"]["data_root"]

    with pytest.raises(ConfigError, match="paths.data_root"):
        validate_config(invalid)


def test_validation_rejects_non_positive_frame_interval() -> None:
    config = load_config("config.example.yaml")
    invalid = copy.deepcopy(config)
    invalid["video"]["frame_interval"] = 0

    with pytest.raises(ConfigError, match="frame_interval"):
        validate_config(invalid)
