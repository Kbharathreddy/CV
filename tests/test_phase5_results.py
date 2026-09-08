from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.phase5_results import (
    Phase5ResultError,
    aggregate_results,
    result_from_experiment_manifest,
    validate_phase5_result,
    write_results_csv,
)


def _manifest(name: str) -> dict:
    return {
        "experiment_name": name,
        "selection_method": "fixed_uniform",
        "training_budget": 30,
        "train_image_count": 30,
        "test_image_count": 37,
        "configuration": {"iterations": 30000},
    }


def test_result_placeholder_contains_no_fake_metrics() -> None:
    result = result_from_experiment_manifest(_manifest("fixed_30")).to_dict()

    validate_phase5_result(result)
    assert result["status"] == "not_run"
    assert result["psnr"] is None
    assert result["ssim"] is None
    assert result["lpips"] is None


def test_result_validation_rejects_missing_metrics_fields() -> None:
    result = result_from_experiment_manifest(_manifest("fixed_30")).to_dict()
    del result["psnr"]

    with pytest.raises(Phase5ResultError, match="missing"):
        validate_phase5_result(result)


def test_aggregate_results_writes_comparison_csv(tmp_path: Path) -> None:
    first = result_from_experiment_manifest(_manifest("fixed_60")).to_dict()
    second = result_from_experiment_manifest(_manifest("fixed_30")).to_dict()
    first["status"] = "success"
    first["psnr"] = 20.0
    first["ssim"] = 0.8
    first["lpips"] = 0.2
    paths = []
    for index, result in enumerate([first, second]):
        path = tmp_path / f"result_{index}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        paths.append(path)

    rows = aggregate_results(paths)
    csv_path = tmp_path / "comparison.csv"
    write_results_csv(csv_path, rows)

    assert [row["experiment"] for row in rows] == ["fixed_30", "fixed_60"]
    with csv_path.open(newline="", encoding="utf-8") as file:
        csv_rows = list(csv.DictReader(file))
    assert csv_rows[1]["psnr"] == "20.0"
