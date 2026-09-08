"""Phase 5 experiment result schema and aggregation helpers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:  # Allows: python src/phase5_results.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir, write_json


class Phase5ResultError(RuntimeError):
    """Raised when result summaries are missing required fields."""


@dataclass(frozen=True)
class Phase5ExperimentResult:
    """Standard JSON schema for one completed Phase 5 experiment."""

    experiment: str
    selection_method: str
    training_budget: int | None
    train_image_count: int
    test_image_count: int
    iterations: int
    gpu: str | None
    training_runtime_seconds: float | None
    render_runtime_seconds: float | None
    initial_loss: float | None
    final_loss: float | None
    number_of_gaussians: int | None
    psnr: float | None
    ssim: float | None
    lpips: float | None
    checkpoint_path: str | None
    point_cloud_path: str | None
    render_path: str | None
    status: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RESULT_COLUMNS = [
    "experiment",
    "selection_method",
    "training_budget",
    "train_image_count",
    "test_image_count",
    "iterations",
    "gpu",
    "training_runtime_seconds",
    "render_runtime_seconds",
    "initial_loss",
    "final_loss",
    "number_of_gaussians",
    "psnr",
    "ssim",
    "lpips",
    "checkpoint_path",
    "point_cloud_path",
    "render_path",
    "status",
    "warnings",
]


def validate_phase5_result(data: dict[str, Any]) -> None:
    """Validate the standard Phase 5 result JSON fields."""

    missing = [key for key in RESULT_COLUMNS if key not in data]
    if missing:
        raise Phase5ResultError(f"Phase 5 result is missing keys: {missing}")
    if not data["experiment"]:
        raise Phase5ResultError("Phase 5 result must include an experiment name.")
    if int(data["train_image_count"]) < 0:
        raise Phase5ResultError("train_image_count must be non-negative.")
    if int(data["test_image_count"]) <= 0:
        raise Phase5ResultError("test_image_count must be positive.")
    if int(data["iterations"]) <= 0:
        raise Phase5ResultError("iterations must be positive.")
    if str(data["status"]).lower() not in {"not_run", "success", "failed"}:
        raise Phase5ResultError("status must be one of: not_run, success, failed.")


def result_from_experiment_manifest(manifest: dict[str, Any], *, status: str = "not_run") -> Phase5ExperimentResult:
    """Create a metric-empty result placeholder for an unrun experiment."""

    return Phase5ExperimentResult(
        experiment=str(manifest["experiment_name"]),
        selection_method=str(manifest["selection_method"]),
        training_budget=manifest.get("training_budget"),
        train_image_count=int(manifest["train_image_count"]),
        test_image_count=int(manifest["test_image_count"]),
        iterations=int(manifest.get("configuration", {}).get("iterations", 30000)),
        gpu=None,
        training_runtime_seconds=None,
        render_runtime_seconds=None,
        initial_loss=None,
        final_loss=None,
        number_of_gaussians=None,
        psnr=None,
        ssim=None,
        lpips=None,
        checkpoint_path=None,
        point_cloud_path=None,
        render_path=None,
        status=status,
        warnings=[],
    )


def load_phase5_result(path: str | Path) -> dict[str, Any]:
    result_path = Path(path)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    validate_phase5_result(data)
    return data


def aggregate_results(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Load completed experiment summaries sorted by experiment name."""

    rows = [load_phase5_result(path) for path in paths]
    return sorted(rows, key=lambda item: str(item["experiment"]))


def write_results_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["warnings"] = "; ".join(row.get("warnings") or [])
            writer.writerow(serialized)


def write_results_json(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    write_json(path, {"phase": 5, "experiments": list(rows)})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate completed Phase 5 experiment result JSON files.")
    parser.add_argument("results", nargs="+", help="Experiment summary JSON files.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = aggregate_results(args.results)
        write_results_csv(args.output_csv, rows)
        if args.output_json:
            write_results_json(args.output_json, rows)
    except (Phase5ResultError, OSError, json.JSONDecodeError) as exc:
        print(f"Phase 5 result comparison failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"experiments": [row["experiment"] for row in rows]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
