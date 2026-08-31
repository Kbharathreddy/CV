"""Project pipeline entry point.

Phase 2 supports smartphone video preprocessing through frame extraction,
quality filtering, redundancy removal, preprocessing, CSV export, and summary
generation. COLMAP and Gaussian Splatting stages are intentionally not invoked.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # Allows: python src/pipeline.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ConfigError, load_config
from src.extract_frames import FrameExtractionError, extract_frames, extraction_strategy_from_config
from src.frame_quality import (
    FrameQualityError,
    FrameSelectionConfig,
    QualityThresholds,
    analyze_and_select_frames,
    summarize_selection,
    write_frame_selection_csv,
)
from src.preprocess import PreprocessConfig, PreprocessError, preprocess_images
from src.utils import configure_logging, ensure_dir, get_runtime_info, set_random_seed, utc_timestamp, write_json


class PipelineError(RuntimeError):
    """Raised when a pipeline run cannot complete."""


def _build_experiment_root(output_root: Path, experiment_name: str, overwrite: bool) -> Path:
    experiment_root = output_root / experiment_name
    if experiment_root.exists() and any(experiment_root.iterdir()):
        if not overwrite:
            raise PipelineError(
                f"Experiment output already exists: {experiment_root}. "
                "Choose a new --experiment-name or pass --overwrite."
            )
        shutil.rmtree(experiment_root)
    ensure_dir(experiment_root)
    return experiment_root


def _resolve_path(path_value: str | Path, *, base: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _strategy_with_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, int | float | None]:
    strategy = extraction_strategy_from_config(config)
    explicit = {
        "frame_interval": args.frame_interval,
        "time_interval": args.time_interval,
        "target_frames": args.target_frames if args.target_frames is not None else args.max_frames,
    }
    provided = {key: value for key, value in explicit.items() if value is not None}
    if provided:
        strategy = {"frame_interval": None, "time_interval": None, "target_frames": None}
        strategy.update(provided)
    enabled = [value for value in strategy.values() if value is not None]
    if len(enabled) != 1:
        raise PipelineError(
            "Exactly one extraction strategy must be configured after CLI overrides."
        )
    return strategy


def run_video_preprocessing(args: argparse.Namespace) -> dict[str, Any]:
    """Run the Phase 2 smartphone-video preprocessing pipeline."""

    start = time.perf_counter()
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        raise PipelineError(str(exc)) from exc

    project_root = Path(__file__).resolve().parents[1]
    data_root = _resolve_path(config["paths"]["data_root"], base=project_root)
    output_root = _resolve_path(config["paths"]["output_root"], base=project_root)
    input_path = _resolve_path(args.input, base=project_root)
    experiment_name = args.experiment_name or input_path.stem
    experiment_root = _build_experiment_root(output_root, experiment_name, args.overwrite)
    logs_dir = experiment_root / "logs"
    log_level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logger = configure_logging(logs_dir, level=log_level)

    set_random_seed(int(config["reproducibility"]["random_seed"]))
    logger.info("Runtime info: %s", get_runtime_info())
    logger.info("Phase 2 pipeline started for experiment '%s'", experiment_name)
    logger.info("Data root: %s", data_root)
    logger.info("Input video: %s", input_path)

    strategy = _strategy_with_overrides(config, args)
    effective_config = copy.deepcopy(config)
    effective_config["video"].update(strategy)
    if args.selection_mode is not None:
        if args.selection_mode.lower() != "automatic":
            raise PipelineError("Phase 2 supports only --selection-mode automatic.")
        effective_config["frame_selection"]["method"] = args.selection_mode.lower()

    preprocessing_config = PreprocessConfig.from_config(config["preprocessing"])
    candidates_dir = experiment_root / "frames" / "candidates"
    selected_dir = experiment_root / "frames" / "selected"

    try:
        video_metadata, extracted = extract_frames(
            input_path,
            candidates_dir,
            frame_interval=strategy.get("frame_interval"),
            time_interval=strategy.get("time_interval"),
            target_frames=strategy.get("target_frames"),
            jpeg_quality=preprocessing_config.jpeg_quality,
            logger=logger,
        )

        thresholds = QualityThresholds.from_config(effective_config["quality"])
        selection_config = FrameSelectionConfig.from_config(effective_config["frame_selection"])
        analyzed, selected = analyze_and_select_frames(
            extracted,
            thresholds=thresholds,
            selection_config=selection_config,
            logger=logger,
        )
        results = [item.result for item in analyzed]
        write_frame_selection_csv(experiment_root / "frame_selection.csv", results)

        selected_paths = [item.path for item in selected]
        preprocess_results = preprocess_images(
            selected_paths,
            selected_dir,
            preprocessing_config,
            logger=logger,
        )
    except (FrameExtractionError, FrameQualityError, PreprocessError) as exc:
        logger.error("%s", exc)
        raise PipelineError(str(exc)) from exc

    selection_summary = summarize_selection(results)
    elapsed = time.perf_counter() - start
    summary: dict[str, Any] = {
        "experiment_name": experiment_name,
        "input_type": "video",
        "video_path": str(input_path),
        "video_duration_seconds": video_metadata.duration_seconds,
        "video_fps": video_metadata.fps,
        "video_total_frames": video_metadata.total_frames,
        "processing_time_seconds": elapsed,
        "configuration": effective_config,
        "timestamp": utc_timestamp(),
    }
    summary.update(selection_summary)
    summary["preprocessed_frames"] = len(preprocess_results)
    summary["output_directories"] = {
        "experiment_root": str(experiment_root),
        "candidate_frames": str(candidates_dir),
        "selected_frames": str(selected_dir),
        "logs": str(logs_dir),
    }

    write_json(experiment_root / "summary.json", summary)
    logger.info("Final retained frames: %d", selection_summary["retained_frames"])
    logger.info("Processing time: %.3fs", elapsed)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the smartphone 3DGS project pipeline.")
    parser.add_argument("--mode", choices=["video", "dataset"], required=True)
    parser.add_argument("--input", help="Input smartphone MP4/MOV path for video mode.")
    parser.add_argument("--dataset", help="Dataset name for future dataset mode.")
    parser.add_argument("--scene", help="Scene name for future dataset mode.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--max-frames", type=int, default=None, help="Alias for target candidate frames.")
    parser.add_argument("--selection-mode", default=None, help="Reserved for Phase 6 experiment modes.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-colmap", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--stop-after", choices=["preprocessing"], default="preprocessing")
    parser.add_argument("--frame-interval", type=int, default=None)
    parser.add_argument("--time-interval", type=float, default=None)
    parser.add_argument("--target-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing experiment directory.")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "dataset":
        print("Dataset pipeline stages after download are not implemented until later phases.", file=sys.stderr)
        return 2
    if not args.input:
        print("--input is required for --mode video", file=sys.stderr)
        return 2

    logging.getLogger().setLevel(getattr(logging, str(args.log_level).upper(), logging.INFO))
    try:
        summary = run_video_preprocessing(args)
    except PipelineError as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
