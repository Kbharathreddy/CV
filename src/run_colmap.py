"""Run COLMAP Structure-from-Motion for prepared image collections."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # Allows: python src/run_colmap.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ConfigError, load_config
from src.convert_colmap import ColmapModelError, read_colmap_model, summarize_model
from src.utils import configure_logging, discover_images, ensure_dir, utc_timestamp, write_json


class ColmapRunError(RuntimeError):
    """Raised when COLMAP is unavailable or a reconstruction stage fails."""


SUPPORTED_BACKENDS = {"auto", "pycolmap", "cli"}


@dataclass(frozen=True)
class ColmapExecutable:
    """Resolved COLMAP executable metadata."""

    path: str
    version: str


@dataclass(frozen=True)
class ColmapStageResult:
    """Execution metadata for a COLMAP command stage."""

    stage: str
    command: list[str]
    returncode: int
    duration_seconds: float
    stdout_log: str
    stderr_log: str


def pycolmap_is_available() -> bool:
    """Return whether PyCOLMAP can be imported in the active Python environment."""

    return importlib.util.find_spec("pycolmap") is not None


def import_pycolmap_module(*, required: bool = True):
    """Import PyCOLMAP, optionally raising a project-level error if unavailable."""

    try:
        import pycolmap  # type: ignore
    except ImportError as exc:
        if required:
            raise ColmapRunError(
                "PyCOLMAP is not installed. Install it with `py -3 -m pip install pycolmap` "
                "or use `--backend cli` with a configured COLMAP executable."
            ) from exc
        return None
    return pycolmap


def pycolmap_has_cuda(pycolmap_module: Any) -> bool | None:
    """Return PyCOLMAP CUDA availability when exposed by the installed package."""

    has_cuda = getattr(pycolmap_module, "has_cuda", None)
    if has_cuda is None:
        return None
    try:
        return bool(has_cuda() if callable(has_cuda) else has_cuda)
    except Exception:
        return None


def select_colmap_backend(
    backend: str = "auto",
    *,
    pycolmap_importable: bool | None = None,
) -> str:
    """Resolve the requested reconstruction backend."""

    requested = backend.lower()
    if requested not in SUPPORTED_BACKENDS:
        raise ColmapRunError("colmap.backend must be one of: auto, pycolmap, cli.")
    available = pycolmap_is_available() if pycolmap_importable is None else pycolmap_importable
    if requested == "auto":
        return "pycolmap" if available else "cli"
    if requested == "pycolmap" and not available:
        raise ColmapRunError("PyCOLMAP backend was requested but PyCOLMAP is not importable.")
    return requested


def detect_colmap(executable: str | Path = "colmap") -> ColmapExecutable:
    """Resolve COLMAP and return its version string."""

    executable_text = str(executable)
    candidate = shutil.which(executable_text)
    if candidate is None:
        path = Path(executable_text).expanduser()
        if path.exists() and path.is_file():
            candidate = str(path)
    if candidate is None:
        raise ColmapRunError(
            "COLMAP executable was not found. Install COLMAP and make it available as "
            "'colmap', or set paths.colmap_executable in config.yaml to the full executable path."
        )

    command = [candidate, "--version"]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ColmapRunError(f"Failed to run COLMAP executable: {candidate}") from exc
    output = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0:
        raise ColmapRunError(
            f"COLMAP version check failed with code {completed.returncode}: {output}"
        )
    return ColmapExecutable(path=candidate, version=output)


def choose_matcher(
    *,
    matcher: str,
    input_type: str,
    dataset_matcher: str = "exhaustive",
    video_matcher: str = "sequential",
) -> str:
    """Choose a COLMAP matcher based on configured mode and input type."""

    requested = matcher.lower()
    if requested in {"exhaustive", "sequential"}:
        return requested
    if requested != "auto":
        raise ColmapRunError("Matcher must be 'auto', 'exhaustive', or 'sequential'.")
    if input_type == "dataset":
        return dataset_matcher.lower()
    if input_type == "video":
        return video_matcher.lower()
    raise ColmapRunError(f"Unsupported input_type for matcher selection: {input_type}")


def find_sparse_model(sparse_root: str | Path) -> Path:
    """Find the first non-empty sparse model directory produced by COLMAP mapper."""

    root = Path(sparse_root)
    if not root.exists():
        raise ColmapRunError(f"Missing sparse reconstruction directory: {root}")
    candidates = [root]
    candidates.extend(path for path in sorted(root.iterdir()) if path.is_dir())
    for candidate in candidates:
        has_binary = all((candidate / name).exists() for name in ("cameras.bin", "images.bin", "points3D.bin"))
        has_text = all((candidate / name).exists() for name in ("cameras.txt", "images.txt", "points3D.txt"))
        if has_binary or has_text:
            return candidate
    raise ColmapRunError(f"No valid COLMAP sparse model found under {root}")


def _write_stage_logs(log_dir: Path, stage: str, stdout: str, stderr: str) -> tuple[Path, Path]:
    stdout_path = log_dir / f"{stage}.stdout.log"
    stderr_path = log_dir / f"{stage}.stderr.log"
    stdout_path.write_text(stdout or "", encoding="utf-8")
    stderr_path.write_text(stderr or "", encoding="utf-8")
    return stdout_path, stderr_path


def run_colmap_stage(
    command: list[str],
    *,
    stage: str,
    log_dir: Path,
    logger: logging.Logger,
) -> ColmapStageResult:
    """Run one COLMAP subprocess, write logs, and fail loudly on errors."""

    ensure_dir(log_dir)
    logger.info("Running COLMAP %s: %s", stage, " ".join(command))
    start = time.perf_counter()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ColmapRunError(f"Failed to start COLMAP stage '{stage}': {exc}") from exc
    duration = time.perf_counter() - start
    stdout_path, stderr_path = _write_stage_logs(log_dir, stage, completed.stdout, completed.stderr)
    result = ColmapStageResult(
        stage=stage,
        command=command,
        returncode=completed.returncode,
        duration_seconds=duration,
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
    )
    if completed.returncode != 0:
        error_tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-20:]
        raise ColmapRunError(
            f"COLMAP stage '{stage}' failed with exit code {completed.returncode}.\n"
            f"Command: {' '.join(command)}\n"
            f"Recent output:\n" + "\n".join(error_tail)
        )
    logger.info("COLMAP %s completed in %.2fs", stage, duration)
    return result


def build_colmap_commands(
    *,
    colmap_path: str,
    image_dir: Path,
    database_path: Path,
    sparse_dir: Path,
    matcher: str,
    camera_model: str,
    single_camera: bool,
    sequential_overlap: int,
) -> dict[str, list[str]]:
    """Build the COLMAP feature extraction, matching, and mapper commands."""

    feature_command = [
        colmap_path,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--ImageReader.camera_model",
        camera_model,
        "--ImageReader.single_camera",
        "1" if single_camera else "0",
    ]

    if matcher == "exhaustive":
        matcher_command = [colmap_path, "exhaustive_matcher", "--database_path", str(database_path)]
    elif matcher == "sequential":
        matcher_command = [
            colmap_path,
            "sequential_matcher",
            "--database_path",
            str(database_path),
            "--SequentialMatching.overlap",
            str(sequential_overlap),
        ]
    else:
        raise ColmapRunError(f"Unsupported matcher: {matcher}")

    mapper_command = [
        colmap_path,
        "mapper",
        "--database_path",
        str(database_path),
        "--image_path",
        str(image_dir),
        "--output_path",
        str(sparse_dir),
    ]
    return {
        "feature_extractor": feature_command,
        "matcher": matcher_command,
        "mapper": mapper_command,
    }


def validate_image_directory(image_dir: str | Path) -> list[Path]:
    """Validate that an image directory exists and contains RGB-like images."""

    directory = Path(image_dir)
    if not directory.exists() or not directory.is_dir():
        raise ColmapRunError(f"Missing image directory: {directory}")
    image_paths = discover_images(directory)
    if not image_paths:
        raise ColmapRunError(f"No images found in directory: {directory}")
    return image_paths


def prepare_output_directory(output_dir: str | Path, *, overwrite: bool) -> tuple[Path, Path, Path, Path]:
    """Create the standard Phase 3 output layout."""

    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise ColmapRunError(f"Output directory already exists and is non-empty: {root}")
        shutil.rmtree(root)
    database_dir = ensure_dir(root / "database")
    sparse_dir = ensure_dir(root / "sparse")
    log_dir = ensure_dir(root / "logs")
    database_path = database_dir / "database.db"
    return root, database_path, sparse_dir, log_dir


def _first_camera(model_summary: dict[str, Any]) -> dict[str, Any] | None:
    cameras = model_summary.get("cameras") or []
    return cameras[0] if cameras else None


def _camera_model(model_summary: dict[str, Any]) -> str | None:
    camera_models = model_summary.get("camera_models") or []
    if not camera_models:
        return None
    return camera_models[0] if len(camera_models) == 1 else ",".join(camera_models)


def build_reconstruction_summary(
    *,
    root: Path,
    backend: str,
    image_dir: Path,
    total_images: int,
    model_path: Path,
    model_summary: dict[str, Any],
    stage_results: list[ColmapStageResult],
    feature_extraction_time_seconds: float | None,
    matching_time_seconds: float | None,
    mapping_time_seconds: float | None,
    total_reconstruction_time_seconds: float,
    database_path: Path,
    matcher: str,
    camera_model_requested: str,
    input_type: str,
    dataset: str | None,
    scene: str | None,
    pycolmap_version: str | None,
    pycolmap_cuda_available: bool | None,
    colmap_executable: str | None,
    colmap_version: str | None,
) -> dict[str, Any]:
    """Create a backend-neutral COLMAP reconstruction summary."""

    camera = _first_camera(model_summary)
    return {
        "timestamp": utc_timestamp(),
        "experiment_name": root.name,
        "backend": backend,
        "pycolmap_version": pycolmap_version,
        "pycolmap_cuda_available": pycolmap_cuda_available,
        "input_type": input_type,
        "dataset": dataset,
        "scene": scene,
        "image_directory": str(image_dir),
        "image_dir": str(image_dir),
        "output_dir": str(root.resolve()),
        "total_images": total_images,
        "total_input_images": total_images,
        "registered_images": model_summary["registered_images"],
        "registration_rate": model_summary["registration_rate"],
        "camera_count": model_summary["number_of_cameras"],
        "camera_model": _camera_model(model_summary),
        "camera_intrinsics": camera["params"] if camera else None,
        "sparse_points": model_summary["number_of_sparse_points"],
        "observations": model_summary["number_of_observations"],
        "feature_extraction_time_seconds": feature_extraction_time_seconds,
        "matching_time_seconds": matching_time_seconds,
        "mapping_time_seconds": mapping_time_seconds,
        "total_reconstruction_time_seconds": total_reconstruction_time_seconds,
        "colmap_processing_time_seconds": total_reconstruction_time_seconds,
        "colmap_executable": colmap_executable,
        "colmap_version": colmap_version,
        "matcher": matcher,
        "camera_model_requested": camera_model_requested,
        "database_path": str(database_path.resolve()),
        "output_model_path": str(model_path.resolve()),
        "sparse_model_path": str(model_path.resolve()),
        "stages": [asdict(result) for result in stage_results],
        "model": model_summary,
    }


def parse_sparse_model(model_path: Path, *, total_images: int) -> dict[str, Any]:
    """Read a sparse model and fail with COLMAP runner context on parse errors."""

    try:
        model = read_colmap_model(model_path)
    except ColmapModelError as exc:
        raise ColmapRunError(f"Could not parse COLMAP sparse model: {exc}") from exc
    model_summary = summarize_model(model, total_images=total_images)
    if model_summary["registered_images"] == 0:
        raise ColmapRunError("COLMAP completed but registered zero images.")
    return model_summary


def run_colmap_cli_pipeline(
    *,
    image_dir: str | Path,
    output_dir: str | Path,
    colmap_executable: str | Path = "colmap",
    matcher: str = "auto",
    input_type: str = "dataset",
    dataset_matcher: str = "exhaustive",
    video_matcher: str = "sequential",
    camera_model: str = "OPENCV",
    single_camera: bool = True,
    sequential_overlap: int = 10,
    overwrite: bool = False,
    dataset: str | None = None,
    scene: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run COLMAP CLI SfM and summarize the sparse reconstruction."""

    image_paths = validate_image_directory(image_dir)
    image_dir_path = Path(image_dir).resolve()
    colmap = detect_colmap(colmap_executable)
    selected_matcher = choose_matcher(
        matcher=matcher,
        input_type=input_type,
        dataset_matcher=dataset_matcher,
        video_matcher=video_matcher,
    )

    root, database_path, sparse_dir, log_dir = prepare_output_directory(output_dir, overwrite=overwrite)
    logger = logger or configure_logging(log_dir, logger_name="smartphone_3dgs.colmap")

    logger.info("COLMAP executable: %s", colmap.path)
    logger.info("COLMAP version: %s", colmap.version)
    logger.info("COLMAP input images: %d from %s", len(image_paths), image_dir_path)
    logger.info("COLMAP matcher selected: %s", selected_matcher)

    commands = build_colmap_commands(
        colmap_path=colmap.path,
        image_dir=image_dir_path,
        database_path=database_path,
        sparse_dir=sparse_dir,
        matcher=selected_matcher,
        camera_model=camera_model,
        single_camera=single_camera,
        sequential_overlap=sequential_overlap,
    )

    start = time.perf_counter()
    stage_results: list[ColmapStageResult] = []
    feature_stage = run_colmap_stage(
        commands["feature_extractor"],
        stage="feature_extractor",
        log_dir=log_dir,
        logger=logger,
    )
    stage_results.append(feature_stage)
    if not database_path.exists() or database_path.stat().st_size == 0:
        raise ColmapRunError(f"COLMAP feature extraction did not create a valid database: {database_path}")

    matching_stage = run_colmap_stage(
        commands["matcher"],
        stage=selected_matcher + "_matcher",
        log_dir=log_dir,
        logger=logger,
    )
    stage_results.append(matching_stage)
    mapping_stage = run_colmap_stage(commands["mapper"], stage="mapper", log_dir=log_dir, logger=logger)
    stage_results.append(mapping_stage)
    processing_time = time.perf_counter() - start

    model_path = find_sparse_model(sparse_dir)
    model_summary = parse_sparse_model(model_path, total_images=len(image_paths))

    summary = build_reconstruction_summary(
        root=root,
        backend="cli",
        image_dir=image_dir_path,
        total_images=len(image_paths),
        model_path=model_path,
        model_summary=model_summary,
        stage_results=stage_results,
        feature_extraction_time_seconds=feature_stage.duration_seconds,
        matching_time_seconds=matching_stage.duration_seconds,
        mapping_time_seconds=mapping_stage.duration_seconds,
        total_reconstruction_time_seconds=processing_time,
        database_path=database_path,
        matcher=selected_matcher,
        camera_model_requested=camera_model,
        input_type=input_type,
        dataset=dataset,
        scene=scene,
        pycolmap_version=None,
        pycolmap_cuda_available=None,
        colmap_executable=colmap.path,
        colmap_version=colmap.version,
    )
    write_json(root / "summary.json", summary)
    logger.info(
        "Sparse reconstruction verified: registered=%d/%d points=%d observations=%s",
        model_summary["registered_images"],
        len(image_paths),
        model_summary["number_of_sparse_points"],
        model_summary["number_of_observations"],
    )
    return summary


def run_pycolmap_stage(
    callback,
    *,
    stage: str,
    log_dir: Path,
    logger: logging.Logger,
) -> tuple[ColmapStageResult, Any]:
    """Run one PyCOLMAP stage and capture Python-level stdout/stderr logs."""

    ensure_dir(log_dir)
    logger.info("Running PyCOLMAP %s", stage)
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            value = callback()
    except Exception as exc:
        duration = time.perf_counter() - start
        stdout_path, stderr_path = _write_stage_logs(
            log_dir,
            stage,
            stdout_buffer.getvalue(),
            stderr_buffer.getvalue(),
        )
        raise ColmapRunError(
            f"PyCOLMAP stage '{stage}' failed after {duration:.2f}s: {exc}. "
            f"Logs: {stdout_path}, {stderr_path}"
        ) from exc
    duration = time.perf_counter() - start
    stdout_path, stderr_path = _write_stage_logs(
        log_dir,
        stage,
        stdout_buffer.getvalue(),
        stderr_buffer.getvalue(),
    )
    result = ColmapStageResult(
        stage=stage,
        command=["pycolmap", stage],
        returncode=0,
        duration_seconds=duration,
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
    )
    logger.info("PyCOLMAP %s completed in %.2fs", stage, duration)
    return result, value


def _write_best_pycolmap_model(reconstructions: dict[int, Any], sparse_dir: Path) -> Path:
    if not reconstructions:
        raise ColmapRunError("PyCOLMAP mapping produced no reconstruction models.")
    best_model_id, best_reconstruction = max(
        reconstructions.items(),
        key=lambda item: item[1].num_reg_images(),
    )
    if best_reconstruction.num_reg_images() == 0:
        raise ColmapRunError("PyCOLMAP completed but registered zero images.")

    try:
        return find_sparse_model(sparse_dir)
    except ColmapRunError:
        model_dir = ensure_dir(sparse_dir / str(best_model_id))
        best_reconstruction.write(model_dir)
        return find_sparse_model(sparse_dir)


def run_pycolmap_pipeline(
    *,
    image_dir: str | Path,
    output_dir: str | Path,
    matcher: str = "auto",
    input_type: str = "dataset",
    dataset_matcher: str = "exhaustive",
    video_matcher: str = "sequential",
    camera_model: str = "OPENCV",
    single_camera: bool = True,
    sequential_overlap: int = 10,
    overwrite: bool = False,
    dataset: str | None = None,
    scene: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run PyCOLMAP SfM and summarize the sparse reconstruction."""

    pycolmap = import_pycolmap_module(required=True)
    image_paths = validate_image_directory(image_dir)
    image_dir_path = Path(image_dir).resolve()
    selected_matcher = choose_matcher(
        matcher=matcher,
        input_type=input_type,
        dataset_matcher=dataset_matcher,
        video_matcher=video_matcher,
    )
    root, database_path, sparse_dir, log_dir = prepare_output_directory(output_dir, overwrite=overwrite)
    logger = logger or configure_logging(log_dir, logger_name="smartphone_3dgs.colmap")

    pycolmap_version = getattr(pycolmap, "__version__", None)
    cuda_available = pycolmap_has_cuda(pycolmap)
    logger.info("PyCOLMAP version: %s", pycolmap_version)
    logger.info("PyCOLMAP CUDA available: %s", cuda_available)
    logger.info("PyCOLMAP input images: %d from %s", len(image_paths), image_dir_path)
    logger.info("PyCOLMAP matcher selected: %s", selected_matcher)

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = camera_model
    camera_mode = pycolmap.CameraMode.SINGLE if single_camera else pycolmap.CameraMode.PER_IMAGE
    extraction_options = pycolmap.FeatureExtractionOptions()
    if cuda_available is not None:
        extraction_options.use_gpu = cuda_available
    device = pycolmap.Device.auto

    start = time.perf_counter()
    stage_results: list[ColmapStageResult] = []
    feature_stage, _ = run_pycolmap_stage(
        lambda: pycolmap.extract_features(
            database_path,
            image_dir_path,
            camera_mode=camera_mode,
            reader_options=reader_options,
            extraction_options=extraction_options,
            device=device,
        ),
        stage="feature_extractor",
        log_dir=log_dir,
        logger=logger,
    )
    stage_results.append(feature_stage)
    if not database_path.exists() or database_path.stat().st_size == 0:
        raise ColmapRunError(f"PyCOLMAP feature extraction did not create a valid database: {database_path}")

    if selected_matcher == "exhaustive":
        pairing_options = pycolmap.ExhaustivePairingOptions()
        matching_stage, _ = run_pycolmap_stage(
            lambda: pycolmap.match_exhaustive(
                database_path,
                pairing_options=pairing_options,
                device=device,
            ),
            stage="exhaustive_matcher",
            log_dir=log_dir,
            logger=logger,
        )
    elif selected_matcher == "sequential":
        pairing_options = pycolmap.SequentialPairingOptions()
        pairing_options.overlap = sequential_overlap
        matching_stage, _ = run_pycolmap_stage(
            lambda: pycolmap.match_sequential(
                database_path,
                pairing_options=pairing_options,
                device=device,
            ),
            stage="sequential_matcher",
            log_dir=log_dir,
            logger=logger,
        )
    else:
        raise ColmapRunError(f"Unsupported matcher: {selected_matcher}")
    stage_results.append(matching_stage)

    mapping_options = pycolmap.IncrementalPipelineOptions()
    mapping_stage, reconstructions = run_pycolmap_stage(
        lambda: pycolmap.incremental_mapping(
            database_path,
            image_dir_path,
            sparse_dir,
            options=mapping_options,
        ),
        stage="mapper",
        log_dir=log_dir,
        logger=logger,
    )
    stage_results.append(mapping_stage)
    processing_time = time.perf_counter() - start

    model_path = _write_best_pycolmap_model(reconstructions, sparse_dir)
    model_summary = parse_sparse_model(model_path, total_images=len(image_paths))
    summary = build_reconstruction_summary(
        root=root,
        backend="pycolmap",
        image_dir=image_dir_path,
        total_images=len(image_paths),
        model_path=model_path,
        model_summary=model_summary,
        stage_results=stage_results,
        feature_extraction_time_seconds=feature_stage.duration_seconds,
        matching_time_seconds=matching_stage.duration_seconds,
        mapping_time_seconds=mapping_stage.duration_seconds,
        total_reconstruction_time_seconds=processing_time,
        database_path=database_path,
        matcher=selected_matcher,
        camera_model_requested=camera_model,
        input_type=input_type,
        dataset=dataset,
        scene=scene,
        pycolmap_version=pycolmap_version,
        pycolmap_cuda_available=cuda_available,
        colmap_executable=None,
        colmap_version=None,
    )
    write_json(root / "summary.json", summary)
    logger.info(
        "Sparse reconstruction verified: registered=%d/%d points=%d observations=%s",
        model_summary["registered_images"],
        len(image_paths),
        model_summary["number_of_sparse_points"],
        model_summary["number_of_observations"],
    )
    return summary


def run_reconstruction(
    *,
    image_dir: str | Path,
    output_dir: str | Path,
    backend: str = "auto",
    colmap_executable: str | Path = "colmap",
    matcher: str = "auto",
    input_type: str = "dataset",
    dataset_matcher: str = "exhaustive",
    video_matcher: str = "sequential",
    camera_model: str = "OPENCV",
    single_camera: bool = True,
    sequential_overlap: int = 10,
    overwrite: bool = False,
    dataset: str | None = None,
    scene: str | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Run SfM through the selected backend and return common metadata."""

    selected_backend = select_colmap_backend(backend)
    if selected_backend == "pycolmap":
        return run_pycolmap_pipeline(
            image_dir=image_dir,
            output_dir=output_dir,
            matcher=matcher,
            input_type=input_type,
            dataset_matcher=dataset_matcher,
            video_matcher=video_matcher,
            camera_model=camera_model,
            single_camera=single_camera,
            sequential_overlap=sequential_overlap,
            overwrite=overwrite,
            dataset=dataset,
            scene=scene,
            logger=logger,
        )
    return run_colmap_cli_pipeline(
        image_dir=image_dir,
        output_dir=output_dir,
        colmap_executable=colmap_executable,
        matcher=matcher,
        input_type=input_type,
        dataset_matcher=dataset_matcher,
        video_matcher=video_matcher,
        camera_model=camera_model,
        single_camera=single_camera,
        sequential_overlap=sequential_overlap,
        overwrite=overwrite,
        dataset=dataset,
        scene=scene,
        logger=logger,
    )


def run_colmap_pipeline(**kwargs: Any) -> dict[str, Any]:
    """Backward-compatible wrapper for the high-level reconstruction interface."""

    return run_reconstruction(**kwargs)


def _resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[1] / path).resolve()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run COLMAP/PyCOLMAP SfM over an image directory.")
    parser.add_argument("--image-dir", required=True, help="Input image directory.")
    parser.add_argument("--output-dir", required=True, help="Output experiment directory.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--backend", choices=["auto", "pycolmap", "cli"], default=None)
    parser.add_argument("--colmap-executable", default=None, help="Override COLMAP executable path.")
    parser.add_argument("--matcher", choices=["auto", "exhaustive", "sequential"], default=None)
    parser.add_argument("--input-type", choices=["dataset", "video"], default="dataset")
    parser.add_argument("--camera-model", default=None)
    parser.add_argument("--single-camera", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sequential-overlap", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    colmap_config = config["colmap"]
    logger = logging.getLogger("smartphone_3dgs.colmap")
    logger.setLevel(getattr(logging, str(args.log_level).upper(), logging.INFO))
    try:
        summary = run_reconstruction(
            image_dir=_resolve_project_path(args.image_dir),
            output_dir=_resolve_project_path(args.output_dir),
            backend=args.backend or colmap_config["backend"],
            colmap_executable=args.colmap_executable or config["paths"]["colmap_executable"],
            matcher=args.matcher or colmap_config["matcher"],
            input_type=args.input_type,
            dataset_matcher=colmap_config["dataset_matcher"],
            video_matcher=colmap_config["video_matcher"],
            camera_model=args.camera_model or colmap_config["camera_model"],
            single_camera=colmap_config["single_camera"] if args.single_camera is None else args.single_camera,
            sequential_overlap=args.sequential_overlap or colmap_config["sequential_overlap"],
            overwrite=args.overwrite,
            dataset=config["dataset"]["name"],
            scene=config["dataset"]["scene"],
        )
    except ColmapRunError as exc:
        print(f"COLMAP pipeline failed: {exc}", file=sys.stderr)
        return 2

    print(f"COLMAP summary written to {summary['output_dir']}")
    print(f"Backend: {summary['backend']}")
    print(f"Registered images: {summary['registered_images']} / {summary['total_images']}")
    print(f"Sparse points: {summary['sparse_points']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
