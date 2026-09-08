"""Phase 5 experiment manifest and scene-preparation utilities.

This module does not run COLMAP or 3DGS. It prepares small, explicit train/test
scene layouts from an existing COLMAP sparse reconstruction so Kaggle can later
train one experiment at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence

if __package__ in {None, ""}:  # Allows: python src/phase5_experiments.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.convert_colmap import CAMERA_MODEL_IDS, CAMERA_MODELS, detect_model_format
from src.frame_selection import (
    ExperimentManifest,
    ImageQualitySignals,
    Phase5SelectionError,
    SplitManifest,
    automatic_budget_selection,
    build_experiment_manifest,
    build_pose_candidates,
    create_heldout_split,
    fixed_budget_selection,
    ordered_image_names_from_model,
    validate_experiment_manifest,
    write_experiment_manifest,
    write_split_manifest,
)
from src.gaussian_splatting import (
    OFFICIAL_3DGS_COMMIT,
    OFFICIAL_3DGS_REPO_URL,
)
from src.utils import PROJECT_ROOT, discover_images, ensure_dir, resolve_project_path, utc_timestamp, write_json


class Phase5ExperimentError(RuntimeError):
    """Raised when Phase 5 local preparation cannot be completed."""


@dataclass(frozen=True)
class ColmapCameraRecord:
    """Full camera record needed to rewrite a COLMAP model."""

    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]


@dataclass(frozen=True)
class ColmapImageRecord:
    """Full registered-image record including 2D point associations."""

    image_id: int
    qvec: list[float]
    tvec: list[float]
    camera_id: int
    name: str
    points2d: list[tuple[float, float, int]]


@dataclass(frozen=True)
class ColmapPoint3DRecord:
    """Sparse point record including the COLMAP track."""

    point3d_id: int
    xyz: list[float]
    rgb: tuple[int, int, int]
    error: float
    track: list[tuple[int, int]]


@dataclass(frozen=True)
class ColmapRawModel:
    """Raw COLMAP sparse model representation that can be filtered and written."""

    cameras: dict[int, ColmapCameraRecord]
    images: dict[int, ColmapImageRecord]
    points3d: dict[int, ColmapPoint3DRecord]
    model_format: str


@dataclass(frozen=True)
class PreparedScene:
    """Output paths and counts for one prepared train/test scene pair."""

    experiment_name: str
    output_dir: str
    train_scene: str
    test_scene: str
    train_image_count: int
    test_image_count: int
    train_sparse_points: int
    test_sparse_points: int
    link_mode: str
    dry_run: bool


DEFAULT_EXPERIMENTS = {
    "fixed_30": ("fixed_uniform", 30),
    "fixed_60": ("fixed_uniform", 60),
    "fixed_100": ("fixed_uniform", 100),
    "automatic_60": ("automatic_quality_pose", 60),
}


def _configured_experiment_specs(config: dict[str, Any]) -> dict[str, tuple[str, int]]:
    phase5_config = config.get("phase5", {})
    experiments = phase5_config.get("experiments", {})
    fixed_budgets = [int(value) for value in experiments.get("fixed_budgets", [30, 60, 100])]
    automatic_budget = int(experiments.get("automatic_budget", 60))
    specs = {f"fixed_{budget}": ("fixed_uniform", budget) for budget in fixed_budgets}
    specs[f"automatic_{automatic_budget}"] = ("automatic_quality_pose", automatic_budget)
    return specs


def _display_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _read_exact(file: BinaryIO, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise Phase5ExperimentError("Unexpected end of COLMAP model file.")
    return data


def _unpack(file: BinaryIO, fmt: str) -> tuple[Any, ...]:
    return struct.unpack("<" + fmt, _read_exact(file, struct.calcsize("<" + fmt)))


def _read_c_string(file: BinaryIO) -> str:
    chars = bytearray()
    while True:
        char = file.read(1)
        if not char:
            raise Phase5ExperimentError("Unexpected end of file while reading image name.")
        if char == b"\x00":
            return chars.decode("utf-8")
        chars.extend(char)


def _is_valid_point3d_id(point3d_id: int) -> bool:
    return point3d_id not in {-1, 2**64 - 1}


def read_raw_colmap_binary(model_path: str | Path) -> ColmapRawModel:
    """Read a binary COLMAP sparse model with tracks intact."""

    root = Path(model_path)
    cameras: dict[int, ColmapCameraRecord] = {}
    images: dict[int, ColmapImageRecord] = {}
    points3d: dict[int, ColmapPoint3DRecord] = {}

    with (root / "cameras.bin").open("rb") as file:
        (num_cameras,) = _unpack(file, "Q")
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _unpack(file, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise Phase5ExperimentError(f"Unsupported COLMAP camera model id: {model_id}")
            model_name, num_params = CAMERA_MODELS[model_id]
            params = [float(value) for value in _unpack(file, "d" * num_params)]
            cameras[int(camera_id)] = ColmapCameraRecord(
                camera_id=int(camera_id),
                model=model_name,
                width=int(width),
                height=int(height),
                params=params,
            )

    with (root / "images.bin").open("rb") as file:
        (num_images,) = _unpack(file, "Q")
        for _ in range(num_images):
            values = _unpack(file, "idddddddi")
            image_id = int(values[0])
            qvec = [float(value) for value in values[1:5]]
            tvec = [float(value) for value in values[5:8]]
            camera_id = int(values[8])
            name = _read_c_string(file)
            (num_points2d,) = _unpack(file, "Q")
            points2d = []
            for _point_index in range(num_points2d):
                x, y, point3d_id = _unpack(file, "ddq")
                points2d.append((float(x), float(y), int(point3d_id)))
            images[image_id] = ColmapImageRecord(
                image_id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                points2d=points2d,
            )

    with (root / "points3D.bin").open("rb") as file:
        (num_points,) = _unpack(file, "Q")
        for _ in range(num_points):
            point_id = int(_unpack(file, "Q")[0])
            xyz = [float(value) for value in _unpack(file, "ddd")]
            rgb = tuple(int(value) for value in _unpack(file, "BBB"))
            error = float(_unpack(file, "d")[0])
            (track_length,) = _unpack(file, "Q")
            track = []
            for _track_index in range(track_length):
                image_id, point2d_index = _unpack(file, "ii")
                track.append((int(image_id), int(point2d_index)))
            points3d[point_id] = ColmapPoint3DRecord(
                point3d_id=point_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                track=track,
            )

    return ColmapRawModel(cameras=cameras, images=images, points3d=points3d, model_format="binary")


def _iter_data_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if line and not line.startswith("#"):
                yield line


def read_raw_colmap_text(model_path: str | Path) -> ColmapRawModel:
    """Read a text COLMAP sparse model with tracks intact."""

    root = Path(model_path)
    cameras: dict[int, ColmapCameraRecord] = {}
    for line in _iter_data_lines(root / "cameras.txt"):
        parts = line.split()
        if len(parts) < 5:
            raise Phase5ExperimentError(f"Malformed cameras.txt line: {line}")
        camera_id = int(parts[0])
        cameras[camera_id] = ColmapCameraRecord(
            camera_id=camera_id,
            model=parts[1],
            width=int(parts[2]),
            height=int(parts[3]),
            params=[float(value) for value in parts[4:]],
        )

    images: dict[int, ColmapImageRecord] = {}
    image_lines = list(_iter_data_lines(root / "images.txt"))
    if len(image_lines) % 2 != 0:
        raise Phase5ExperimentError("images.txt should contain two non-comment lines per image.")
    for metadata_line, points_line in zip(image_lines[0::2], image_lines[1::2]):
        parts = metadata_line.split()
        if len(parts) < 10:
            raise Phase5ExperimentError(f"Malformed images.txt metadata line: {metadata_line}")
        image_id = int(parts[0])
        point_parts = points_line.split()
        if len(point_parts) % 3 != 0:
            raise Phase5ExperimentError(f"Malformed images.txt points line for image {image_id}")
        points2d = [
            (
                float(point_parts[index]),
                float(point_parts[index + 1]),
                int(point_parts[index + 2]),
            )
            for index in range(0, len(point_parts), 3)
        ]
        images[image_id] = ColmapImageRecord(
            image_id=image_id,
            qvec=[float(value) for value in parts[1:5]],
            tvec=[float(value) for value in parts[5:8]],
            camera_id=int(parts[8]),
            name=" ".join(parts[9:]),
            points2d=points2d,
        )

    points3d: dict[int, ColmapPoint3DRecord] = {}
    for line in _iter_data_lines(root / "points3D.txt"):
        parts = line.split()
        if len(parts) < 8:
            raise Phase5ExperimentError(f"Malformed points3D.txt line: {line}")
        track_values = parts[8:]
        if len(track_values) % 2 != 0:
            raise Phase5ExperimentError(f"Malformed points3D track: {line}")
        points3d[int(parts[0])] = ColmapPoint3DRecord(
            point3d_id=int(parts[0]),
            xyz=[float(value) for value in parts[1:4]],
            rgb=(int(parts[4]), int(parts[5]), int(parts[6])),
            error=float(parts[7]),
            track=[
                (int(track_values[index]), int(track_values[index + 1]))
                for index in range(0, len(track_values), 2)
            ],
        )

    return ColmapRawModel(cameras=cameras, images=images, points3d=points3d, model_format="text")


def read_raw_colmap_model(model_path: str | Path, *, model_format: str = "auto") -> ColmapRawModel:
    """Read a COLMAP sparse model for filtering."""

    root = Path(model_path)
    if not root.exists() or not root.is_dir():
        raise Phase5ExperimentError(f"Invalid COLMAP sparse model directory: {root}")
    resolved_format = detect_model_format(root) if model_format == "auto" else model_format
    if resolved_format == "binary":
        return read_raw_colmap_binary(root)
    if resolved_format == "text":
        return read_raw_colmap_text(root)
    raise Phase5ExperimentError("model_format must be 'auto', 'binary', or 'text'.")


def filter_colmap_model(
    model: ColmapRawModel,
    image_names: Sequence[str],
    *,
    min_track_length: int = 1,
) -> ColmapRawModel:
    """Filter cameras/images/points to a selected image-name subset."""

    requested_names = set(image_names)
    if not requested_names:
        raise Phase5ExperimentError("Cannot filter a COLMAP model to zero images.")
    selected_images = {
        image_id: image
        for image_id, image in model.images.items()
        if image.name in requested_names
    }
    missing = sorted(requested_names - {image.name for image in selected_images.values()})
    if missing:
        raise Phase5ExperimentError(f"Selected images missing from COLMAP model: {missing[:5]}")
    selected_image_ids = set(selected_images)

    filtered_points: dict[int, ColmapPoint3DRecord] = {}
    for point_id, point in model.points3d.items():
        track = [
            (image_id, point2d_index)
            for image_id, point2d_index in point.track
            if image_id in selected_image_ids
        ]
        if len(track) >= min_track_length:
            filtered_points[point_id] = ColmapPoint3DRecord(
                point3d_id=point.point3d_id,
                xyz=list(point.xyz),
                rgb=point.rgb,
                error=point.error,
                track=track,
            )
    valid_point_ids = set(filtered_points)

    rewritten_images = {}
    for image_id, image in selected_images.items():
        points2d = [
            (x, y, point_id if _is_valid_point3d_id(point_id) and point_id in valid_point_ids else -1)
            for x, y, point_id in image.points2d
        ]
        rewritten_images[image_id] = ColmapImageRecord(
            image_id=image.image_id,
            qvec=list(image.qvec),
            tvec=list(image.tvec),
            camera_id=image.camera_id,
            name=image.name,
            points2d=points2d,
        )

    camera_ids = {image.camera_id for image in rewritten_images.values()}
    cameras = {camera_id: model.cameras[camera_id] for camera_id in sorted(camera_ids)}
    return ColmapRawModel(
        cameras=cameras,
        images=rewritten_images,
        points3d=filtered_points,
        model_format="binary",
    )


def write_colmap_binary_model(model: ColmapRawModel, output_dir: str | Path) -> None:
    """Write cameras/images/points3D as a binary COLMAP sparse model."""

    target = ensure_dir(output_dir)
    with (target / "cameras.bin").open("wb") as file:
        file.write(struct.pack("<Q", len(model.cameras)))
        for camera in sorted(model.cameras.values(), key=lambda item: item.camera_id):
            if camera.model not in CAMERA_MODEL_IDS:
                raise Phase5ExperimentError(f"Unsupported camera model for writing: {camera.model}")
            model_id, expected_params = CAMERA_MODEL_IDS[camera.model]
            if len(camera.params) != expected_params:
                raise Phase5ExperimentError(
                    f"Camera {camera.camera_id} model {camera.model} expects "
                    f"{expected_params} params, found {len(camera.params)}."
                )
            file.write(struct.pack("<iiQQ", camera.camera_id, model_id, camera.width, camera.height))
            file.write(struct.pack("<" + "d" * len(camera.params), *camera.params))

    with (target / "images.bin").open("wb") as file:
        file.write(struct.pack("<Q", len(model.images)))
        for image in sorted(model.images.values(), key=lambda item: item.image_id):
            file.write(
                struct.pack(
                    "<idddddddi",
                    image.image_id,
                    *image.qvec,
                    *image.tvec,
                    image.camera_id,
                )
            )
            file.write(image.name.encode("utf-8") + b"\x00")
            file.write(struct.pack("<Q", len(image.points2d)))
            for x, y, point3d_id in image.points2d:
                file.write(struct.pack("<ddq", x, y, int(point3d_id)))

    with (target / "points3D.bin").open("wb") as file:
        file.write(struct.pack("<Q", len(model.points3d)))
        for point in sorted(model.points3d.values(), key=lambda item: item.point3d_id):
            file.write(struct.pack("<Q", point.point3d_id))
            file.write(struct.pack("<ddd", *point.xyz))
            file.write(struct.pack("<BBB", *point.rgb))
            file.write(struct.pack("<d", point.error))
            file.write(struct.pack("<Q", len(point.track)))
            for image_id, point2d_index in point.track:
                file.write(struct.pack("<ii", image_id, point2d_index))


def _copy_or_link_image(source: Path, destination: Path, *, link_mode: str) -> str:
    ensure_dir(destination.parent)
    if link_mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    if link_mode == "symlink":
        try:
            destination.symlink_to(source)
            return "symlink"
        except OSError:
            shutil.copy2(source, destination)
            return "copy"
    if link_mode != "hardlink":
        raise Phase5ExperimentError("link_mode must be one of: hardlink, symlink, copy.")
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _materialize_images(
    image_names: Sequence[str],
    *,
    source_images: Path,
    target_images: Path,
    link_mode: str,
) -> str:
    actual_mode = link_mode
    image_lookup = {path.name: path for path in discover_images(source_images)}
    missing = [name for name in image_names if name not in image_lookup]
    if missing:
        raise Phase5ExperimentError(f"Images missing from source directory: {missing[:5]}")
    for name in image_names:
        used_mode = _copy_or_link_image(image_lookup[name], target_images / name, link_mode=link_mode)
        if used_mode != link_mode:
            actual_mode = used_mode
    return actual_mode


def prepare_phase5_scene(
    *,
    manifest: dict[str, Any] | ExperimentManifest,
    source_images: str | Path,
    source_model: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    link_mode: str = "hardlink",
    dry_run: bool = False,
) -> PreparedScene:
    """Prepare train and held-out render scenes for one experiment manifest."""

    manifest_dict = manifest.to_dict() if isinstance(manifest, ExperimentManifest) else dict(manifest)
    validate_experiment_manifest(manifest_dict)
    experiment_name = str(manifest_dict["experiment_name"])
    train_images = list(manifest_dict["training_images"])
    test_images = list(manifest_dict["test_images"])
    root = Path(output_dir)

    if dry_run:
        return PreparedScene(
            experiment_name=experiment_name,
            output_dir=str(root),
            train_scene=str(root / "train_scene"),
            test_scene=str(root / "test_scene"),
            train_image_count=len(train_images),
            test_image_count=len(test_images),
            train_sparse_points=0,
            test_sparse_points=0,
            link_mode=link_mode,
            dry_run=True,
        )

    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise Phase5ExperimentError(f"Output directory already exists: {root}")
        shutil.rmtree(root)

    source_images_path = Path(source_images)
    source_model_path = Path(source_model)
    raw_model = read_raw_colmap_model(source_model_path)
    train_model = filter_colmap_model(raw_model, train_images)
    test_model = filter_colmap_model(raw_model, test_images)

    train_scene = root / "train_scene"
    test_scene = root / "test_scene"
    used_mode = _materialize_images(
        train_images,
        source_images=source_images_path,
        target_images=train_scene / "images",
        link_mode=link_mode,
    )
    test_used_mode = _materialize_images(
        test_images,
        source_images=source_images_path,
        target_images=test_scene / "images",
        link_mode=link_mode,
    )
    if test_used_mode != used_mode:
        used_mode = "mixed"

    write_colmap_binary_model(train_model, train_scene / "sparse" / "0")
    write_colmap_binary_model(test_model, test_scene / "sparse" / "0")

    summary = PreparedScene(
        experiment_name=experiment_name,
        output_dir=str(root),
        train_scene=str(train_scene),
        test_scene=str(test_scene),
        train_image_count=len(train_images),
        test_image_count=len(test_images),
        train_sparse_points=len(train_model.points3d),
        test_sparse_points=len(test_model.points3d),
        link_mode=used_mode,
        dry_run=False,
    )
    write_json(root / "scene_manifest.json", {"prepared_scene": asdict(summary), "experiment": manifest_dict})
    return summary


def compute_quality_signals(
    *,
    image_dir: str | Path,
    image_names: Sequence[str],
    quality_config: dict[str, Any],
    frame_selection_config: dict[str, Any],
) -> dict[str, ImageQualitySignals]:
    """Compute optional Phase 2 quality signals for public-dataset frames."""

    try:
        from src.frame_quality import (
            FrameQualityError,
            FrameSelectionConfig,
            QualityThresholds,
            analyze_frame,
            create_feature_detector,
        )
    except Exception as exc:  # pragma: no cover - depends on optional OpenCV install
        raise Phase5ExperimentError("OpenCV quality analysis is unavailable.") from exc

    thresholds = QualityThresholds.from_config(quality_config)
    selector_config = FrameSelectionConfig.from_config(frame_selection_config)
    try:
        detector = create_feature_detector(selector_config.feature_detector, selector_config.maximum_features)
    except FrameQualityError as exc:
        raise Phase5ExperimentError(str(exc)) from exc

    image_lookup = {path.name: path for path in discover_images(image_dir)}
    signals: dict[str, ImageQualitySignals] = {}
    for index, name in enumerate(image_names, start=1):
        path = image_lookup.get(name)
        if path is None:
            signals[name] = ImageQualitySignals(valid=False, rejection_reason="missing_image")
            continue
        analysis = analyze_frame(
            path,
            thresholds=thresholds,
            detector=detector,
            frame_id=index,
        )
        result = analysis.result
        signals[name] = ImageQualitySignals(
            valid=result.selection_reason == "selected",
            blur_score=result.blur_score,
            mean_brightness=result.mean_brightness,
            dark_pixel_ratio=result.dark_pixel_ratio,
            bright_pixel_ratio=result.bright_pixel_ratio,
            feature_count=result.feature_count,
            rejection_reason=None if result.selection_reason == "selected" else result.selection_reason,
        )
    return signals


def generate_phase5_manifests(
    *,
    source_model: str | Path,
    manifest_dir: str | Path,
    source_images: str | Path | None = None,
    config: dict[str, Any] | None = None,
    include_full: bool = False,
    compute_quality: bool = True,
) -> dict[str, Any]:
    """Generate the common split and default Phase 5 experiment manifests."""

    config = config or load_config("config.example.yaml")
    phase5_config = config.get("phase5", {})
    dataset = str(config["dataset"]["name"])
    scene = str(config["dataset"]["scene"])
    split_config = phase5_config.get("split", {})
    seed = int(config["reproducibility"]["random_seed"])
    ordered_images = ordered_image_names_from_model(source_model)
    split = create_heldout_split(
        ordered_images,
        dataset=dataset,
        scene=scene,
        holdout_interval=int(split_config.get("holdout_interval", 8)),
        holdout_offset=int(split_config.get("holdout_offset", 0)),
        seed=seed,
    )

    target_dir = ensure_dir(manifest_dir)
    write_split_manifest(target_dir / "split_manifest.json", split)

    experiments: dict[str, ExperimentManifest] = {}
    iterations = int(phase5_config.get("iterations", config["gaussian"]["iterations"]))
    for experiment_name, (method, budget) in _configured_experiment_specs(config).items():
        if method == "fixed_uniform":
            selected = fixed_budget_selection(split, budget)
            details = {
                "algorithm": "uniform_sample_over_training_candidates",
                "budget": budget,
                "ordered_candidate_count": len(split.training_candidate_images),
            }
        else:
            quality_by_name = None
            quality_status = "not_requested"
            if compute_quality and source_images is not None:
                try:
                    quality_by_name = compute_quality_signals(
                        image_dir=source_images,
                        image_names=split.training_candidate_images,
                        quality_config=config["quality"],
                        frame_selection_config=config["frame_selection"],
                    )
                    quality_status = "computed"
                except Phase5ExperimentError as exc:
                    quality_status = f"unavailable: {exc}"
            candidates = build_pose_candidates(
                split.training_candidate_images,
                model_path=source_model,
                quality_by_name=quality_by_name,
            )
            selected, details = automatic_budget_selection(candidates, budget=budget)
            details["quality_status"] = quality_status
        manifest = build_experiment_manifest(
            split=split,
            experiment_name=experiment_name,
            selection_method=method,
            training_images=selected,
            training_budget=budget,
            selection_details=details,
            iterations=iterations,
        )
        experiments[experiment_name] = manifest
        write_experiment_manifest(target_dir / f"{experiment_name}.json", manifest)

    if include_full:
        manifest = build_experiment_manifest(
            split=split,
            experiment_name="full",
            selection_method="all_training_candidates",
            training_images=split.training_candidate_images,
            training_budget=len(split.training_candidate_images),
            selection_details={"algorithm": "all_non_heldout_images"},
            iterations=iterations,
        )
        experiments["full"] = manifest
        write_experiment_manifest(target_dir / "full.json", manifest)

    index = {
        "manifest_version": 1,
        "dataset": dataset,
        "scene": scene,
        "generated_at_utc": utc_timestamp(),
        "source_model": _display_path(source_model),
        "source_images": _display_path(source_images),
        "graphdeco_repository": OFFICIAL_3DGS_REPO_URL,
        "graphdeco_commit": OFFICIAL_3DGS_COMMIT,
        "split_manifest": "split_manifest.json",
        "experiments": sorted(experiments),
        "notes": "Generated locally for Phase 5 preparation. No training has been run.",
    }
    write_json(target_dir / "index.json", index)
    return {"split": split.to_dict(), "experiments": {name: item.to_dict() for name, item in experiments.items()}}


def load_experiment_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_experiment_manifest(data)
    return data


def validate_experiment_name(name: str) -> str:
    valid_budget_name = False
    for prefix in ("fixed_", "automatic_"):
        if name.startswith(prefix):
            suffix = name.removeprefix(prefix)
            valid_budget_name = suffix.isdigit() and int(suffix) > 0
    if name != "full" and not valid_budget_name:
        raise Phase5ExperimentError(
            f"Unknown Phase 5 experiment '{name}'. Expected fixed_N, automatic_N, or full."
        )
    return name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase 5 3DGS experiment manifests and scenes.")
    parser.add_argument("--config", default="config.example.yaml")
    parser.add_argument("--manifest-dir", default="experiments/phase5")
    parser.add_argument("--source-images", default="data/public/mipnerf360/bonsai/images_4")
    parser.add_argument("--source-model", default="outputs/mipnerf_bonsai_colmap/sparse/0")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--generate-manifests", action="store_true")
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--skip-quality-analysis", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--link-mode", choices=["hardlink", "symlink", "copy"], default="hardlink")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_config(args.config, allow_example_fallback=True)
        source_images = resolve_project_path(args.source_images)
        source_model = resolve_project_path(args.source_model)
        manifest_dir = resolve_project_path(args.manifest_dir)

        if args.generate_manifests:
            result = generate_phase5_manifests(
                source_model=source_model,
                source_images=source_images,
                manifest_dir=manifest_dir,
                config=config,
                include_full=args.include_full,
                compute_quality=not args.skip_quality_analysis,
            )
            print(json.dumps({"generated_manifests": sorted(result["experiments"])}, indent=2))

        if args.experiment:
            experiment = validate_experiment_name(args.experiment)
            manifest_path = manifest_dir / f"{experiment}.json"
            manifest = load_experiment_manifest(manifest_path)
            output = Path(args.output) if args.output else Path("outputs") / "phase5" / experiment
            scene = prepare_phase5_scene(
                manifest=manifest,
                source_images=source_images,
                source_model=source_model,
                output_dir=resolve_project_path(output),
                overwrite=args.overwrite,
                link_mode=args.link_mode,
                dry_run=args.dry_run,
            )
            print(json.dumps(asdict(scene), indent=2))

        if not args.generate_manifests and not args.experiment:
            raise Phase5ExperimentError("Use --generate-manifests and/or --experiment.")
    except (Phase5ExperimentError, Phase5SelectionError, OSError, json.JSONDecodeError) as exc:
        print(f"Phase 5 preparation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
