from __future__ import annotations

from pathlib import Path

import pytest

from src.run_colmap import (
    ColmapRunError,
    build_colmap_commands,
    choose_matcher,
    find_sparse_model,
    prepare_output_directory,
    select_colmap_backend,
    validate_image_directory,
)


def test_choose_matcher_uses_dataset_and_video_defaults() -> None:
    assert choose_matcher(matcher="auto", input_type="dataset") == "exhaustive"
    assert choose_matcher(matcher="auto", input_type="video") == "sequential"
    assert choose_matcher(matcher="sequential", input_type="dataset") == "sequential"


def test_backend_auto_prefers_pycolmap_when_importable() -> None:
    assert select_colmap_backend("auto", pycolmap_importable=True) == "pycolmap"


def test_backend_auto_falls_back_to_cli_when_pycolmap_missing() -> None:
    assert select_colmap_backend("auto", pycolmap_importable=False) == "cli"


def test_explicit_pycolmap_backend_requires_importable_package() -> None:
    assert select_colmap_backend("pycolmap", pycolmap_importable=True) == "pycolmap"

    with pytest.raises(ColmapRunError, match="PyCOLMAP backend was requested"):
        select_colmap_backend("pycolmap", pycolmap_importable=False)


def test_explicit_cli_backend_remains_supported() -> None:
    assert select_colmap_backend("cli", pycolmap_importable=False) == "cli"


def test_validate_image_directory_rejects_missing_or_empty(tmp_path: Path) -> None:
    with pytest.raises(ColmapRunError, match="Missing image directory"):
        validate_image_directory(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ColmapRunError, match="No images"):
        validate_image_directory(empty)


def test_build_colmap_commands_for_exhaustive_matcher(tmp_path: Path) -> None:
    commands = build_colmap_commands(
        colmap_path="colmap",
        image_dir=tmp_path / "images",
        database_path=tmp_path / "database" / "database.db",
        sparse_dir=tmp_path / "sparse",
        matcher="exhaustive",
        camera_model="OPENCV",
        single_camera=True,
        sequential_overlap=10,
    )

    assert commands["feature_extractor"][1] == "feature_extractor"
    assert "--ImageReader.camera_model" in commands["feature_extractor"]
    assert commands["matcher"][1] == "exhaustive_matcher"
    assert commands["mapper"][1] == "mapper"


def test_find_sparse_model_prefers_numbered_model_directory(tmp_path: Path) -> None:
    sparse_root = tmp_path / "sparse"
    model_dir = sparse_root / "0"
    model_dir.mkdir(parents=True)
    for filename in ("cameras.txt", "images.txt", "points3D.txt"):
        (model_dir / filename).write_text("", encoding="utf-8")

    assert find_sparse_model(sparse_root) == model_dir


def test_prepare_output_directory_uses_database_subdirectory(tmp_path: Path) -> None:
    root, database_path, sparse_dir, log_dir = prepare_output_directory(tmp_path / "experiment", overwrite=False)

    assert root == tmp_path / "experiment"
    assert database_path == root / "database" / "database.db"
    assert database_path.parent.is_dir()
    assert sparse_dir.is_dir()
    assert log_dir.is_dir()
