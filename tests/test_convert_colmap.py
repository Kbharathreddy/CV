from __future__ import annotations

import struct
from pathlib import Path

import pytest

from src.convert_colmap import (
    ColmapModelError,
    camera_center_from_pose,
    read_colmap_model,
    summarize_model,
)


def test_camera_center_from_identity_pose() -> None:
    center = camera_center_from_pose([1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0])

    assert center == pytest.approx([-1.0, -2.0, -3.0])


def test_read_text_colmap_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "sparse" / "0"
    model_dir.mkdir(parents=True)
    (model_dir / "cameras.txt").write_text("1 PINHOLE 640 480 500 500 320 240\n", encoding="utf-8")
    (model_dir / "images.txt").write_text(
        "1 1 0 0 0 1 2 3 1 image001.jpg\n"
        "10 20 100 30 40 -1\n",
        encoding="utf-8",
    )
    (model_dir / "points3D.txt").write_text(
        "100 1.0 2.0 3.0 255 128 64 0.25 1 0\n",
        encoding="utf-8",
    )

    model = read_colmap_model(model_dir)
    summary = summarize_model(model, total_images=2)

    assert model.model_format == "text"
    assert model.cameras[1].model == "PINHOLE"
    assert model.images[1].name == "image001.jpg"
    assert model.images[1].camera_center == pytest.approx([-1.0, -2.0, -3.0])
    assert model.points3d[100].track_length == 1
    assert summary["registered_images"] == 1
    assert summary["registration_rate"] == 0.5
    assert summary["number_of_sparse_points"] == 1
    assert summary["number_of_observations"] == 1


def test_read_binary_colmap_model(tmp_path: Path) -> None:
    model_dir = tmp_path / "sparse" / "0"
    model_dir.mkdir(parents=True)
    with (model_dir / "cameras.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<iiQQ", 1, 1, 640, 480))
        file.write(struct.pack("<dddd", 500.0, 500.0, 320.0, 240.0))
    with (model_dir / "images.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<idddddddi", 1, 1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1))
        file.write(b"image001.jpg\x00")
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<ddq", 10.0, 20.0, 100))
    with (model_dir / "points3D.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<Q", 100))
        file.write(struct.pack("<ddd", 1.0, 2.0, 3.0))
        file.write(struct.pack("<BBB", 255, 128, 64))
        file.write(struct.pack("<d", 0.25))
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<ii", 1, 0))

    model = read_colmap_model(model_dir)

    assert model.model_format == "binary"
    assert model.cameras[1].params == pytest.approx([500.0, 500.0, 320.0, 240.0])
    assert model.images[1].tvec == pytest.approx([1.0, 2.0, 3.0])
    assert model.points3d[100].xyz == pytest.approx([1.0, 2.0, 3.0])


def test_read_binary_image_ignores_unsigned_missing_point_sentinel(tmp_path: Path) -> None:
    model_dir = tmp_path / "sparse" / "0"
    model_dir.mkdir(parents=True)
    with (model_dir / "cameras.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<iiQQ", 1, 1, 640, 480))
        file.write(struct.pack("<dddd", 500.0, 500.0, 320.0, 240.0))
    with (model_dir / "images.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<idddddddi", 1, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1))
        file.write(b"image001.jpg\x00")
        file.write(struct.pack("<Q", 2))
        file.write(struct.pack("<ddQ", 10.0, 20.0, 2**64 - 1))
        file.write(struct.pack("<ddq", 30.0, 40.0, 100))
    with (model_dir / "points3D.bin").open("wb") as file:
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<Q", 100))
        file.write(struct.pack("<ddd", 1.0, 2.0, 3.0))
        file.write(struct.pack("<BBB", 255, 128, 64))
        file.write(struct.pack("<d", 0.25))
        file.write(struct.pack("<Q", 1))
        file.write(struct.pack("<ii", 1, 1))

    model = read_colmap_model(model_dir)

    assert model.images[1].num_points2d == 2
    assert model.images[1].num_observed_points == 1


def test_missing_model_files_raise_clear_error(tmp_path: Path) -> None:
    model_dir = tmp_path / "empty"
    model_dir.mkdir()

    with pytest.raises(ColmapModelError, match="Missing COLMAP sparse model"):
        read_colmap_model(model_dir)
