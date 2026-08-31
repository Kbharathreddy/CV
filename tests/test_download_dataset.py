from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from src.download_dataset import (
    DatasetDownloadError,
    extract_scene_from_archive,
    get_mipnerf360_source,
    validate_rgb_images,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (8, 6), color=(120, 80, 40))
    image.save(path, format="JPEG")


def test_bonsai_uses_official_base_archive() -> None:
    source = get_mipnerf360_source("bonsai")

    assert source.filename == "360_v2.zip"
    assert source.url.startswith("https://storage.googleapis.com/gresearch/refraw360/")


def test_extract_scene_from_archive_selects_requested_scene(tmp_path: Path) -> None:
    scene_root = tmp_path / "source"
    _write_image(scene_root / "bonsai" / "images" / "frame_0001.jpg")
    _write_image(scene_root / "room" / "images" / "frame_0001.jpg")

    archive_path = tmp_path / "360_v2.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for image_path in scene_root.rglob("*.jpg"):
            archive.write(image_path, image_path.relative_to(scene_root).as_posix())

    destination = tmp_path / "data" / "public" / "mipnerf360" / "bonsai"
    extract_scene_from_archive(archive_path, "bonsai", destination, logging.getLogger("test"))

    assert (destination / "images" / "frame_0001.jpg").exists()
    assert not (destination.parent / "room").exists()
    assert validate_rgb_images(destination) == 1


def test_extract_scene_rejects_missing_scene(tmp_path: Path) -> None:
    archive_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("room/images/frame_0001.jpg", b"not-a-real-image")

    with pytest.raises(DatasetDownloadError, match="was not found"):
        extract_scene_from_archive(
            archive_path,
            "bonsai",
            tmp_path / "bonsai",
            logging.getLogger("test"),
        )
