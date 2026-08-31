from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.preprocess import PreprocessConfig, preprocess_image


def _write_image(path: Path, size: tuple[int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(90, 120, 150)).save(path, format="JPEG")
    return path


def test_preprocessing_preserves_aspect_ratio(tmp_path: Path) -> None:
    source = _write_image(tmp_path / "wide.jpg", (800, 400))
    destination = tmp_path / "out" / "wide.jpg"

    result = preprocess_image(source, destination, PreprocessConfig(400, 400, 95))

    assert result.output_width == 400
    assert result.output_height == 200
    with Image.open(destination) as image:
        assert image.size == (400, 200)
        assert image.mode == "RGB"


def test_preprocessing_does_not_upscale(tmp_path: Path) -> None:
    source = _write_image(tmp_path / "small.jpg", (100, 50))
    destination = tmp_path / "out" / "small.jpg"

    result = preprocess_image(source, destination, PreprocessConfig(400, 400, 95))

    assert result.output_width == 100
    assert result.output_height == 50
    assert result.resized is False
