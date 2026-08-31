"""Image preprocessing for selected reconstruction frames."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:  # Allows: python src/preprocess.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageOps

from src.utils import discover_images, ensure_dir


class PreprocessError(RuntimeError):
    """Raised when selected images cannot be preprocessed."""


@dataclass(frozen=True)
class PreprocessConfig:
    """Configurable image preprocessing settings."""

    max_width: int
    max_height: int
    jpeg_quality: int

    @classmethod
    def from_config(cls, config: dict) -> "PreprocessConfig":
        return cls(
            max_width=int(config["max_width"]),
            max_height=int(config["max_height"]),
            jpeg_quality=int(config["jpeg_quality"]),
        )


@dataclass(frozen=True)
class PreprocessResult:
    """Metadata for one preprocessed selected frame."""

    input_path: str
    output_path: str
    original_width: int
    original_height: int
    output_width: int
    output_height: int
    resized: bool


def _resized_dimensions(width: int, height: int, max_width: int, max_height: int) -> tuple[int, int]:
    scale = min(max_width / width, max_height / height, 1.0)
    return max(1, round(width * scale)), max(1, round(height * scale))


def preprocess_image(
    input_path: str | Path,
    output_path: str | Path,
    config: PreprocessConfig,
) -> PreprocessResult:
    """Validate, orient, resize without upscaling, convert to RGB, and save JPEG."""

    source = Path(input_path)
    destination = Path(output_path)
    if not source.exists() or not source.is_file():
        raise PreprocessError(f"Selected image does not exist: {source}")

    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            original_width, original_height = image.size
            if original_width <= 0 or original_height <= 0:
                raise PreprocessError(f"Invalid image dimensions: {source}")
            output_width, output_height = _resized_dimensions(
                original_width,
                original_height,
                config.max_width,
                config.max_height,
            )
            resized = (output_width, output_height) != (original_width, original_height)
            if resized:
                image = image.resize((output_width, output_height), Image.Resampling.LANCZOS)
            ensure_dir(destination.parent)
            image.save(destination, format="JPEG", quality=config.jpeg_quality)
    except PreprocessError:
        raise
    except Exception as exc:
        raise PreprocessError(f"Failed to preprocess image {source}: {exc}") from exc

    return PreprocessResult(
        input_path=str(source),
        output_path=str(destination),
        original_width=original_width,
        original_height=original_height,
        output_width=output_width,
        output_height=output_height,
        resized=resized,
    )


def preprocess_images(
    image_paths: Sequence[str | Path],
    output_dir: str | Path,
    config: PreprocessConfig,
    *,
    logger: logging.Logger | None = None,
) -> list[PreprocessResult]:
    """Preprocess selected frames into a consistent RGB JPEG directory."""

    logger = logger or logging.getLogger("smartphone_3dgs.preprocess")
    destination_dir = ensure_dir(output_dir)
    results: list[PreprocessResult] = []
    logger.info(
        "Preprocessing selected frames: count=%d max_size=%dx%d jpeg_quality=%d",
        len(image_paths),
        config.max_width,
        config.max_height,
        config.jpeg_quality,
    )
    for path_like in image_paths:
        source = Path(path_like)
        destination = destination_dir / source.name
        result = preprocess_image(source, destination, config)
        results.append(result)
        logger.debug("Preprocessed frame: %s", asdict(result))
    logger.info("Preprocessed %d selected frames to %s", len(results), destination_dir)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess selected frames.")
    parser.add_argument("--input-dir", required=True, help="Input selected/candidate image directory.")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed RGB JPEG images.")
    parser.add_argument("--max-width", type=int, default=1600)
    parser.add_argument("--max-height", type=int, default=1200)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config = PreprocessConfig(args.max_width, args.max_height, args.jpeg_quality)
    try:
        results = preprocess_images(discover_images(args.input_dir), args.output_dir, config)
    except PreprocessError as exc:
        logging.getLogger("smartphone_3dgs.preprocess").error("%s", exc)
        return 2
    print(json.dumps([asdict(result) for result in results], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
