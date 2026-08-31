"""Smartphone video validation and candidate frame extraction."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:  # Allows: python src/extract_frames.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ensure_dir

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover - depends on local environment
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None


class FrameExtractionError(RuntimeError):
    """Raised when a video cannot be validated or extracted."""


@dataclass(frozen=True)
class VideoMetadata:
    """Basic metadata reported by OpenCV for a validated video."""

    path: str
    fps: float
    total_frames: int
    duration_seconds: float
    width: int
    height: int


@dataclass(frozen=True)
class ExtractedFrame:
    """Metadata for one extracted candidate frame."""

    frame_id: int
    source_frame_index: int
    timestamp_seconds: float
    filename: str
    path: str


def _require_cv2() -> None:
    if cv2 is None:
        raise FrameExtractionError(
            "OpenCV is required for video frame extraction. Install opencv-python."
        ) from _CV2_IMPORT_ERROR


def open_video(input_path: str | Path) -> tuple[object, VideoMetadata]:
    """Open and validate a video, returning an OpenCV capture and metadata."""

    _require_cv2()
    video_path = Path(input_path).expanduser()
    if not video_path.exists():
        raise FrameExtractionError(f"Input video does not exist: {video_path}")
    if not video_path.is_file():
        raise FrameExtractionError(f"Input path is not a file: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise FrameExtractionError(f"OpenCV could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if fps <= 0:
        capture.release()
        raise FrameExtractionError(f"Video FPS is invalid: {fps}")
    if total_frames <= 0:
        capture.release()
        raise FrameExtractionError(f"Video frame count is invalid: {total_frames}")
    if width <= 0 or height <= 0:
        capture.release()
        raise FrameExtractionError(f"Video dimensions are invalid: {width}x{height}")

    metadata = VideoMetadata(
        path=str(video_path),
        fps=fps,
        total_frames=total_frames,
        duration_seconds=total_frames / fps,
        width=width,
        height=height,
    )
    return capture, metadata


def select_frame_indices(
    total_frames: int,
    fps: float,
    *,
    frame_interval: int | None = None,
    time_interval: float | None = None,
    target_frames: int | None = None,
) -> list[int]:
    """Choose chronological frame indices for exactly one extraction strategy."""

    strategies = [frame_interval is not None, time_interval is not None, target_frames is not None]
    if sum(strategies) != 1:
        raise FrameExtractionError(
            "Use exactly one extraction strategy: frame_interval, time_interval, or target_frames."
        )
    if total_frames <= 0:
        raise FrameExtractionError("total_frames must be positive.")
    if fps <= 0:
        raise FrameExtractionError("fps must be positive.")

    if frame_interval is not None:
        if frame_interval <= 0:
            raise FrameExtractionError("frame_interval must be positive.")
        return list(range(0, total_frames, frame_interval))

    if time_interval is not None:
        if time_interval <= 0:
            raise FrameExtractionError("time_interval must be positive.")
        step = max(1, int(round(time_interval * fps)))
        return list(range(0, total_frames, step))

    assert target_frames is not None
    if target_frames <= 0:
        raise FrameExtractionError("target_frames must be positive.")
    count = min(target_frames, total_frames)
    if count == 1:
        return [0]
    span = total_frames - 1
    indices = [round(i * span / (count - 1)) for i in range(count)]
    return sorted(dict.fromkeys(indices))


def extraction_strategy_from_config(config: dict) -> dict[str, int | float | None]:
    """Extract the configured video sampling strategy."""

    video_config = config.get("video", {})
    return {
        "frame_interval": video_config.get("frame_interval"),
        "time_interval": video_config.get("time_interval"),
        "target_frames": video_config.get("target_frames"),
    }


def _write_frame(output_path: Path, frame, jpeg_quality: int) -> None:
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    ok = cv2.imwrite(str(output_path), frame, params)
    if not ok:
        raise FrameExtractionError(f"Failed to write extracted frame: {output_path}")


def extract_frames(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    frame_interval: int | None = None,
    time_interval: float | None = None,
    target_frames: int | None = None,
    jpeg_quality: int = 95,
    logger: logging.Logger | None = None,
) -> tuple[VideoMetadata, list[ExtractedFrame]]:
    """Extract candidate frames from a smartphone video."""

    logger = logger or logging.getLogger("smartphone_3dgs.extract_frames")
    output_path = ensure_dir(output_dir)
    capture, metadata = open_video(input_path)
    desired_indices = select_frame_indices(
        metadata.total_frames,
        metadata.fps,
        frame_interval=frame_interval,
        time_interval=time_interval,
        target_frames=target_frames,
    )
    desired_lookup = set(desired_indices)
    extracted: list[ExtractedFrame] = []

    strategy = {
        "frame_interval": frame_interval,
        "time_interval": time_interval,
        "target_frames": target_frames,
    }
    logger.info(
        "Video metadata: fps=%.3f total_frames=%d duration=%.3fs width=%d height=%d",
        metadata.fps,
        metadata.total_frames,
        metadata.duration_seconds,
        metadata.width,
        metadata.height,
    )
    logger.info("Extraction strategy: %s", {k: v for k, v in strategy.items() if v is not None})

    try:
        source_index = 0
        while source_index < metadata.total_frames:
            ok, frame = capture.read()
            if not ok:
                if source_index in desired_lookup:
                    logger.warning("Could not read requested frame index %d", source_index)
                break
            if source_index in desired_lookup:
                frame_id = len(extracted) + 1
                filename = f"frame_{frame_id:06d}.jpg"
                target_path = output_path / filename
                _write_frame(target_path, frame, jpeg_quality)
                extracted.append(
                    ExtractedFrame(
                        frame_id=frame_id,
                        source_frame_index=source_index,
                        timestamp_seconds=source_index / metadata.fps,
                        filename=filename,
                        path=str(target_path),
                    )
                )
            source_index += 1
    finally:
        capture.release()

    if not extracted:
        raise FrameExtractionError("No frames were extracted from the video.")

    logger.info("Extracted %d candidate frames to %s", len(extracted), output_path)
    return metadata, extracted


def frames_to_jsonable(frames: Iterable[ExtractedFrame]) -> list[dict]:
    """Convert extracted frame metadata to JSON-friendly dictionaries."""

    return [asdict(frame) for frame in frames]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract candidate frames from a smartphone video.")
    parser.add_argument("--input", required=True, help="Input MP4/MOV video path.")
    parser.add_argument("--output", required=True, help="Output directory for candidate JPEG frames.")
    strategy = parser.add_mutually_exclusive_group()
    strategy.add_argument("--frame-interval", type=int, default=None, help="Extract every N frames.")
    strategy.add_argument("--time-interval", type=float, default=None, help="Extract every N seconds.")
    strategy.add_argument("--target-frames", type=int, default=None, help="Extract approximately this many frames.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for candidate frames.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    frame_interval = args.frame_interval
    if frame_interval is None and args.time_interval is None and args.target_frames is None:
        frame_interval = 10

    try:
        metadata, frames = extract_frames(
            args.input,
            args.output,
            frame_interval=frame_interval,
            time_interval=args.time_interval,
            target_frames=args.target_frames,
            jpeg_quality=args.jpeg_quality,
        )
    except FrameExtractionError as exc:
        logging.getLogger("smartphone_3dgs.extract_frames").error("%s", exc)
        return 2

    print(json.dumps({"video": asdict(metadata), "frames": frames_to_jsonable(frames)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
