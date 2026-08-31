from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.extract_frames import (
    FrameExtractionError,
    extract_frames,
    open_video,
    select_frame_indices,
)


def _create_test_video(path: Path, *, frame_count: int = 12, fps: float = 6.0) -> Path:
    size = (64, 48)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        path = path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    assert writer.isOpened()

    for index in range(frame_count):
        frame = np.full((size[1], size[0], 3), 20 + index * 10, dtype=np.uint8)
        cv2.circle(frame, (8 + index * 3, 20), 5, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


def test_temporary_video_can_be_opened(tmp_path: Path) -> None:
    video_path = _create_test_video(tmp_path / "input.mp4")

    capture, metadata = open_video(video_path)
    capture.release()

    assert metadata.total_frames == 12
    assert metadata.fps == pytest.approx(6.0, rel=0.05)
    assert metadata.width == 64
    assert metadata.height == 48


def test_frame_indices_support_all_strategies() -> None:
    assert select_frame_indices(12, 6.0, frame_interval=3) == [0, 3, 6, 9]
    assert select_frame_indices(12, 6.0, time_interval=0.5) == [0, 3, 6, 9]
    assert select_frame_indices(10, 5.0, target_frames=4) == [0, 3, 6, 9]


def test_frame_indices_reject_ambiguous_strategy() -> None:
    with pytest.raises(FrameExtractionError):
        select_frame_indices(10, 5.0, frame_interval=2, target_frames=3)


def test_frame_extraction_count_names_and_timestamps(tmp_path: Path) -> None:
    video_path = _create_test_video(tmp_path / "input.mp4", frame_count=12, fps=6.0)

    metadata, frames = extract_frames(video_path, tmp_path / "frames", frame_interval=3)

    assert metadata.total_frames == 12
    assert len(frames) == 4
    assert [frame.filename for frame in frames] == [
        "frame_000001.jpg",
        "frame_000002.jpg",
        "frame_000003.jpg",
        "frame_000004.jpg",
    ]
    assert [frame.source_frame_index for frame in frames] == [0, 3, 6, 9]
    assert [frame.timestamp_seconds for frame in frames] == pytest.approx([0.0, 0.5, 1.0, 1.5])
    assert all(Path(frame.path).exists() for frame in frames)
