from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.frame_quality import (
    FrameSelectionConfig,
    QualityThresholds,
    analyze_and_select_frames,
    analyze_frame,
    brightness_statistics,
    create_feature_detector,
    variance_of_laplacian,
)


def _thresholds(*, blur_threshold: float = 0.0) -> QualityThresholds:
    return QualityThresholds(
        blur_threshold=blur_threshold,
        min_brightness=30.0,
        max_brightness=225.0,
        dark_intensity_cutoff=30,
        bright_intensity_cutoff=225,
        max_dark_pixel_ratio=0.45,
        max_bright_pixel_ratio=0.45,
        minimum_feature_count=5,
    )


def _selection_config() -> FrameSelectionConfig:
    return FrameSelectionConfig(
        method="automatic",
        feature_detector="orb",
        maximum_features=1000,
        redundancy_threshold=0.2,
        minimum_matches=5,
        lowe_ratio=0.75,
        use_geometric_check=True,
        ransac_reprojection_threshold=3.0,
        max_redundant_motion=0.01,
    )


def _feature_rich_image() -> np.ndarray:
    rng = np.random.default_rng(7)
    image = np.full((180, 240, 3), 128, dtype=np.uint8)
    for _ in range(40):
        center = (int(rng.integers(10, 230)), int(rng.integers(10, 170)))
        color = tuple(int(value) for value in rng.integers(40, 230, size=3))
        cv2.circle(image, center, int(rng.integers(3, 9)), color, -1)
    for x in range(20, 220, 40):
        cv2.line(image, (x, 10), (240 - x // 2, 170), (20, 240, 120), 2)
    cv2.putText(image, "3DGS", (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return image


def _write(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), image)
    return path


def test_sharp_image_has_higher_laplacian_variance_than_blurred() -> None:
    sharp = _feature_rich_image()
    blurred = cv2.GaussianBlur(sharp, (21, 21), 0)

    assert variance_of_laplacian(sharp) > variance_of_laplacian(blurred)


def test_dark_bright_and_normal_exposure_statistics() -> None:
    dark = np.full((20, 20, 3), 10, dtype=np.uint8)
    bright = np.full((20, 20, 3), 245, dtype=np.uint8)
    normal = np.full((20, 20, 3), 128, dtype=np.uint8)

    assert brightness_statistics(dark, dark_intensity_cutoff=30, bright_intensity_cutoff=225)[1] == 1.0
    assert brightness_statistics(bright, dark_intensity_cutoff=30, bright_intensity_cutoff=225)[2] == 1.0
    mean, dark_ratio, bright_ratio = brightness_statistics(
        normal,
        dark_intensity_cutoff=30,
        bright_intensity_cutoff=225,
    )
    assert mean == 128.0
    assert dark_ratio == 0.0
    assert bright_ratio == 0.0


def test_quality_reasons_for_exposure_and_invalid_images(tmp_path: Path) -> None:
    detector = create_feature_detector("orb", 500)
    thresholds = _thresholds(blur_threshold=0.0)

    dark_result = analyze_frame(
        _write(tmp_path / "dark.jpg", np.full((30, 30, 3), 10, dtype=np.uint8)),
        thresholds=thresholds,
        detector=detector,
        frame_id=1,
    ).result
    bright_result = analyze_frame(
        _write(tmp_path / "bright.jpg", np.full((30, 30, 3), 245, dtype=np.uint8)),
        thresholds=thresholds,
        detector=detector,
        frame_id=2,
    ).result
    normal_result = analyze_frame(
        _write(tmp_path / "normal.jpg", _feature_rich_image()),
        thresholds=thresholds,
        detector=detector,
        frame_id=3,
    ).result
    invalid_path = tmp_path / "invalid.jpg"
    invalid_path.write_bytes(b"not an image")
    invalid_result = analyze_frame(
        invalid_path,
        thresholds=thresholds,
        detector=detector,
        frame_id=4,
    ).result

    assert dark_result.selection_reason == "underexposed"
    assert bright_result.selection_reason == "overexposed"
    assert normal_result.selection_reason == "selected"
    assert invalid_result.selection_reason == "invalid_image"


def test_identical_frames_are_marked_redundant(tmp_path: Path) -> None:
    image = _feature_rich_image()
    first = _write(tmp_path / "frame_000001.jpg", image)
    second = _write(tmp_path / "frame_000002.jpg", image)

    analyzed, selected = analyze_and_select_frames(
        [first, second],
        thresholds=_thresholds(blur_threshold=0.0),
        selection_config=_selection_config(),
    )

    assert len(selected) == 1
    assert analyzed[1].result.selection_reason == "redundant"
    assert analyzed[1].result.match_count is not None


def test_moderately_changed_frame_can_remain_selectable(tmp_path: Path) -> None:
    first_image = _feature_rich_image()
    transform = np.float32([[1, 0, 24], [0, 1, 0]])
    shifted = cv2.warpAffine(first_image, transform, (first_image.shape[1], first_image.shape[0]))
    first = _write(tmp_path / "frame_000001.jpg", first_image)
    second = _write(tmp_path / "frame_000002.jpg", shifted)

    analyzed, selected = analyze_and_select_frames(
        [first, second],
        thresholds=_thresholds(blur_threshold=0.0),
        selection_config=_selection_config(),
    )

    assert len(selected) == 2
    assert analyzed[1].result.selection_reason == "selected"
