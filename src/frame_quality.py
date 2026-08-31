"""Frame quality analysis and feature-based redundancy selection."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:  # Allows: python src/frame_quality.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover - depends on local environment
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - numpy is a required dependency
    np = None
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

from src.extract_frames import ExtractedFrame
from src.utils import discover_images, ensure_dir


class FrameQualityError(RuntimeError):
    """Raised when frame quality analysis cannot be completed."""


@dataclass(frozen=True)
class QualityThresholds:
    """Configurable quality thresholds for candidate frame rejection."""

    blur_threshold: float
    min_brightness: float
    max_brightness: float
    dark_intensity_cutoff: int
    bright_intensity_cutoff: int
    max_dark_pixel_ratio: float
    max_bright_pixel_ratio: float
    minimum_feature_count: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "QualityThresholds":
        return cls(
            blur_threshold=float(config["blur_threshold"]),
            min_brightness=float(config["min_brightness"]),
            max_brightness=float(config["max_brightness"]),
            dark_intensity_cutoff=int(config["dark_intensity_cutoff"]),
            bright_intensity_cutoff=int(config["bright_intensity_cutoff"]),
            max_dark_pixel_ratio=float(config["max_dark_pixel_ratio"]),
            max_bright_pixel_ratio=float(config["max_bright_pixel_ratio"]),
            minimum_feature_count=int(config["minimum_feature_count"]),
        )


@dataclass(frozen=True)
class FrameSelectionConfig:
    """Feature matching settings for redundancy removal."""

    method: str
    feature_detector: str
    maximum_features: int
    redundancy_threshold: float
    minimum_matches: int
    lowe_ratio: float
    use_geometric_check: bool
    ransac_reprojection_threshold: float
    max_redundant_motion: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FrameSelectionConfig":
        return cls(
            method=str(config["method"]).lower(),
            feature_detector=str(config["feature_detector"]).lower(),
            maximum_features=int(config["maximum_features"]),
            redundancy_threshold=float(config["redundancy_threshold"]),
            minimum_matches=int(config["minimum_matches"]),
            lowe_ratio=float(config["lowe_ratio"]),
            use_geometric_check=bool(config["use_geometric_check"]),
            ransac_reprojection_threshold=float(config["ransac_reprojection_threshold"]),
            max_redundant_motion=float(config["max_redundant_motion"]),
        )


@dataclass
class FrameQualityResult:
    """CSV-ready quality and selection result for one candidate frame."""

    frame_id: int
    source_frame_index: int | None
    filename: str
    timestamp_seconds: float | None
    blur_score: float | None
    mean_brightness: float | None
    dark_pixel_ratio: float | None
    bright_pixel_ratio: float | None
    feature_count: int | None
    match_count: int | None = None
    similarity_score: float | None = None
    selected: bool = False
    selection_reason: str = "invalid_image"

    def to_csv_row(self) -> dict[str, str | int | float | bool]:
        row = asdict(self)
        for key, value in list(row.items()):
            if value is None:
                row[key] = ""
        return row


@dataclass
class FeatureComparison:
    """Result of comparing one candidate against the latest selected frame."""

    match_count: int
    similarity_score: float | None
    median_motion_ratio: float | None
    geometric_inlier_count: int | None
    redundant: bool


@dataclass
class AnalyzedFrame:
    """Internal frame analysis state used by the selector."""

    path: Path
    result: FrameQualityResult
    keypoints: tuple[Any, ...]
    descriptors: Any | None
    image_shape: tuple[int, int] | None


CSV_COLUMNS = [
    "frame_id",
    "source_frame_index",
    "filename",
    "timestamp_seconds",
    "blur_score",
    "mean_brightness",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
    "feature_count",
    "match_count",
    "similarity_score",
    "selected",
    "selection_reason",
]


def _require_dependencies() -> None:
    if cv2 is None:
        raise FrameQualityError("OpenCV is required for frame quality analysis.") from _CV2_IMPORT_ERROR
    if np is None:
        raise FrameQualityError("NumPy is required for frame quality analysis.") from _NUMPY_IMPORT_ERROR


def create_feature_detector(feature_detector: str = "orb", maximum_features: int = 3000):
    """Create an OpenCV feature detector, using ORB by default."""

    _require_dependencies()
    detector_name = feature_detector.lower()
    if detector_name == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise FrameQualityError("SIFT is not available in this OpenCV build; use ORB instead.")
        return cv2.SIFT_create(nfeatures=maximum_features)
    if detector_name != "orb":
        raise FrameQualityError(f"Unsupported feature detector: {feature_detector}")
    return cv2.ORB_create(nfeatures=maximum_features)


def read_image_bgr(path: str | Path):
    """Read an image with OpenCV, returning None when unreadable."""

    _require_dependencies()
    image_path = Path(path)
    if not image_path.exists() or not image_path.is_file():
        return None
    return cv2.imread(str(image_path), cv2.IMREAD_COLOR)


def variance_of_laplacian(image_bgr) -> float:
    """Measure blur with the variance of the grayscale Laplacian."""

    _require_dependencies()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_statistics(
    image_bgr,
    *,
    dark_intensity_cutoff: int,
    bright_intensity_cutoff: int,
) -> tuple[float, float, float]:
    """Return mean brightness, dark-pixel ratio, and bright-pixel ratio."""

    _require_dependencies()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    total = gray.size
    if total == 0:
        raise FrameQualityError("Image contains no pixels.")
    mean_brightness = float(gray.mean())
    dark_ratio = float((gray <= dark_intensity_cutoff).sum() / total)
    bright_ratio = float((gray >= bright_intensity_cutoff).sum() / total)
    return mean_brightness, dark_ratio, bright_ratio


def detect_features(image_bgr, detector) -> tuple[tuple[Any, ...], Any | None]:
    """Detect OpenCV keypoints and descriptors for a BGR image."""

    _require_dependencies()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return tuple(keypoints or ()), descriptors


def _quality_reason(
    *,
    blur_score: float,
    mean_brightness: float,
    dark_pixel_ratio: float,
    bright_pixel_ratio: float,
    thresholds: QualityThresholds,
) -> str:
    if blur_score < thresholds.blur_threshold:
        return "blur"
    if (
        mean_brightness < thresholds.min_brightness
        or dark_pixel_ratio > thresholds.max_dark_pixel_ratio
    ):
        return "underexposed"
    if (
        mean_brightness > thresholds.max_brightness
        or bright_pixel_ratio > thresholds.max_bright_pixel_ratio
    ):
        return "overexposed"
    return "selected"


def analyze_frame(
    path: str | Path,
    *,
    thresholds: QualityThresholds,
    detector,
    frame_id: int,
    source_frame_index: int | None = None,
    timestamp_seconds: float | None = None,
) -> AnalyzedFrame:
    """Analyze one candidate frame for quality and feature count."""

    image_path = Path(path)
    image = read_image_bgr(image_path)
    if image is None:
        return AnalyzedFrame(
            path=image_path,
            result=FrameQualityResult(
                frame_id=frame_id,
                source_frame_index=source_frame_index,
                filename=image_path.name,
                timestamp_seconds=timestamp_seconds,
                blur_score=None,
                mean_brightness=None,
                dark_pixel_ratio=None,
                bright_pixel_ratio=None,
                feature_count=None,
                selected=False,
                selection_reason="invalid_image",
            ),
            keypoints=(),
            descriptors=None,
            image_shape=None,
        )

    blur_score = variance_of_laplacian(image)
    mean_brightness, dark_ratio, bright_ratio = brightness_statistics(
        image,
        dark_intensity_cutoff=thresholds.dark_intensity_cutoff,
        bright_intensity_cutoff=thresholds.bright_intensity_cutoff,
    )
    keypoints, descriptors = detect_features(image, detector)
    reason = _quality_reason(
        blur_score=blur_score,
        mean_brightness=mean_brightness,
        dark_pixel_ratio=dark_ratio,
        bright_pixel_ratio=bright_ratio,
        thresholds=thresholds,
    )
    height, width = image.shape[:2]

    return AnalyzedFrame(
        path=image_path,
        result=FrameQualityResult(
            frame_id=frame_id,
            source_frame_index=source_frame_index,
            filename=image_path.name,
            timestamp_seconds=timestamp_seconds,
            blur_score=blur_score,
            mean_brightness=mean_brightness,
            dark_pixel_ratio=dark_ratio,
            bright_pixel_ratio=bright_ratio,
            feature_count=len(keypoints),
            selected=reason == "selected",
            selection_reason=reason,
        ),
        keypoints=keypoints,
        descriptors=descriptors,
        image_shape=(height, width),
    )


def _norm_for_detector(feature_detector: str) -> int:
    return cv2.NORM_L2 if feature_detector.lower() == "sift" else cv2.NORM_HAMMING


def _match_descriptors(descriptors_a, descriptors_b, config: FrameSelectionConfig) -> list[Any]:
    if descriptors_a is None or descriptors_b is None:
        return []
    if len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return []

    matcher = cv2.BFMatcher(_norm_for_detector(config.feature_detector), crossCheck=False)
    knn_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    good_matches = []
    for pair in knn_matches:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance < config.lowe_ratio * second.distance:
            good_matches.append(first)

    if config.feature_detector == "orb" and len(good_matches) < config.minimum_matches:
        cross_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        cross_matches = cross_matcher.match(descriptors_a, descriptors_b)
        cross_matches = sorted(cross_matches, key=lambda match: match.distance)
        distance_filtered = [match for match in cross_matches if match.distance <= 32]
        if len(distance_filtered) > len(good_matches):
            return distance_filtered

    return sorted(good_matches, key=lambda match: match.distance)


def _match_motion(
    reference: AnalyzedFrame,
    candidate: AnalyzedFrame,
    matches: Sequence[Any],
) -> float | None:
    if not matches or reference.image_shape is None or candidate.image_shape is None:
        return None

    distances: list[float] = []
    for match in matches:
        ref_point = reference.keypoints[match.queryIdx].pt
        cand_point = candidate.keypoints[match.trainIdx].pt
        distances.append(math.dist(ref_point, cand_point))

    diagonal = math.hypot(candidate.image_shape[1], candidate.image_shape[0])
    if diagonal <= 0:
        return None
    return float(np.median(np.asarray(distances)) / diagonal)


def _geometric_inlier_count(
    reference: AnalyzedFrame,
    candidate: AnalyzedFrame,
    matches: Sequence[Any],
    config: FrameSelectionConfig,
) -> int | None:
    if not config.use_geometric_check or len(matches) < 8:
        return None

    points_a = np.float32([reference.keypoints[m.queryIdx].pt for m in matches])
    points_b = np.float32([candidate.keypoints[m.trainIdx].pt for m in matches])
    try:
        _, mask = cv2.findFundamentalMat(
            points_a,
            points_b,
            cv2.FM_RANSAC,
            config.ransac_reprojection_threshold,
            0.99,
        )
    except cv2.error:
        mask = None

    if mask is None:
        try:
            _, mask = cv2.findHomography(
                points_a,
                points_b,
                cv2.RANSAC,
                config.ransac_reprojection_threshold,
            )
        except cv2.error:
            mask = None
    if mask is None:
        return None
    return int(mask.ravel().astype(bool).sum())


def compare_frame_features(
    reference: AnalyzedFrame,
    candidate: AnalyzedFrame,
    config: FrameSelectionConfig,
    *,
    minimum_feature_count: int,
) -> FeatureComparison:
    """Compare candidate to the latest selected frame and decide redundancy.

    The selector treats high overlap plus very small median keypoint motion as
    a duplicate. Moderate keypoint motion is retained because SfM needs overlap,
    not only large baseline jumps.
    """

    _require_dependencies()
    ref_features = reference.result.feature_count or 0
    cand_features = candidate.result.feature_count or 0
    if ref_features < minimum_feature_count or cand_features < minimum_feature_count:
        return FeatureComparison(
            match_count=0,
            similarity_score=None,
            median_motion_ratio=None,
            geometric_inlier_count=None,
            redundant=False,
        )

    matches = _match_descriptors(reference.descriptors, candidate.descriptors, config)
    match_count = len(matches)
    if match_count == 0:
        return FeatureComparison(
            match_count=0,
            similarity_score=0.0,
            median_motion_ratio=None,
            geometric_inlier_count=None,
            redundant=False,
        )

    inlier_count = _geometric_inlier_count(reference, candidate, matches, config)
    reliable_count = inlier_count if inlier_count is not None else match_count
    denominator = max(1, min(ref_features, cand_features))
    match_overlap = min(1.0, reliable_count / denominator)
    motion_ratio = _match_motion(reference, candidate, matches)
    if motion_ratio is None:
        motion_penalty = 1.0
    else:
        motion_penalty = max(0.0, 1.0 - min(1.0, motion_ratio / max(config.max_redundant_motion, 1e-9)))
    similarity_score = float(match_overlap * motion_penalty)
    redundant = (
        match_count >= config.minimum_matches
        and similarity_score >= config.redundancy_threshold
        and (motion_ratio is None or motion_ratio <= config.max_redundant_motion)
    )

    return FeatureComparison(
        match_count=match_count,
        similarity_score=similarity_score,
        median_motion_ratio=motion_ratio,
        geometric_inlier_count=inlier_count,
        redundant=redundant,
    )


def analyze_and_select_frames(
    frames: Sequence[ExtractedFrame | Path | str],
    *,
    thresholds: QualityThresholds,
    selection_config: FrameSelectionConfig,
    logger: logging.Logger | None = None,
) -> tuple[list[AnalyzedFrame], list[AnalyzedFrame]]:
    """Analyze candidates, reject low-quality frames, then remove duplicates."""

    logger = logger or logging.getLogger("smartphone_3dgs.frame_quality")
    detector = create_feature_detector(
        selection_config.feature_detector,
        selection_config.maximum_features,
    )
    analyzed: list[AnalyzedFrame] = []
    selected: list[AnalyzedFrame] = []
    latest_selected: AnalyzedFrame | None = None

    logger.info(
        "Quality thresholds: blur>=%.3f brightness=[%.1f, %.1f] dark_ratio<=%.3f bright_ratio<=%.3f",
        thresholds.blur_threshold,
        thresholds.min_brightness,
        thresholds.max_brightness,
        thresholds.max_dark_pixel_ratio,
        thresholds.max_bright_pixel_ratio,
    )
    logger.info(
        "Frame selector: detector=%s max_features=%d redundancy_threshold=%.3f minimum_matches=%d",
        selection_config.feature_detector,
        selection_config.maximum_features,
        selection_config.redundancy_threshold,
        selection_config.minimum_matches,
    )

    for index, frame in enumerate(frames, start=1):
        if isinstance(frame, ExtractedFrame):
            path = Path(frame.path)
            frame_id = frame.frame_id
            source_frame_index = frame.source_frame_index
            timestamp_seconds = frame.timestamp_seconds
        else:
            path = Path(frame)
            frame_id = index
            source_frame_index = None
            timestamp_seconds = None

        analysis = analyze_frame(
            path,
            thresholds=thresholds,
            detector=detector,
            frame_id=frame_id,
            source_frame_index=source_frame_index,
            timestamp_seconds=timestamp_seconds,
        )

        if analysis.result.selected and latest_selected is not None:
            comparison = compare_frame_features(
                latest_selected,
                analysis,
                selection_config,
                minimum_feature_count=thresholds.minimum_feature_count,
            )
            analysis.result.match_count = comparison.match_count
            analysis.result.similarity_score = comparison.similarity_score
            if comparison.redundant:
                analysis.result.selected = False
                analysis.result.selection_reason = "redundant"

        if analysis.result.selected:
            latest_selected = analysis
            selected.append(analysis)

        analyzed.append(analysis)
        logger.debug("Frame selection result: %s", asdict(analysis.result))

    logger.info(
        "Frame selection complete: candidates=%d selected=%d rejected=%d",
        len(analyzed),
        len(selected),
        len(analyzed) - len(selected),
    )
    return analyzed, selected


def write_frame_selection_csv(path: str | Path, results: Sequence[FrameQualityResult]) -> None:
    """Write frame quality and selection results to CSV."""

    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_csv_row())


def summarize_selection(results: Sequence[FrameQualityResult]) -> dict[str, int]:
    """Return selection counts grouped for the Phase 2 summary."""

    return {
        "candidate_frames": len(results),
        "blur_rejected": sum(result.selection_reason == "blur" for result in results),
        "underexposed_rejected": sum(result.selection_reason == "underexposed" for result in results),
        "overexposed_rejected": sum(result.selection_reason == "overexposed" for result in results),
        "redundant_rejected": sum(result.selection_reason == "redundant" for result in results),
        "invalid_rejected": sum(result.selection_reason == "invalid_image" for result in results),
        "retained_frames": sum(result.selected for result in results),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze candidate frame quality and redundancy.")
    parser.add_argument("--input-dir", required=True, help="Directory containing candidate frames.")
    parser.add_argument("--output-csv", required=True, help="Path to write frame_selection.csv.")
    parser.add_argument("--blur-threshold", type=float, default=100.0)
    parser.add_argument("--min-brightness", type=float, default=30.0)
    parser.add_argument("--max-brightness", type=float, default=225.0)
    parser.add_argument("--dark-intensity-cutoff", type=int, default=30)
    parser.add_argument("--bright-intensity-cutoff", type=int, default=225)
    parser.add_argument("--max-dark-pixel-ratio", type=float, default=0.45)
    parser.add_argument("--max-bright-pixel-ratio", type=float, default=0.45)
    parser.add_argument("--minimum-feature-count", type=int, default=40)
    parser.add_argument("--feature-detector", choices=["orb", "sift"], default="orb")
    parser.add_argument("--maximum-features", type=int, default=3000)
    parser.add_argument("--redundancy-threshold", type=float, default=0.85)
    parser.add_argument("--minimum-matches", type=int, default=50)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("smartphone_3dgs.frame_quality")
    thresholds = QualityThresholds(
        blur_threshold=args.blur_threshold,
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
        dark_intensity_cutoff=args.dark_intensity_cutoff,
        bright_intensity_cutoff=args.bright_intensity_cutoff,
        max_dark_pixel_ratio=args.max_dark_pixel_ratio,
        max_bright_pixel_ratio=args.max_bright_pixel_ratio,
        minimum_feature_count=args.minimum_feature_count,
    )
    selection_config = FrameSelectionConfig(
        method="automatic",
        feature_detector=args.feature_detector,
        maximum_features=args.maximum_features,
        redundancy_threshold=args.redundancy_threshold,
        minimum_matches=args.minimum_matches,
        lowe_ratio=0.75,
        use_geometric_check=True,
        ransac_reprojection_threshold=3.0,
        max_redundant_motion=0.015,
    )
    frames = discover_images(args.input_dir)
    try:
        analyzed, _ = analyze_and_select_frames(
            frames,
            thresholds=thresholds,
            selection_config=selection_config,
            logger=logger,
        )
    except FrameQualityError as exc:
        logger.error("%s", exc)
        return 2

    results = [item.result for item in analyzed]
    write_frame_selection_csv(args.output_csv, results)
    print(json.dumps(summarize_selection(results), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
