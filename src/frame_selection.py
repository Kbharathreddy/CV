"""Reusable Phase 5 train/test split and frame-selection utilities.

The split is explicit and manifest-driven so every 3DGS experiment evaluates
against the same held-out image names. Fixed selectors use uniform coverage of
the ordered camera trajectory. The automatic selector is deterministic and
explainable: it combines image quality/feature signals with greedy camera-pose
coverage when those signals are available.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from src.convert_colmap import ImagePose, read_colmap_model, qvec_to_rotation_matrix
from src.utils import write_json


class Phase5SelectionError(ValueError):
    """Raised when a Phase 5 split or selector is invalid."""


@dataclass(frozen=True)
class ImageQualitySignals:
    """Optional quality and feature-richness measurements for one image."""

    valid: bool = True
    blur_score: float | None = None
    mean_brightness: float | None = None
    dark_pixel_ratio: float | None = None
    bright_pixel_ratio: float | None = None
    feature_count: int | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class PoseSignals:
    """Camera-pose features used by the automatic selector."""

    camera_center: tuple[float, float, float] | None = None
    viewing_direction: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class FrameCandidate:
    """Candidate image and all deterministic selection signals."""

    name: str
    order_index: int
    pose: PoseSignals = field(default_factory=PoseSignals)
    quality: ImageQualitySignals = field(default_factory=ImageQualitySignals)


@dataclass(frozen=True)
class SplitManifest:
    """Common Phase 5 held-out split shared by all experiments."""

    manifest_version: int
    dataset: str
    scene: str
    total_scene_images: int
    ordered_images: list[str]
    training_candidate_images: list[str]
    test_images: list[str]
    split_algorithm: str
    holdout_interval: int
    holdout_offset: int
    seed: int
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentManifest:
    """Small committed manifest for a Phase 5 experiment."""

    manifest_version: int
    experiment_name: str
    selection_method: str
    training_budget: int | None
    train_image_count: int
    test_image_count: int
    total_scene_images: int
    training_images: list[str]
    test_images: list[str]
    seed: int
    configuration: dict[str, Any]
    selection_details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


AUTOMATIC_SELECTION_WEIGHTS = {
    "pose_diversity": 0.65,
    "image_quality": 0.25,
    "feature_richness": 0.10,
}


def _ensure_unique_image_names(image_names: Sequence[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for name in image_names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise Phase5SelectionError(f"Duplicate image names are not allowed: {sorted(duplicates)[:5]}")


def validate_image_names(image_names: Sequence[str]) -> list[str]:
    """Return image names as a list after basic reproducibility checks."""

    names = [str(name) for name in image_names]
    if not names:
        raise Phase5SelectionError("At least one image name is required.")
    if any(Path(name).is_absolute() for name in names):
        raise Phase5SelectionError("Committed manifests must use relative image names.")
    _ensure_unique_image_names(names)
    return names


def ordered_image_names_from_model(model_path: str | Path) -> list[str]:
    """Read a COLMAP sparse model and return image names sorted by filename."""

    model = read_colmap_model(model_path)
    return sorted(image.name for image in model.images.values())


def create_heldout_split(
    ordered_images: Sequence[str],
    *,
    dataset: str = "mipnerf360",
    scene: str = "bonsai",
    holdout_interval: int = 8,
    holdout_offset: int = 0,
    seed: int = 42,
) -> SplitManifest:
    """Create the common explicit held-out split.

    The default mirrors the common LLFF-style every-8th policy, but the output
    is stored as concrete filenames and reused across all subset experiments.
    """

    names = validate_image_names(ordered_images)
    if holdout_interval <= 1:
        raise Phase5SelectionError("holdout_interval must be greater than 1.")
    if not 0 <= holdout_offset < holdout_interval:
        raise Phase5SelectionError("holdout_offset must be in [0, holdout_interval).")

    test_images = [
        name for index, name in enumerate(names) if index % holdout_interval == holdout_offset
    ]
    training_candidates = [name for name in names if name not in set(test_images)]
    if not test_images:
        raise Phase5SelectionError("Held-out split produced zero test images.")
    if not training_candidates:
        raise Phase5SelectionError("Held-out split produced zero training candidates.")

    return SplitManifest(
        manifest_version=1,
        dataset=dataset,
        scene=scene,
        total_scene_images=len(names),
        ordered_images=list(names),
        training_candidate_images=training_candidates,
        test_images=test_images,
        split_algorithm="ordered_every_nth",
        holdout_interval=int(holdout_interval),
        holdout_offset=int(holdout_offset),
        seed=int(seed),
        notes=(
            "Held-out filenames are explicit and must be reused unchanged for "
            "all Phase 5 3DGS experiments."
        ),
    )


def assert_disjoint_train_test(train_images: Sequence[str], test_images: Sequence[str]) -> None:
    """Raise if an experiment would train on held-out views."""

    overlap = sorted(set(train_images) & set(test_images))
    if overlap:
        raise Phase5SelectionError(f"Train/test overlap is not allowed: {overlap[:5]}")


def uniform_sample(image_names: Sequence[str], budget: int) -> list[str]:
    """Sample a deterministic, trajectory-spanning subset from ordered names."""

    names = validate_image_names(image_names)
    if budget <= 0:
        raise Phase5SelectionError("training budget must be positive.")
    if budget > len(names):
        raise Phase5SelectionError(
            f"Requested {budget} images but only {len(names)} candidates are available."
        )
    if budget == len(names):
        return list(names)
    if budget == 1:
        return [names[len(names) // 2]]

    denominator = budget - 1
    indices = [
        (index * (len(names) - 1) + denominator // 2) // denominator
        for index in range(budget)
    ]
    selected = [names[index] for index in indices]
    if len(set(selected)) != budget:
        raise Phase5SelectionError("Uniform sampling produced duplicate images.")
    return selected


def fixed_budget_selection(split: SplitManifest, budget: int) -> list[str]:
    """Select fixed-budget training images from the common training candidates."""

    selected = uniform_sample(split.training_candidate_images, budget)
    assert_disjoint_train_test(selected, split.test_images)
    return selected


def viewing_direction_from_qvec(qvec: Iterable[float]) -> tuple[float, float, float]:
    """Return the COLMAP camera optical axis in world coordinates."""

    rotation = qvec_to_rotation_matrix(qvec)
    direction = [rotation[2][0], rotation[2][1], rotation[2][2]]
    norm = math.sqrt(sum(value * value for value in direction))
    if norm <= 0:
        return (0.0, 0.0, 1.0)
    return tuple(float(value / norm) for value in direction)


def _quality_score(signals: ImageQualitySignals) -> float:
    if not signals.valid:
        return 0.0

    blur = 1.0 if signals.blur_score is None else min(1.0, max(0.0, signals.blur_score / 1000.0))
    brightness = 1.0
    if signals.mean_brightness is not None:
        brightness = max(0.0, 1.0 - abs(signals.mean_brightness - 128.0) / 128.0)
    dark = signals.dark_pixel_ratio or 0.0
    bright = signals.bright_pixel_ratio or 0.0
    exposure = max(0.0, 1.0 - max(dark, bright))
    return float(max(0.0, min(1.0, 0.4 * blur + 0.4 * brightness + 0.2 * exposure)))


def _feature_score(signals: ImageQualitySignals) -> float:
    if signals.feature_count is None:
        return 0.5 if signals.valid else 0.0
    return float(max(0.0, min(1.0, signals.feature_count / 1500.0)))


def _center_scale(candidates: Sequence[FrameCandidate]) -> float:
    centers = [candidate.pose.camera_center for candidate in candidates if candidate.pose.camera_center]
    if len(centers) < 2:
        return 1.0
    mins = [min(center[axis] for center in centers) for axis in range(3)]
    maxs = [max(center[axis] for center in centers) for axis in range(3)]
    diagonal = math.sqrt(sum((maxs[axis] - mins[axis]) ** 2 for axis in range(3)))
    return diagonal if diagonal > 0 else 1.0


def _normalized_pose_distance(
    a: FrameCandidate,
    b: FrameCandidate,
    *,
    center_scale: float,
    sequence_scale: int,
) -> float:
    center_distance = 0.0
    if a.pose.camera_center and b.pose.camera_center:
        center_distance = math.dist(a.pose.camera_center, b.pose.camera_center) / center_scale
        center_distance = min(1.0, center_distance)

    direction_distance = 0.0
    if a.pose.viewing_direction and b.pose.viewing_direction:
        dot = sum(x * y for x, y in zip(a.pose.viewing_direction, b.pose.viewing_direction))
        dot = max(-1.0, min(1.0, dot))
        direction_distance = math.acos(dot) / math.pi

    sequence_distance = abs(a.order_index - b.order_index) / max(1, sequence_scale)
    return float(0.55 * center_distance + 0.30 * direction_distance + 0.15 * sequence_distance)


def build_pose_candidates(
    image_names: Sequence[str],
    *,
    model_path: str | Path,
    quality_by_name: dict[str, ImageQualitySignals] | None = None,
) -> list[FrameCandidate]:
    """Create automatic-selection candidates from COLMAP poses and signals."""

    names = validate_image_names(image_names)
    model = read_colmap_model(model_path)
    pose_by_name: dict[str, ImagePose] = {image.name: image for image in model.images.values()}
    missing = [name for name in names if name not in pose_by_name]
    if missing:
        raise Phase5SelectionError(f"Images missing from COLMAP model: {missing[:5]}")

    candidates = []
    for index, name in enumerate(names):
        pose = pose_by_name[name]
        fallback_features = pose.num_observed_points if pose.num_observed_points > 0 else None
        quality = (quality_by_name or {}).get(
            name,
            ImageQualitySignals(valid=True, feature_count=fallback_features),
        )
        if quality.feature_count is None and fallback_features is not None:
            quality = ImageQualitySignals(
                valid=quality.valid,
                blur_score=quality.blur_score,
                mean_brightness=quality.mean_brightness,
                dark_pixel_ratio=quality.dark_pixel_ratio,
                bright_pixel_ratio=quality.bright_pixel_ratio,
                feature_count=fallback_features,
                rejection_reason=quality.rejection_reason,
            )
        candidates.append(
            FrameCandidate(
                name=name,
                order_index=index,
                pose=PoseSignals(
                    camera_center=tuple(float(value) for value in pose.camera_center),
                    viewing_direction=viewing_direction_from_qvec(pose.qvec),
                ),
                quality=quality,
            )
        )
    return candidates


def automatic_budget_selection(
    candidates: Sequence[FrameCandidate],
    *,
    budget: int,
    min_quality_score: float = 0.35,
    weights: dict[str, float] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Select frames with deterministic quality-aware pose coverage.

    The algorithm filters clearly invalid candidates when enough remain, anchors
    the first and last usable views, then greedily adds the candidate that best
    balances distance from already selected camera poses with quality and
    feature-richness. Returned filenames are sorted back into scene order.
    """

    if budget <= 0:
        raise Phase5SelectionError("automatic budget must be positive.")
    if not candidates:
        raise Phase5SelectionError("automatic selection requires candidates.")
    _ensure_unique_image_names([candidate.name for candidate in candidates])

    normalized_weights = dict(AUTOMATIC_SELECTION_WEIGHTS)
    if weights:
        normalized_weights.update(weights)
    total_weight = sum(normalized_weights.values())
    if total_weight <= 0:
        raise Phase5SelectionError("automatic selection weights must be positive.")
    normalized_weights = {key: value / total_weight for key, value in normalized_weights.items()}

    quality_scores = {candidate.name: _quality_score(candidate.quality) for candidate in candidates}
    usable = [
        candidate
        for candidate in candidates
        if candidate.quality.valid and quality_scores[candidate.name] >= min_quality_score
    ]
    quality_filter_used = True
    if len(usable) < budget:
        usable = [candidate for candidate in candidates if candidate.quality.valid]
        quality_filter_used = False
    if len(usable) < budget:
        raise Phase5SelectionError(
            f"Only {len(usable)} valid automatic candidates are available for budget {budget}."
        )

    ordered = sorted(usable, key=lambda candidate: candidate.order_index)
    if budget == len(ordered):
        selected = ordered
    elif budget == 1:
        selected = [max(ordered, key=lambda candidate: (quality_scores[candidate.name], -candidate.order_index))]
    else:
        selected = [ordered[0], ordered[-1]]
        selected_names = {ordered[0].name, ordered[-1].name}
        center_scale = _center_scale(ordered)
        sequence_scale = max(candidate.order_index for candidate in candidates)
        while len(selected) < budget:
            best_candidate: FrameCandidate | None = None
            best_score = -1.0
            for candidate in ordered:
                if candidate.name in selected_names:
                    continue
                diversity = min(
                    _normalized_pose_distance(
                        candidate,
                        existing,
                        center_scale=center_scale,
                        sequence_scale=sequence_scale,
                    )
                    for existing in selected
                )
                score = (
                    normalized_weights["pose_diversity"] * diversity
                    + normalized_weights["image_quality"] * quality_scores[candidate.name]
                    + normalized_weights["feature_richness"] * _feature_score(candidate.quality)
                )
                if score > best_score or (
                    math.isclose(score, best_score) and best_candidate is not None
                    and candidate.order_index < best_candidate.order_index
                ):
                    best_score = score
                    best_candidate = candidate
            if best_candidate is None:
                raise Phase5SelectionError("Automatic selection could not choose the requested budget.")
            selected.append(best_candidate)
            selected_names.add(best_candidate.name)

    selected_ordered = sorted(selected, key=lambda candidate: candidate.order_index)
    diagnostics = {
        "algorithm": "quality_filtered_greedy_pose_coverage",
        "budget": int(budget),
        "candidate_count": len(candidates),
        "usable_candidate_count": len(usable),
        "quality_filter_used": quality_filter_used,
        "min_quality_score": float(min_quality_score),
        "weights": normalized_weights,
        "quality_signal_count": sum(candidate.quality.blur_score is not None for candidate in candidates),
        "pose_signal_count": sum(candidate.pose.camera_center is not None for candidate in candidates),
    }
    return [candidate.name for candidate in selected_ordered], diagnostics


def build_experiment_manifest(
    *,
    split: SplitManifest,
    experiment_name: str,
    selection_method: str,
    training_images: Sequence[str],
    training_budget: int | None,
    selection_details: dict[str, Any] | None = None,
    iterations: int = 30000,
) -> ExperimentManifest:
    """Create and validate an experiment manifest."""

    train = validate_image_names(training_images)
    assert_disjoint_train_test(train, split.test_images)
    if any(name not in set(split.training_candidate_images) for name in train):
        raise Phase5SelectionError("Training images must come from split.training_candidate_images.")
    if training_budget is not None and len(train) != training_budget:
        raise Phase5SelectionError(
            f"Manifest {experiment_name} expected {training_budget} training images, found {len(train)}."
        )

    return ExperimentManifest(
        manifest_version=1,
        experiment_name=experiment_name,
        selection_method=selection_method,
        training_budget=training_budget,
        train_image_count=len(train),
        test_image_count=len(split.test_images),
        total_scene_images=split.total_scene_images,
        training_images=list(train),
        test_images=list(split.test_images),
        seed=split.seed,
        configuration={
            "dataset": split.dataset,
            "scene": split.scene,
            "iterations": int(iterations),
            "split_manifest": "experiments/phase5/split_manifest.json",
            "graphdeco_eval_flag": False,
            "heldout_policy": (
                "Train on a train-only scene, then render a separate held-out "
                "scene from the trained checkpoint."
            ),
        },
        selection_details=selection_details or {},
    )


def validate_experiment_manifest(manifest: dict[str, Any]) -> None:
    """Validate a serialized Phase 5 experiment manifest."""

    required = {
        "experiment_name",
        "selection_method",
        "training_images",
        "test_images",
        "train_image_count",
        "test_image_count",
        "total_scene_images",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise Phase5SelectionError(f"Experiment manifest missing keys: {missing}")
    train = validate_image_names(manifest["training_images"])
    test = validate_image_names(manifest["test_images"])
    assert_disjoint_train_test(train, test)
    if int(manifest["train_image_count"]) != len(train):
        raise Phase5SelectionError("train_image_count does not match training_images.")
    if int(manifest["test_image_count"]) != len(test):
        raise Phase5SelectionError("test_image_count does not match test_images.")


def write_split_manifest(path: str | Path, split: SplitManifest) -> None:
    write_json(path, split.to_dict())


def write_experiment_manifest(path: str | Path, manifest: ExperimentManifest) -> None:
    validate_experiment_manifest(manifest.to_dict())
    write_json(path, manifest.to_dict())
