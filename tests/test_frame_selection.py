from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.frame_selection import (
    FrameCandidate,
    ImageQualitySignals,
    Phase5SelectionError,
    PoseSignals,
    automatic_budget_selection,
    build_experiment_manifest,
    create_heldout_split,
    fixed_budget_selection,
    validate_experiment_manifest,
    write_experiment_manifest,
)


def _names(count: int) -> list[str]:
    return [f"image_{index:03d}.jpg" for index in range(count)]


def _candidate(index: int, *, valid: bool = True) -> FrameCandidate:
    return FrameCandidate(
        name=f"image_{index:03d}.jpg",
        order_index=index,
        pose=PoseSignals(
            camera_center=(float(index), float(index % 5), float(index % 3)),
            viewing_direction=(0.0, 0.0, 1.0),
        ),
        quality=ImageQualitySignals(
            valid=valid,
            blur_score=900.0,
            mean_brightness=128.0,
            dark_pixel_ratio=0.0,
            bright_pixel_ratio=0.0,
            feature_count=800 + index,
        ),
    )


def test_common_heldout_split_is_deterministic_and_disjoint() -> None:
    split = create_heldout_split(_names(292), holdout_interval=8, holdout_offset=0, seed=42)
    same_split = create_heldout_split(_names(292), holdout_interval=8, holdout_offset=0, seed=42)

    assert split.to_dict() == same_split.to_dict()
    assert len(split.test_images) == 37
    assert len(split.training_candidate_images) == 255
    assert set(split.training_candidate_images).isdisjoint(split.test_images)
    assert split.test_images[:3] == ["image_000.jpg", "image_008.jpg", "image_016.jpg"]


def test_fixed_budget_selection_samples_uniformly_not_first_n() -> None:
    split = create_heldout_split(_names(292), holdout_interval=8, holdout_offset=0)

    selected = fixed_budget_selection(split, 30)

    assert len(selected) == 30
    assert selected != split.training_candidate_images[:30]
    assert selected[0] == split.training_candidate_images[0]
    assert selected[-1] == split.training_candidate_images[-1]
    assert set(selected).isdisjoint(split.test_images)


def test_fixed_budget_rejects_more_images_than_available() -> None:
    split = create_heldout_split(_names(10), holdout_interval=5)

    with pytest.raises(Phase5SelectionError, match="only"):
        fixed_budget_selection(split, 20)


def test_automatic_selection_is_deterministic_and_respects_budget() -> None:
    candidates = [_candidate(index, valid=index not in {3, 7, 11}) for index in range(40)]

    selected, details = automatic_budget_selection(candidates, budget=12)
    selected_again, details_again = automatic_budget_selection(candidates, budget=12)

    assert selected == selected_again
    assert details == details_again
    assert len(selected) == 12
    assert "image_003.jpg" not in selected
    assert details["algorithm"] == "quality_filtered_greedy_pose_coverage"


def test_experiment_manifest_serialization(tmp_path: Path) -> None:
    split = create_heldout_split(_names(32), holdout_interval=8)
    train = fixed_budget_selection(split, 6)
    manifest = build_experiment_manifest(
        split=split,
        experiment_name="fixed_6",
        selection_method="fixed_uniform",
        training_images=train,
        training_budget=6,
        iterations=7000,
    )

    path = tmp_path / "fixed_6.json"
    write_experiment_manifest(path, manifest)
    loaded = json.loads(path.read_text(encoding="utf-8"))

    validate_experiment_manifest(loaded)
    assert loaded["train_image_count"] == 6
    assert loaded["test_image_count"] == 4
    assert loaded["configuration"]["iterations"] == 7000


def test_manifest_validation_rejects_train_test_overlap() -> None:
    manifest = {
        "experiment_name": "bad",
        "selection_method": "fixed_uniform",
        "training_images": ["a.jpg"],
        "test_images": ["a.jpg"],
        "train_image_count": 1,
        "test_image_count": 1,
        "total_scene_images": 1,
    }

    with pytest.raises(Phase5SelectionError, match="overlap"):
        validate_experiment_manifest(manifest)
