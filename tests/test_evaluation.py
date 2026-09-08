from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.evaluation import evaluate_heldout_views, psnr, ssim


def _image(path: Path, value: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(array, mode="RGB").save(path)
    return path


def _manifest() -> dict:
    return {
        "experiment_name": "fixed_1",
        "selection_method": "fixed_uniform",
        "training_images": ["train.jpg"],
        "test_images": ["test.jpg"],
        "train_image_count": 1,
        "test_image_count": 1,
        "total_scene_images": 2,
    }


def test_identical_images_have_infinite_psnr_and_unit_ssim() -> None:
    reference = np.full((16, 16, 3), 0.5, dtype=np.float32)
    prediction = np.full((16, 16, 3), 0.5, dtype=np.float32)

    assert math.isinf(psnr(reference, prediction))
    assert ssim(reference, prediction) == pytest.approx(1.0)


def test_different_images_reduce_psnr_and_ssim() -> None:
    reference = np.full((16, 16, 3), 0.2, dtype=np.float32)
    prediction = np.full((16, 16, 3), 0.8, dtype=np.float32)

    assert psnr(reference, prediction) < 10.0
    assert ssim(reference, prediction) < 0.5


def test_evaluate_heldout_views_uses_manifest_test_images(tmp_path: Path) -> None:
    ground_truth = tmp_path / "gt"
    renders = tmp_path / "renders"
    _image(ground_truth / "test.jpg", 128)
    _image(renders / "00000.png", 128)

    summary = evaluate_heldout_views(
        ground_truth_dir=ground_truth,
        render_dir=renders,
        experiment_manifest=_manifest(),
    )

    assert summary.experiment == "fixed_1"
    assert summary.test_image_count == 1
    assert math.isinf(summary.psnr_mean or 0.0)
    assert summary.ssim_mean == pytest.approx(1.0)
    assert summary.lpips_mean is None
