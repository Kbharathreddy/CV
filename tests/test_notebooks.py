from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _notebook_source(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in data["cells"])


def test_phase5_kaggle_notebook_is_valid_json_and_single_experiment_runner() -> None:
    notebook_path = PROJECT_ROOT / "notebooks" / "phase5_kaggle.ipynb"
    source = _notebook_source(notebook_path)

    assert 'EXPERIMENT_NAME = "fixed_30"' in source
    assert "fixed_30, fixed_60, fixed_100, automatic_60, full" in source
    assert "prepare_phase5_experiment.py" in source
    assert "evaluate_phase5.py" in source
    assert "train_scene" in source
    assert "test_scene" in source
    assert "assert train_images.isdisjoint(test_images)" in source
    assert "ITERATIONS != 300" in source
    assert "54c035f7834b564019656c3e3fcc3646292f727d" in source
    assert r"(?<!Depth )Loss=([0-9eE+\-.]+)" in source


def test_phase4_kaggle_notebook_still_exists() -> None:
    notebook_path = PROJECT_ROOT / "notebooks" / "phase4_kaggle.ipynb"

    assert notebook_path.exists()
