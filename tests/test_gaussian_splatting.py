from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.gaussian_splatting import (
    GaussianSmokeConfig,
    GaussianSplattingError,
    build_render_command,
    build_training_command,
    expected_checkpoint_path,
    expected_point_cloud_path,
    expected_render_dir,
    prepare_kaggle_bonsai_input,
    validate_colmap_scene,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_minimal_colmap_scene(root: Path, *, image_count: int = 2) -> None:
    images_dir = root / "images"
    sparse_dir = root / "sparse" / "0"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    for index in range(image_count):
        (images_dir / f"frame_{index:06d}.jpg").write_bytes(b"not-a-real-jpeg")
    for filename in ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"):
        (sparse_dir / filename).write_bytes(b"phase4-test")


def test_build_training_command_for_official_smoke_test() -> None:
    config = GaussianSmokeConfig(
        source_path="/kaggle/working/bonsai_scene",
        model_path="/kaggle/working/outputs/kaggle/bonsai_smoke",
        implementation_dir="/kaggle/working/gaussian-splatting",
        iterations=300,
    )

    command = build_training_command(config)

    assert command[:2] == ["python", "train.py"]
    assert command[command.index("-s") + 1] == "/kaggle/working/bonsai_scene"
    assert command[command.index("-i") + 1] == "images"
    assert command[command.index("--iterations") + 1] == "300"
    assert "--disable_viewer" in command
    assert "--quiet" in command
    assert command[command.index("--checkpoint_iterations") + 1] == "300"


def test_build_render_command_targets_smoke_iteration() -> None:
    config = GaussianSmokeConfig(
        source_path="bonsai_scene",
        model_path="outputs/kaggle/bonsai_smoke",
        implementation_dir="external/gaussian-splatting",
        iterations=200,
    )

    command = build_render_command(config)

    assert command[:2] == ["python", "render.py"]
    assert command[command.index("--iteration") + 1] == "200"
    assert "--skip_test" in command


def test_expected_smoke_output_paths() -> None:
    config = GaussianSmokeConfig(
        source_path="bonsai_scene",
        model_path="outputs/kaggle/bonsai_smoke",
        implementation_dir="external/gaussian-splatting",
        iterations=150,
    )

    assert expected_checkpoint_path(config) == Path("outputs/kaggle/bonsai_smoke/chkpnt150.pth")
    assert expected_point_cloud_path(config) == Path(
        "outputs/kaggle/bonsai_smoke/point_cloud/iteration_150/point_cloud.ply"
    )
    assert expected_render_dir(config) == Path("outputs/kaggle/bonsai_smoke/train/ours_150/renders")


def test_validate_colmap_scene_reports_images_and_sparse_files(tmp_path: Path) -> None:
    scene_root = tmp_path / "bonsai"
    _write_minimal_colmap_scene(scene_root, image_count=3)

    manifest = validate_colmap_scene(scene_root, expected_images=3)

    assert manifest.image_count == 3
    assert manifest.image_extensions == [".jpg"]
    assert "cameras.bin" in manifest.sparse_files
    assert "images.bin" in manifest.sparse_files
    assert "points3D.bin" in manifest.sparse_files


def test_validate_colmap_scene_rejects_missing_sparse_model(tmp_path: Path) -> None:
    scene_root = tmp_path / "bonsai"
    (scene_root / "images").mkdir(parents=True)
    (scene_root / "images" / "frame_000000.jpg").write_bytes(b"image")

    with pytest.raises(GaussianSplattingError, match="Missing COLMAP sparse model"):
        validate_colmap_scene(scene_root)


def test_prepare_kaggle_bonsai_input_copies_only_required_layout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output_dir = tmp_path / "kaggle_input"
    archive_path = tmp_path / "kaggle_input.zip"
    _write_minimal_colmap_scene(source, image_count=2)

    manifest = prepare_kaggle_bonsai_input(
        images_dir=source / "images",
        sparse_model_dir=source / "sparse" / "0",
        output_dir=output_dir,
        archive_path=archive_path,
    )

    assert manifest.image_count == 2
    assert Path(manifest.images_dir).is_dir()
    assert Path(manifest.sparse_model_dir, "cameras.bin").exists()
    assert Path(manifest.output_dir, "manifest.json").exists()
    assert archive_path.exists()
    with zipfile.ZipFile(archive_path) as archive:
        assert "manifest.json" in archive.namelist()
    assert (output_dir / "bonsai" / "images" / "frame_000000.jpg").exists()
    assert (output_dir / "bonsai" / "sparse" / "0" / "points3D.bin").exists()


def test_phase4_kaggle_notebook_uses_writable_scene_and_depth_safe_loss_parser() -> None:
    import json

    notebook_path = PROJECT_ROOT / "notebooks" / "phase4_kaggle.ipynb"
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in data["cells"])

    assert "/kaggle/input/datasets/kopperlabharathreddy/" in source
    assert "mip-nerf-360-bonsai-our-colmap-reconstruction/bonsai" in source
    assert 'WORK_ROOT = Path("/kaggle/working")' in source
    assert 'SCENE_PATH = WORK_ROOT / "bonsai_scene"' in source
    assert 'MODEL_PATH = WORK_ROOT / "outputs/kaggle/bonsai_smoke"' in source
    assert "shutil.copytree" in source
    assert '"-s", str(SCENE_PATH)' in source
    assert "54c035f7834b564019656c3e3fcc3646292f727d" in source
    assert "ITERATIONS = 300" in source
    assert r"(?<!Depth )Loss=([0-9eE+\-.]+)" in source
    assert r"Loss[=:]\s*" not in source
