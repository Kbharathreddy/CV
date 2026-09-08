from __future__ import annotations

import json
from pathlib import Path

from src.convert_colmap import read_colmap_model
from src.frame_selection import build_experiment_manifest, create_heldout_split
from src.phase5_experiments import (
    ColmapCameraRecord,
    ColmapImageRecord,
    ColmapPoint3DRecord,
    ColmapRawModel,
    filter_colmap_model,
    prepare_phase5_scene,
    validate_experiment_name,
    write_colmap_binary_model,
)


def _raw_model() -> ColmapRawModel:
    cameras = {
        1: ColmapCameraRecord(
            camera_id=1,
            model="PINHOLE",
            width=640,
            height=480,
            params=[500.0, 500.0, 320.0, 240.0],
        )
    }
    images = {
        image_id: ColmapImageRecord(
            image_id=image_id,
            qvec=[1.0, 0.0, 0.0, 0.0],
            tvec=[float(image_id), 0.0, 0.0],
            camera_id=1,
            name=f"image_{image_id:03d}.jpg",
            points2d=[(10.0 + image_id, 20.0, 100), (30.0, 40.0, -1)],
        )
        for image_id in range(1, 5)
    }
    points = {
        100: ColmapPoint3DRecord(
            point3d_id=100,
            xyz=[1.0, 2.0, 3.0],
            rgb=(255, 128, 64),
            error=0.2,
            track=[(1, 0), (2, 0), (3, 0), (4, 0)],
        )
    }
    return ColmapRawModel(cameras=cameras, images=images, points3d=points, model_format="binary")


def _write_source_images(root: Path) -> None:
    root.mkdir(parents=True)
    for image_id in range(1, 5):
        (root / f"image_{image_id:03d}.jpg").write_bytes(b"synthetic-image")


def test_filter_colmap_model_preserves_selected_poses_and_tracks() -> None:
    filtered = filter_colmap_model(_raw_model(), ["image_001.jpg", "image_003.jpg"])

    assert sorted(image.name for image in filtered.images.values()) == ["image_001.jpg", "image_003.jpg"]
    assert sorted(filtered.cameras) == [1]
    assert filtered.points3d[100].track == [(1, 0), (3, 0)]


def test_write_filtered_binary_model_is_readable(tmp_path: Path) -> None:
    filtered = filter_colmap_model(_raw_model(), ["image_001.jpg", "image_002.jpg"])
    sparse_dir = tmp_path / "sparse" / "0"

    write_colmap_binary_model(filtered, sparse_dir)
    parsed = read_colmap_model(sparse_dir)

    assert len(parsed.images) == 2
    assert len(parsed.cameras) == 1
    assert len(parsed.points3d) == 1


def test_prepare_phase5_scene_uses_manifest_without_rerunning_colmap(tmp_path: Path) -> None:
    source_images = tmp_path / "source_images"
    source_model = tmp_path / "source_model"
    output = tmp_path / "phase5" / "fixed_2"
    _write_source_images(source_images)
    write_colmap_binary_model(_raw_model(), source_model)
    split = create_heldout_split(
        ["image_001.jpg", "image_002.jpg", "image_003.jpg", "image_004.jpg"],
        holdout_interval=4,
        holdout_offset=0,
    )
    manifest = build_experiment_manifest(
        split=split,
        experiment_name="fixed_2",
        selection_method="fixed_uniform",
        training_images=["image_002.jpg", "image_003.jpg"],
        training_budget=2,
    )

    prepared = prepare_phase5_scene(
        manifest=manifest,
        source_images=source_images,
        source_model=source_model,
        output_dir=output,
        link_mode="copy",
    )

    assert prepared.train_image_count == 2
    assert prepared.test_image_count == 1
    assert Path(prepared.train_scene, "images", "image_002.jpg").exists()
    assert Path(prepared.test_scene, "images", "image_001.jpg").exists()
    assert Path(prepared.train_scene, "sparse", "0", "images.bin").exists()
    scene_manifest = json.loads((output / "scene_manifest.json").read_text(encoding="utf-8"))
    assert scene_manifest["prepared_scene"]["dry_run"] is False


def test_prepare_phase5_scene_dry_run_creates_no_files(tmp_path: Path) -> None:
    split = create_heldout_split(["a.jpg", "b.jpg", "c.jpg", "d.jpg"], holdout_interval=4)
    manifest = build_experiment_manifest(
        split=split,
        experiment_name="dry",
        selection_method="fixed_uniform",
        training_images=["b.jpg"],
        training_budget=1,
    )

    prepared = prepare_phase5_scene(
        manifest=manifest,
        source_images=tmp_path / "missing_images",
        source_model=tmp_path / "missing_sparse",
        output_dir=tmp_path / "out",
        dry_run=True,
    )

    assert prepared.dry_run is True
    assert not (tmp_path / "out").exists()


def test_validate_experiment_name_allows_known_phase5_names() -> None:
    assert validate_experiment_name("fixed_30") == "fixed_30"
