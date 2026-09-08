"""Thin helpers for Phase 4 Gaussian Splatting smoke-test orchestration.

The project keeps the training implementation as an external dependency. This
module validates input layout, prepares portable Kaggle input bundles, and
builds commands for a pinned 3DGS implementation without importing CUDA-heavy
libraries locally.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:  # Allows: python src/gaussian_splatting.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import PROJECT_ROOT, discover_images, ensure_dir, write_json


class GaussianSplattingError(RuntimeError):
    """Raised when Phase 4 input preparation or command construction fails."""


OFFICIAL_3DGS_REPO_URL = "https://github.com/graphdeco-inria/gaussian-splatting.git"
OFFICIAL_3DGS_COMMIT = "54c035f7834b564019656c3e3fcc3646292f727d"
OFFICIAL_3DGS_LICENSE_URL = "https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md"
OFFICIAL_3DGS_LICENSE_SUMMARY = "Non-commercial research/evaluation license from Inria and MPII"
REQUIRED_SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")


@dataclass(frozen=True)
class GaussianSplattingImplementation:
    """External 3DGS implementation metadata recorded for reproducibility."""

    name: str
    repository_url: str
    commit: str
    license_summary: str
    license_url: str

    @classmethod
    def official_graphdeco(cls) -> "GaussianSplattingImplementation":
        return cls(
            name="graphdeco-inria/gaussian-splatting",
            repository_url=OFFICIAL_3DGS_REPO_URL,
            commit=OFFICIAL_3DGS_COMMIT,
            license_summary=OFFICIAL_3DGS_LICENSE_SUMMARY,
            license_url=OFFICIAL_3DGS_LICENSE_URL,
        )


@dataclass(frozen=True)
class ColmapSceneManifest:
    """Minimal validated COLMAP scene metadata for 3DGS input."""

    scene_root: str
    images_dir: str
    sparse_model_dir: str
    image_count: int
    image_extensions: list[str]
    sparse_files: list[str]


@dataclass(frozen=True)
class KaggleInputManifest:
    """Manifest for a Kaggle input bundle prepared from existing local outputs."""

    scene_name: str
    output_dir: str
    scene_root: str
    images_dir: str
    sparse_model_dir: str
    source_images_dir: str
    source_sparse_model_dir: str
    image_count: int
    image_extensions: list[str]
    sparse_files: list[str]
    archive_path: str | None = None


@dataclass(frozen=True)
class GaussianSmokeConfig:
    """Training/render command configuration for a short 3DGS smoke test."""

    source_path: str | Path
    model_path: str | Path
    implementation_dir: str | Path
    iterations: int = 300
    images_subdir: str = "images"
    python_executable: str = "python"
    quiet: bool = True
    disable_viewer: bool = True
    test_iterations: tuple[int, ...] = (-1,)
    save_iterations: tuple[int, ...] | None = None
    checkpoint_iterations: tuple[int, ...] | None = None
    extra_train_args: tuple[str, ...] = field(default_factory=tuple)
    extra_render_args: tuple[str, ...] = field(default_factory=tuple)

    def resolved_save_iterations(self) -> tuple[int, ...]:
        return self.save_iterations if self.save_iterations is not None else (self.iterations,)

    def resolved_checkpoint_iterations(self) -> tuple[int, ...]:
        if self.checkpoint_iterations is not None:
            return self.checkpoint_iterations
        return (self.iterations,)


def validate_smoke_config(config: GaussianSmokeConfig) -> None:
    """Validate basic smoke-test command settings."""

    if int(config.iterations) <= 0:
        raise GaussianSplattingError("Smoke-test iterations must be positive.")
    if not config.images_subdir:
        raise GaussianSplattingError("images_subdir must not be empty.")
    if not config.python_executable:
        raise GaussianSplattingError("python_executable must not be empty.")


def _extend_int_args(command: list[str], flag: str, values: Iterable[int]) -> None:
    command.append(flag)
    command.extend(str(int(value)) for value in values)


def build_training_command(config: GaussianSmokeConfig) -> list[str]:
    """Build the official GraphDeco `train.py` smoke-test command."""

    validate_smoke_config(config)
    command = [
        config.python_executable,
        "train.py",
        "-s",
        str(config.source_path),
        "-m",
        str(config.model_path),
        "-i",
        config.images_subdir,
        "--iterations",
        str(int(config.iterations)),
    ]
    _extend_int_args(command, "--test_iterations", config.test_iterations)
    _extend_int_args(command, "--save_iterations", config.resolved_save_iterations())
    _extend_int_args(command, "--checkpoint_iterations", config.resolved_checkpoint_iterations())
    if config.disable_viewer:
        command.append("--disable_viewer")
    if config.quiet:
        command.append("--quiet")
    command.extend(config.extra_train_args)
    return command


def build_render_command(config: GaussianSmokeConfig, *, iteration: int | None = None) -> list[str]:
    """Build a render command for the smoke-test checkpoint."""

    validate_smoke_config(config)
    selected_iteration = config.iterations if iteration is None else int(iteration)
    if selected_iteration <= 0:
        raise GaussianSplattingError("Render iteration must be positive.")
    command = [
        config.python_executable,
        "render.py",
        "-s",
        str(config.source_path),
        "-m",
        str(config.model_path),
        "-i",
        config.images_subdir,
        "--iteration",
        str(selected_iteration),
        "--skip_test",
    ]
    if config.quiet:
        command.append("--quiet")
    command.extend(config.extra_render_args)
    return command


def expected_checkpoint_path(config: GaussianSmokeConfig) -> Path:
    """Return the checkpoint path written by the official training script."""

    return Path(config.model_path) / f"chkpnt{int(config.iterations)}.pth"


def expected_point_cloud_path(config: GaussianSmokeConfig) -> Path:
    """Return the saved point-cloud path for the requested iteration."""

    return Path(config.model_path) / "point_cloud" / f"iteration_{int(config.iterations)}" / "point_cloud.ply"


def expected_render_dir(config: GaussianSmokeConfig, *, iteration: int | None = None) -> Path:
    """Return the official train-set render output directory."""

    selected_iteration = config.iterations if iteration is None else int(iteration)
    return Path(config.model_path) / "train" / f"ours_{selected_iteration}" / "renders"


def validate_sparse_model_dir(sparse_model_dir: str | Path) -> list[Path]:
    """Validate that a COLMAP sparse/0 directory has required binary files."""

    directory = Path(sparse_model_dir)
    if not directory.exists() or not directory.is_dir():
        raise GaussianSplattingError(f"Missing COLMAP sparse model directory: {directory}")
    missing = [name for name in REQUIRED_SPARSE_FILES if not (directory / name).exists()]
    if missing:
        raise GaussianSplattingError(
            f"Missing required COLMAP sparse model files in {directory}: {', '.join(missing)}"
        )
    return sorted(path for path in directory.iterdir() if path.is_file())


def validate_colmap_scene(
    scene_root: str | Path,
    *,
    images_subdir: str = "images",
    expected_images: int | None = None,
) -> ColmapSceneManifest:
    """Validate a 3DGS-ready COLMAP scene layout."""

    root = Path(scene_root)
    images_dir = root / images_subdir
    sparse_model_dir = root / "sparse" / "0"
    images = discover_images(images_dir)
    if not images:
        raise GaussianSplattingError(f"No input images found in {images_dir}")
    if expected_images is not None and len(images) != expected_images:
        raise GaussianSplattingError(
            f"Expected {expected_images} images in {images_dir}, found {len(images)}"
        )
    sparse_files = validate_sparse_model_dir(sparse_model_dir)
    return ColmapSceneManifest(
        scene_root=str(root),
        images_dir=str(images_dir),
        sparse_model_dir=str(sparse_model_dir),
        image_count=len(images),
        image_extensions=sorted({path.suffix.lower() for path in images}),
        sparse_files=[path.name for path in sparse_files],
    )


def _reject_dataset_reference_sparse(sparse_model_dir: Path) -> None:
    """Avoid accidentally packaging the downloaded dataset reference model."""

    reference_sparse = (PROJECT_ROOT / "data" / "public" / "mipnerf360" / "bonsai" / "sparse").resolve()
    resolved = sparse_model_dir.resolve()
    try:
        resolved.relative_to(reference_sparse)
    except ValueError:
        return
    raise GaussianSplattingError(
        "Refusing to package the dataset-provided reference sparse model. "
        "Use outputs/mipnerf_bonsai_colmap/sparse/0 from our Phase 3 reconstruction."
    )


def _copy_files(paths: Iterable[Path], destination: Path) -> None:
    ensure_dir(destination)
    for path in paths:
        shutil.copy2(path, destination / path.name)


def _archive_directory(root: Path, archive_path: Path) -> Path:
    ensure_dir(archive_path.parent)
    with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    return archive_path


def prepare_kaggle_bonsai_input(
    *,
    images_dir: str | Path,
    sparse_model_dir: str | Path,
    output_dir: str | Path,
    scene_name: str = "bonsai",
    archive_path: str | Path | None = None,
    overwrite: bool = False,
) -> KaggleInputManifest:
    """Prepare `bonsai/images` and `bonsai/sparse/0` for Kaggle upload."""

    source_images_dir = Path(images_dir)
    source_sparse_dir = Path(sparse_model_dir)
    target_root = Path(output_dir)
    if target_root.exists() and any(target_root.iterdir()):
        if not overwrite:
            raise GaussianSplattingError(
                f"Kaggle input output directory already exists and is non-empty: {target_root}"
            )
        shutil.rmtree(target_root)

    images = discover_images(source_images_dir)
    if not images:
        raise GaussianSplattingError(f"No images found in source directory: {source_images_dir}")
    _reject_dataset_reference_sparse(source_sparse_dir)
    sparse_files = validate_sparse_model_dir(source_sparse_dir)

    scene_root = target_root / scene_name
    target_images_dir = scene_root / "images"
    target_sparse_dir = scene_root / "sparse" / "0"
    _copy_files(images, target_images_dir)
    _copy_files(sparse_files, target_sparse_dir)

    manifest = KaggleInputManifest(
        scene_name=scene_name,
        output_dir=str(target_root),
        scene_root=str(scene_root),
        images_dir=str(target_images_dir),
        sparse_model_dir=str(target_sparse_dir),
        source_images_dir=str(source_images_dir),
        source_sparse_model_dir=str(source_sparse_dir),
        image_count=len(images),
        image_extensions=sorted({path.suffix.lower() for path in images}),
        sparse_files=[path.name for path in sparse_files],
    )

    if archive_path is not None:
        manifest = KaggleInputManifest(**{**asdict(manifest), "archive_path": str(Path(archive_path))})

    write_json(target_root / "manifest.json", asdict(manifest))
    if archive_path is not None:
        _archive_directory(target_root, Path(archive_path))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the Phase 4 Kaggle bonsai 3DGS input bundle.")
    parser.add_argument("--images-dir", default="data/public/mipnerf360/bonsai/images_4")
    parser.add_argument("--sparse-dir", default="outputs/mipnerf_bonsai_colmap/sparse/0")
    parser.add_argument("--output-dir", default="outputs/kaggle/bonsai_3dgs_input")
    parser.add_argument("--archive-path", default="outputs/kaggle/bonsai_3dgs_input.zip")
    parser.add_argument("--scene-name", default="bonsai")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-archive", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = prepare_kaggle_bonsai_input(
            images_dir=args.images_dir,
            sparse_model_dir=args.sparse_dir,
            output_dir=args.output_dir,
            scene_name=args.scene_name,
            archive_path=None if args.no_archive else args.archive_path,
            overwrite=args.overwrite,
        )
    except GaussianSplattingError as exc:
        print(f"Kaggle input preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
