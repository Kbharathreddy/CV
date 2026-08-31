"""Dataset acquisition utilities and CLI entry point.

Phase 1 supports Mip-NeRF 360 scene extraction from the official public Google
Storage archives linked by the project page.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

if __package__ in {None, ""}:  # Allows: python src/download_dataset.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import discover_images, ensure_dir, format_bytes

LOGGER_NAME = "smartphone_3dgs.dataset"
MIPNERF360_PROJECT_PAGE = "https://jonbarron.info/mipnerf360/"


class DatasetDownloadError(RuntimeError):
    """Raised when a dataset cannot be downloaded, extracted, or validated."""


@dataclass(frozen=True)
class ArchiveSource:
    """Official archive metadata for a dataset scene."""

    filename: str
    url: str
    scenes: frozenset[str]


@dataclass(frozen=True)
class DatasetResult:
    """Summary returned by downloader operations."""

    dataset: str
    scene: str
    status: str
    source_url: str
    destination: str
    archive_path: str
    image_count: int
    message: str


MIPNERF360_ARCHIVES: tuple[ArchiveSource, ...] = (
    ArchiveSource(
        filename="360_v2.zip",
        url="https://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
        scenes=frozenset({"bicycle", "bonsai", "counter", "garden", "kitchen", "room", "stump"}),
    ),
    ArchiveSource(
        filename="360_extra_scenes.zip",
        url="https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip",
        scenes=frozenset({"flowers", "treehill"}),
    ),
)


def get_mipnerf360_source(scene: str) -> ArchiveSource:
    """Return the official archive that contains a Mip-NeRF 360 scene."""

    normalized = scene.strip().lower()
    for source in MIPNERF360_ARCHIVES:
        if normalized in source.scenes:
            return source
    valid = sorted(scene for source in MIPNERF360_ARCHIVES for scene in source.scenes)
    raise DatasetDownloadError(
        f"Unsupported Mip-NeRF 360 scene '{scene}'. Valid scenes: {', '.join(valid)}"
    )


class _ProgressBar:
    """Small tqdm wrapper with a no-dependency fallback."""

    def __init__(self, total: int | None, description: str) -> None:
        self._bar = None
        try:
            from tqdm import tqdm  # type: ignore

            self._bar = tqdm(total=total, unit="B", unit_scale=True, desc=description)
        except Exception:
            self._seen = 0
            self._total = total
            self._description = description
            print(description)

    def update(self, count: int) -> None:
        if self._bar is not None:
            self._bar.update(count)
            return
        self._seen += count
        if self._total:
            percent = min(100.0, 100.0 * self._seen / self._total)
            print(f"\r  {format_bytes(self._seen)} / {format_bytes(self._total)} ({percent:.1f}%)", end="")

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
        else:
            print()


def manual_download_instructions(source: ArchiveSource, archive_path: Path, scene: str) -> str:
    """Return exact manual download instructions for environments without access."""

    return (
        f"Automatic download was not completed for Mip-NeRF 360 scene '{scene}'.\n"
        f"Download the official archive from:\n  {source.url}\n"
        f"Expected archive name:\n  {source.filename}\n"
        f"Place it at:\n  {archive_path}\n"
        "Then continue with:\n"
        f"  python src/download_dataset.py --dataset mipnerf360 --scene {scene} "
        f"--archive-path {archive_path.as_posix()}"
    )


def download_file(url: str, destination: Path, logger: logging.Logger) -> None:
    """Download a URL to a destination path with size validation."""

    ensure_dir(destination.parent)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "smartphone-3dgs/0.1"})

    logger.info("Downloading %s to %s", url, destination)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else None
            progress = _ProgressBar(total, f"Downloading {destination.name}")
            try:
                with temp_path.open("wb") as file:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        file.write(chunk)
                        progress.update(len(chunk))
            finally:
                progress.close()
    except (OSError, urllib.error.URLError) as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise DatasetDownloadError(f"Failed to download {url}: {exc}") from exc

    if not temp_path.exists() or temp_path.stat().st_size == 0:
        if temp_path.exists():
            temp_path.unlink()
        raise DatasetDownloadError(f"Downloaded archive is empty: {destination}")

    temp_path.replace(destination)
    logger.info("Downloaded archive size: %s", format_bytes(destination.stat().st_size))


def _belongs_to_scene(member_name: str, scene: str) -> bool:
    parts = PurePosixPath(member_name).parts
    return bool(parts) and parts[0] == scene


def _safe_target(root: Path, relative_parts: Iterable[str]) -> Path:
    for part in relative_parts:
        if part in {"", ".", ".."}:
            raise DatasetDownloadError(f"Unsafe archive path component: {part!r}")
    target = root.joinpath(*relative_parts).resolve()
    root_resolved = root.resolve()
    if os.path.commonpath([str(root_resolved), str(target)]) != str(root_resolved):
        raise DatasetDownloadError(f"Archive member would escape destination: {target}")
    return target


def extract_scene_from_archive(archive_path: Path, scene: str, destination: Path, logger: logging.Logger) -> None:
    """Extract one scene from an official Mip-NeRF 360 archive."""

    if not archive_path.exists():
        raise DatasetDownloadError(f"Archive not found: {archive_path}")
    if archive_path.stat().st_size == 0:
        raise DatasetDownloadError(f"Archive is empty: {archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise DatasetDownloadError(f"Archive is not a valid zip file: {archive_path}")

    temp_destination = destination.with_name(destination.name + ".partial")
    if temp_destination.exists():
        shutil.rmtree(temp_destination)
    ensure_dir(temp_destination)

    extracted = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = [info for info in archive.infolist() if _belongs_to_scene(info.filename, scene)]
            if not members:
                raise DatasetDownloadError(f"Scene '{scene}' was not found in {archive_path}")

            for member in members:
                parts = PurePosixPath(member.filename).parts
                relative_parts = parts[1:]
                if not relative_parts:
                    continue
                target = _safe_target(temp_destination, relative_parts)
                if member.is_dir():
                    ensure_dir(target)
                    continue
                ensure_dir(target.parent)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted += 1
    except zipfile.BadZipFile as exc:
        raise DatasetDownloadError(f"Corrupted zip archive: {archive_path}") from exc
    except Exception:
        if temp_destination.exists():
            shutil.rmtree(temp_destination)
        raise

    if extracted == 0:
        shutil.rmtree(temp_destination)
        raise DatasetDownloadError(f"No files were extracted for scene '{scene}'.")

    if destination.exists():
        shutil.rmtree(destination)
    temp_destination.replace(destination)
    logger.info("Extracted %d files for scene '%s' to %s", extracted, scene, destination)


def validate_rgb_images(scene_dir: Path) -> int:
    """Validate discovered image files and return the count."""

    image_paths = discover_images(scene_dir)
    if not image_paths:
        raise DatasetDownloadError(f"No RGB images found under {scene_dir}")

    try:
        from PIL import Image
    except Exception as exc:
        raise DatasetDownloadError("Pillow is required to validate dataset images.") from exc

    invalid: list[str] = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                image.verify()
                if image.mode not in {"RGB", "RGBA", "L"}:
                    invalid.append(f"{path} has mode {image.mode}")
        except Exception as exc:
            invalid.append(f"{path}: {exc}")

    if invalid:
        details = "\n".join(invalid[:10])
        raise DatasetDownloadError(f"Invalid image files detected:\n{details}")

    return len(image_paths)


def scene_already_available(scene_dir: Path) -> bool:
    """Return true when an extracted scene appears usable."""

    return scene_dir.exists() and bool(discover_images(scene_dir))


def download_mipnerf360_scene(
    scene: str,
    *,
    data_root: Path,
    archive_path: Path | None = None,
    force: bool = False,
    logger: logging.Logger | None = None,
) -> DatasetResult:
    """Download or organize one Mip-NeRF 360 scene."""

    logger = logger or logging.getLogger(LOGGER_NAME)
    source = get_mipnerf360_source(scene)
    dataset_root = data_root / "public" / "mipnerf360"
    scene_dir = dataset_root / scene
    archives_dir = dataset_root / "_archives"
    archive_path = archive_path or archives_dir / source.filename

    logger.info("Dataset source page: %s", MIPNERF360_PROJECT_PAGE)
    logger.info("Requested Mip-NeRF 360 scene: %s", scene)
    logger.info("Scene destination: %s", scene_dir)

    if scene_already_available(scene_dir) and not force:
        image_count = validate_rgb_images(scene_dir)
        return DatasetResult(
            dataset="mipnerf360",
            scene=scene,
            status="exists",
            source_url=source.url,
            destination=str(scene_dir),
            archive_path=str(archive_path),
            image_count=image_count,
            message="Scene already exists; no download performed.",
        )

    if not archive_path.exists():
        try:
            download_file(source.url, archive_path, logger)
        except DatasetDownloadError as exc:
            instructions = manual_download_instructions(source, archive_path, scene)
            raise DatasetDownloadError(f"{exc}\n\n{instructions}") from exc
    elif archive_path.stat().st_size == 0:
        raise DatasetDownloadError(
            f"Archive exists but is empty: {archive_path}\n\n"
            f"{manual_download_instructions(source, archive_path, scene)}"
        )
    else:
        logger.info("Using existing archive: %s (%s)", archive_path, format_bytes(archive_path.stat().st_size))

    extract_scene_from_archive(archive_path, scene, scene_dir, logger)
    image_count = validate_rgb_images(scene_dir)
    return DatasetResult(
        dataset="mipnerf360",
        scene=scene,
        status="downloaded",
        source_url=source.url,
        destination=str(scene_dir),
        archive_path=str(archive_path),
        image_count=image_count,
        message="Scene extracted and validated.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or organize supported public datasets.")
    parser.add_argument("--dataset", required=True, choices=["mipnerf360"], help="Dataset name.")
    parser.add_argument("--scene", required=True, help="Scene name, for example 'bonsai'.")
    parser.add_argument("--data-root", default="data", help="Local data root.")
    parser.add_argument(
        "--archive-path",
        default=None,
        help="Path to an already downloaded official archive. Useful for manual download flow.",
    )
    parser.add_argument("--force", action="store_true", help="Re-extract even when a scene already exists.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger(LOGGER_NAME)

    data_root = Path(args.data_root).expanduser()
    archive_path = Path(args.archive_path).expanduser() if args.archive_path else None

    try:
        result = download_mipnerf360_scene(
            args.scene.lower(),
            data_root=data_root,
            archive_path=archive_path,
            force=args.force,
            logger=logger,
        )
    except DatasetDownloadError as exc:
        logger.error("%s", exc)
        return 2

    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
