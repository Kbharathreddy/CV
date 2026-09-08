"""Read and summarize COLMAP sparse reconstruction models.

The parser supports COLMAP binary and text sparse models. It reads cameras,
registered image poses, sparse 3D points, and track observations without
requiring COLMAP to be installed.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

if __package__ in {None, ""}:  # Allows: python src/convert_colmap.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import write_json


class ColmapModelError(RuntimeError):
    """Raised when COLMAP sparse model files are missing or malformed."""


CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_IDS = {name: (model_id, params) for model_id, (name, params) in CAMERA_MODELS.items()}
INVALID_POINT3D_IDS = {-1, 2**64 - 1}


@dataclass(frozen=True)
class Camera:
    """COLMAP camera intrinsics."""

    camera_id: int
    model: str
    width: int
    height: int
    params: list[float]


@dataclass(frozen=True)
class ImagePose:
    """COLMAP registered image pose.

    qvec and tvec follow COLMAP's world-to-camera convention.
    """

    image_id: int
    qvec: list[float]
    tvec: list[float]
    camera_id: int
    name: str
    num_points2d: int
    num_observed_points: int
    camera_center: list[float]


@dataclass(frozen=True)
class Point3D:
    """Sparse COLMAP 3D point with color, reprojection error, and track length."""

    point3d_id: int
    xyz: list[float]
    rgb: list[int]
    error: float
    track_length: int


@dataclass(frozen=True)
class ColmapModel:
    """Parsed COLMAP sparse model."""

    cameras: dict[int, Camera]
    images: dict[int, ImagePose]
    points3d: dict[int, Point3D]
    model_path: str
    model_format: str


def _read_exact(file: BinaryIO, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise ColmapModelError("Unexpected end of COLMAP binary model.")
    return data


def _unpack(file: BinaryIO, fmt: str) -> tuple:
    return struct.unpack("<" + fmt, _read_exact(file, struct.calcsize("<" + fmt)))


def _read_c_string(file: BinaryIO) -> str:
    chars = bytearray()
    while True:
        char = file.read(1)
        if not char:
            raise ColmapModelError("Unexpected end of file while reading image name.")
        if char == b"\x00":
            return chars.decode("utf-8")
        chars.extend(char)


def qvec_to_rotation_matrix(qvec: Iterable[float]) -> list[list[float]]:
    """Convert COLMAP quaternion `[qw, qx, qy, qz]` to a 3x3 rotation matrix."""

    qw, qx, qy, qz = [float(value) for value in qvec]
    return [
        [
            1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
            2.0 * qx * qy - 2.0 * qz * qw,
            2.0 * qx * qz + 2.0 * qy * qw,
        ],
        [
            2.0 * qx * qy + 2.0 * qz * qw,
            1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
            2.0 * qy * qz - 2.0 * qx * qw,
        ],
        [
            2.0 * qx * qz - 2.0 * qy * qw,
            2.0 * qy * qz + 2.0 * qx * qw,
            1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
        ],
    ]


def camera_center_from_pose(qvec: Iterable[float], tvec: Iterable[float]) -> list[float]:
    """Return camera center `-R.T @ t` from COLMAP world-to-camera pose."""

    rotation = qvec_to_rotation_matrix(qvec)
    tx, ty, tz = [float(value) for value in tvec]
    return [
        -(rotation[0][0] * tx + rotation[1][0] * ty + rotation[2][0] * tz),
        -(rotation[0][1] * tx + rotation[1][1] * ty + rotation[2][1] * tz),
        -(rotation[0][2] * tx + rotation[1][2] * ty + rotation[2][2] * tz),
    ]


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    """Read `cameras.bin`."""

    cameras: dict[int, Camera] = {}
    try:
        with path.open("rb") as file:
            (num_cameras,) = _unpack(file, "Q")
            for _ in range(num_cameras):
                camera_id, model_id, width, height = _unpack(file, "iiQQ")
                if model_id not in CAMERA_MODELS:
                    raise ColmapModelError(f"Unsupported COLMAP camera model id: {model_id}")
                model_name, num_params = CAMERA_MODELS[model_id]
                params = list(_unpack(file, "d" * num_params))
                cameras[int(camera_id)] = Camera(
                    camera_id=int(camera_id),
                    model=model_name,
                    width=int(width),
                    height=int(height),
                    params=[float(value) for value in params],
                )
    except OSError as exc:
        raise ColmapModelError(f"Could not read cameras file: {path}") from exc
    return cameras


def read_images_binary(path: Path) -> dict[int, ImagePose]:
    """Read `images.bin`."""

    images: dict[int, ImagePose] = {}
    try:
        with path.open("rb") as file:
            (num_images,) = _unpack(file, "Q")
            for _ in range(num_images):
                values = _unpack(file, "idddddddi")
                image_id = int(values[0])
                qvec = [float(value) for value in values[1:5]]
                tvec = [float(value) for value in values[5:8]]
                camera_id = int(values[8])
                name = _read_c_string(file)
                (num_points2d,) = _unpack(file, "Q")
                observed = 0
                for _point_index in range(num_points2d):
                    _x, _y, point3d_id = _unpack(file, "ddq")
                    if int(point3d_id) not in INVALID_POINT3D_IDS:
                        observed += 1
                images[image_id] = ImagePose(
                    image_id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=name,
                    num_points2d=int(num_points2d),
                    num_observed_points=observed,
                    camera_center=camera_center_from_pose(qvec, tvec),
                )
    except OSError as exc:
        raise ColmapModelError(f"Could not read images file: {path}") from exc
    return images


def read_points3d_binary(path: Path) -> dict[int, Point3D]:
    """Read `points3D.bin`."""

    points: dict[int, Point3D] = {}
    try:
        with path.open("rb") as file:
            (num_points,) = _unpack(file, "Q")
            for _ in range(num_points):
                point_id = int(_unpack(file, "Q")[0])
                xyz = [float(value) for value in _unpack(file, "ddd")]
                rgb = [int(value) for value in _unpack(file, "BBB")]
                error = float(_unpack(file, "d")[0])
                track_length = int(_unpack(file, "Q")[0])
                for _track_index in range(track_length):
                    _image_id, _point2d_index = _unpack(file, "ii")
                points[point_id] = Point3D(
                    point3d_id=point_id,
                    xyz=xyz,
                    rgb=rgb,
                    error=error,
                    track_length=track_length,
                )
    except OSError as exc:
        raise ColmapModelError(f"Could not read points3D file: {path}") from exc
    return points


def _iter_data_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if line and not line.startswith("#"):
                yield line


def read_cameras_text(path: Path) -> dict[int, Camera]:
    """Read `cameras.txt`."""

    cameras: dict[int, Camera] = {}
    for line in _iter_data_lines(path):
        parts = line.split()
        if len(parts) < 5:
            raise ColmapModelError(f"Malformed cameras.txt line: {line}")
        camera_id = int(parts[0])
        model = parts[1]
        width = int(parts[2])
        height = int(parts[3])
        params = [float(value) for value in parts[4:]]
        cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras


def read_images_text(path: Path) -> dict[int, ImagePose]:
    """Read `images.txt`."""

    images: dict[int, ImagePose] = {}
    lines = list(_iter_data_lines(path))
    if len(lines) % 2 != 0:
        raise ColmapModelError("images.txt should contain two non-comment lines per registered image.")
    for metadata_line, points_line in zip(lines[0::2], lines[1::2]):
        parts = metadata_line.split()
        if len(parts) < 10:
            raise ColmapModelError(f"Malformed images.txt metadata line: {metadata_line}")
        image_id = int(parts[0])
        qvec = [float(value) for value in parts[1:5]]
        tvec = [float(value) for value in parts[5:8]]
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        point_parts = points_line.split()
        if len(point_parts) % 3 != 0:
            raise ColmapModelError(f"Malformed images.txt points line for image {image_id}")
        observed = sum(int(point_parts[index + 2]) != -1 for index in range(0, len(point_parts), 3))
        images[image_id] = ImagePose(
            image_id=image_id,
            qvec=qvec,
            tvec=tvec,
            camera_id=camera_id,
            name=name,
            num_points2d=len(point_parts) // 3,
            num_observed_points=observed,
            camera_center=camera_center_from_pose(qvec, tvec),
        )
    return images


def read_points3d_text(path: Path) -> dict[int, Point3D]:
    """Read `points3D.txt`."""

    points: dict[int, Point3D] = {}
    for line in _iter_data_lines(path):
        parts = line.split()
        if len(parts) < 8:
            raise ColmapModelError(f"Malformed points3D.txt line: {line}")
        point_id = int(parts[0])
        track_values = parts[8:]
        if len(track_values) % 2 != 0:
            raise ColmapModelError(f"Malformed track for point {point_id}")
        points[point_id] = Point3D(
            point3d_id=point_id,
            xyz=[float(value) for value in parts[1:4]],
            rgb=[int(value) for value in parts[4:7]],
            error=float(parts[7]),
            track_length=len(track_values) // 2,
        )
    return points


def detect_model_format(model_path: str | Path) -> str:
    """Return `binary` or `text` based on available sparse model files."""

    path = Path(model_path)
    binary_files = [path / "cameras.bin", path / "images.bin", path / "points3D.bin"]
    text_files = [path / "cameras.txt", path / "images.txt", path / "points3D.txt"]
    if all(file.exists() for file in binary_files):
        return "binary"
    if all(file.exists() for file in text_files):
        return "text"
    raise ColmapModelError(
        f"Missing COLMAP sparse model files in {path}. Expected cameras/images/points3D as .bin or .txt."
    )


def read_colmap_model(model_path: str | Path, *, model_format: str = "auto") -> ColmapModel:
    """Read a COLMAP sparse model from a directory."""

    path = Path(model_path)
    if not path.exists() or not path.is_dir():
        raise ColmapModelError(f"Invalid COLMAP sparse model directory: {path}")
    resolved_format = detect_model_format(path) if model_format == "auto" else model_format
    if resolved_format == "binary":
        cameras = read_cameras_binary(path / "cameras.bin")
        images = read_images_binary(path / "images.bin")
        points = read_points3d_binary(path / "points3D.bin")
    elif resolved_format == "text":
        cameras = read_cameras_text(path / "cameras.txt")
        images = read_images_text(path / "images.txt")
        points = read_points3d_text(path / "points3D.txt")
    else:
        raise ColmapModelError("model_format must be 'auto', 'binary', or 'text'.")

    if not cameras:
        raise ColmapModelError("COLMAP model contains zero cameras.")
    if not images:
        raise ColmapModelError("COLMAP model contains zero registered images.")

    return ColmapModel(
        cameras=cameras,
        images=images,
        points3d=points,
        model_path=str(path),
        model_format=resolved_format,
    )


def summarize_model(model: ColmapModel, *, total_images: int | None = None) -> dict:
    """Create JSON-serializable summary statistics for a parsed model."""

    registered_images = len(model.images)
    total = total_images if total_images is not None else registered_images
    observations_from_points = sum(point.track_length for point in model.points3d.values())
    observations_from_images = sum(image.num_observed_points for image in model.images.values())
    camera_models = sorted({camera.model for camera in model.cameras.values()})
    cameras = [asdict(camera) for camera in sorted(model.cameras.values(), key=lambda item: item.camera_id)]
    sample_poses = [
        asdict(image)
        for image in sorted(model.images.values(), key=lambda item: item.image_id)[:10]
    ]
    sample_points = [
        asdict(point)
        for point in sorted(model.points3d.values(), key=lambda item: item.point3d_id)[:10]
    ]
    return {
        "model_path": model.model_path,
        "model_format": model.model_format,
        "number_of_cameras": len(model.cameras),
        "camera_models": camera_models,
        "cameras": cameras,
        "total_images": total,
        "registered_images": registered_images,
        "registration_rate": registered_images / total if total else None,
        "number_of_sparse_points": len(model.points3d),
        "number_of_observations": observations_from_points or observations_from_images,
        "sample_camera_poses": sample_poses,
        "sample_sparse_points": sample_points,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse and summarize a COLMAP sparse model.")
    parser.add_argument("--model-path", required=True, help="Path to sparse model directory, usually sparse/0.")
    parser.add_argument("--format", choices=["auto", "binary", "text"], default="auto")
    parser.add_argument("--total-images", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        model = read_colmap_model(args.model_path, model_format=args.format)
        summary = summarize_model(model, total_images=args.total_images)
    except ColmapModelError as exc:
        print(f"COLMAP model parsing failed: {exc}", file=sys.stderr)
        return 2

    if args.output_json:
        write_json(args.output_json, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
