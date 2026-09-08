"""Phase 5 held-out-view evaluation metrics.

PSNR and SSIM are CPU-safe and deterministic. LPIPS is supported lazily when
the optional `lpips` package and PyTorch are available in the execution
environment, such as Kaggle. No metric value is fabricated when a dependency is
missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:  # Allows: python src/evaluation.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - numpy is required in normal use
    np = None
    _NUMPY_IMPORT_ERROR = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover - pillow is required in normal use
    Image = None
    _PIL_IMPORT_ERROR = exc
else:
    _PIL_IMPORT_ERROR = None

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - fallback is handled at runtime
    structural_similarity = None

from src.frame_selection import validate_experiment_manifest
from src.utils import ensure_dir, write_json


class EvaluationError(RuntimeError):
    """Raised when Phase 5 metrics cannot be computed."""


@dataclass(frozen=True)
class PerImageMetrics:
    """Metrics for one held-out render/ground-truth pair."""

    image_name: str
    render_name: str
    psnr: float | None
    ssim: float | None
    lpips: float | None


@dataclass(frozen=True)
class EvaluationSummary:
    """Machine-readable Phase 5 evaluation result."""

    experiment: str
    test_image_count: int
    psnr_mean: float | None
    ssim_mean: float | None
    lpips_mean: float | None
    lpips_network: str | None
    per_image_metrics: list[PerImageMetrics]
    training_runtime_seconds: float | None = None
    render_runtime_seconds: float | None = None
    number_of_gaussians: int | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["per_image_metrics"] = [asdict(item) for item in self.per_image_metrics]
        return data


def _require_image_dependencies() -> None:
    if np is None:
        raise EvaluationError("NumPy is required for evaluation.") from _NUMPY_IMPORT_ERROR
    if Image is None:
        raise EvaluationError("Pillow is required for evaluation.") from _PIL_IMPORT_ERROR


def load_rgb_float(path: str | Path):
    """Load an RGB image as float32 in [0, 1]."""

    _require_image_dependencies()
    image_path = Path(path)
    if not image_path.exists():
        raise EvaluationError(f"Image not found: {image_path}")
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.float32) / 255.0


def psnr(reference, prediction, *, max_value: float = 1.0) -> float:
    """Compute PSNR, returning inf for identical images."""

    _require_image_dependencies()
    if reference.shape != prediction.shape:
        raise EvaluationError(f"Image shapes differ: {reference.shape} vs {prediction.shape}")
    mse = float(np.mean((reference - prediction) ** 2))
    if mse == 0.0:
        return math.inf
    return float(20.0 * math.log10(max_value) - 10.0 * math.log10(mse))


def ssim(reference, prediction) -> float:
    """Compute RGB SSIM with a reliable skimage implementation."""

    _require_image_dependencies()
    if structural_similarity is None:
        raise EvaluationError("scikit-image is required for SSIM evaluation.")
    if reference.shape != prediction.shape:
        raise EvaluationError(f"Image shapes differ: {reference.shape} vs {prediction.shape}")
    return float(
        structural_similarity(
            reference,
            prediction,
            channel_axis=-1,
            data_range=1.0,
        )
    )


def _build_lpips_model(network: str = "alex", device: str = "cpu"):
    try:
        import lpips  # type: ignore
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise EvaluationError("LPIPS requires the optional lpips and torch packages.") from exc
    model = lpips.LPIPS(net=network).to(device)
    model.eval()
    return model, torch


def _lpips_tensor(torch_module, array, *, device: str):
    tensor = torch_module.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch_module.float32) * 2.0 - 1.0


def lpips_distance(reference, prediction, *, model, torch_module, device: str = "cpu") -> float:
    """Compute LPIPS distance for one image pair."""

    if reference.shape != prediction.shape:
        raise EvaluationError(f"Image shapes differ: {reference.shape} vs {prediction.shape}")
    with torch_module.no_grad():
        value = model(
            _lpips_tensor(torch_module, reference, device=device),
            _lpips_tensor(torch_module, prediction, device=device),
        )
    return float(value.item())


def _find_render_for_index(render_dir: Path, index: int, image_name: str) -> Path:
    exact = render_dir / image_name
    if exact.exists():
        return exact
    stem = Path(image_name).stem
    for suffix in (".png", ".jpg", ".jpeg"):
        named = render_dir / f"{stem}{suffix}"
        if named.exists():
            return named
    numbered = render_dir / f"{index:05d}.png"
    if numbered.exists():
        return numbered
    raise EvaluationError(f"No render found for held-out image {image_name} in {render_dir}")


def evaluate_heldout_views(
    *,
    ground_truth_dir: str | Path,
    render_dir: str | Path,
    experiment_manifest: dict[str, Any],
    compute_lpips: bool = False,
    lpips_network: str = "alex",
    lpips_device: str = "cpu",
) -> EvaluationSummary:
    """Evaluate renders against the manifest's held-out image list."""

    validate_experiment_manifest(experiment_manifest)
    gt_dir = Path(ground_truth_dir)
    renders = Path(render_dir)
    test_images = list(experiment_manifest["test_images"])
    if not test_images:
        raise EvaluationError("Experiment manifest contains no held-out test images.")

    lpips_model = None
    torch_module = None
    if compute_lpips:
        lpips_model, torch_module = _build_lpips_model(lpips_network, lpips_device)

    per_image: list[PerImageMetrics] = []
    for index, image_name in enumerate(test_images):
        reference = load_rgb_float(gt_dir / image_name)
        render_path = _find_render_for_index(renders, index, image_name)
        prediction = load_rgb_float(render_path)
        image_lpips = None
        if lpips_model is not None and torch_module is not None:
            image_lpips = lpips_distance(
                reference,
                prediction,
                model=lpips_model,
                torch_module=torch_module,
                device=lpips_device,
            )
        per_image.append(
            PerImageMetrics(
                image_name=image_name,
                render_name=render_path.name,
                psnr=psnr(reference, prediction),
                ssim=ssim(reference, prediction),
                lpips=image_lpips,
            )
        )

    return EvaluationSummary(
        experiment=str(experiment_manifest["experiment_name"]),
        test_image_count=len(test_images),
        psnr_mean=_finite_or_inf_mean([item.psnr for item in per_image]),
        ssim_mean=_nullable_mean([item.ssim for item in per_image]),
        lpips_mean=_nullable_mean([item.lpips for item in per_image]),
        lpips_network=lpips_network if compute_lpips else None,
        per_image_metrics=per_image,
    )


def _nullable_mean(values: Sequence[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    if not available:
        return None
    return float(sum(available) / len(available))


def _finite_or_inf_mean(values: Sequence[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    if not available:
        return None
    if any(math.isinf(value) for value in available):
        return math.inf if all(math.isinf(value) for value in available) else None
    return float(sum(available) / len(available))


def write_per_image_csv(path: str | Path, metrics: Sequence[PerImageMetrics]) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["image_name", "render_name", "psnr", "ssim", "lpips"])
        writer.writeheader()
        for item in metrics:
            writer.writerow(asdict(item))


def json_safe_metrics(data: Any) -> Any:
    """Convert non-finite float values to explicit strings for valid JSON."""

    if isinstance(data, float):
        if math.isinf(data):
            return "inf" if data > 0 else "-inf"
        if math.isnan(data):
            return None
        return data
    if isinstance(data, list):
        return [json_safe_metrics(item) for item in data]
    if isinstance(data, dict):
        return {key: json_safe_metrics(value) for key, value in data.items()}
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 5 held-out test-view renders.")
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--per-image-csv", default=None)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--lpips-network", default="alex")
    parser.add_argument("--lpips-device", default="cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        summary = evaluate_heldout_views(
            ground_truth_dir=args.ground_truth_dir,
            render_dir=args.render_dir,
            experiment_manifest=manifest,
            compute_lpips=args.lpips,
            lpips_network=args.lpips_network,
            lpips_device=args.lpips_device,
        )
        summary_data = json_safe_metrics(summary.to_dict())
        write_json(args.output_json, summary_data)
        if args.per_image_csv:
            write_per_image_csv(args.per_image_csv, summary.per_image_metrics)
    except (EvaluationError, OSError, json.JSONDecodeError) as exc:
        print(f"Phase 5 evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary_data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
