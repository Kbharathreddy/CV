# 3D Scene Reconstruction from Smartphone Videos Using 3D Gaussian Splatting

Professional, reproducible Computer Vision pipeline for reconstructing a realistic 3D scene from public multi-view datasets and smartphone videos. The repository focuses on data handling, frame selection, preprocessing, COLMAP orchestration, Gaussian Splatting integration, rendering, evaluation, and experiment management.

Results will be added after completion of the corresponding experiment.

## Motivation

Smartphone videos are convenient, but they contain blur, exposure variation, and redundant neighboring frames. This project studies whether intelligent frame selection can reduce reconstruction cost while preserving or improving 3D Gaussian Splatting quality.

## Problem Statement

Given an online dataset scene or a smartphone video of a mostly static scene, build an end-to-end workflow that prepares images, estimates camera poses with COLMAP, trains an established 3D Gaussian Splatting implementation, renders novel views, and evaluates reconstruction quality.

## Objectives

- Support public benchmark data, starting with Mip-NeRF 360 `bonsai`.
- Support smartphone MP4/MOV input in later phases.
- Filter low-quality and redundant frames before SfM.
- Use COLMAP for camera poses and sparse point clouds.
- Integrate an established open-source 3D Gaussian Splatting implementation without rewriting the CUDA rasterizer.
- Record reproducible experiment outputs and metrics without fabricating results.

## System Architecture

```text
Online Dataset / Smartphone Video
        |
Image / Frame Preparation
        |
Frame Quality Filtering
        |
Redundant Frame Removal
        |
Image Preprocessing
        |
COLMAP Structure-from-Motion
        |
Camera Poses + Sparse Point Cloud
        |
3D Gaussian Initialization
        |
3D Gaussian Splatting Optimization
        |
Novel View Rendering
        |
Evaluation + Experiment Comparison
```

## Repository Structure

```text
.
|-- README.md
|-- LICENSE
|-- config.example.yaml
|-- pyproject.toml
|-- requirements.txt
|-- environment.yml
|-- data/
|-- src/
|-- scripts/
|-- tests/
|-- docs/
|-- assets/
|-- outputs/
`-- .github/workflows/
```

## Installation

Use Python 3.10 or newer. Python 3.11 is recommended for the most predictable PyTorch and COLMAP-related ecosystem support.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install PyTorch according to your CUDA version using the official PyTorch selector. This repository does not pin CUDA-specific wheels in `requirements.txt`.

PyCOLMAP is the preferred Phase 3 backend. The COLMAP command-line interface remains supported as a fallback when PyCOLMAP is unavailable or when `colmap.backend: cli` is selected. For CLI use, COLMAP must be installed separately and available as `colmap`, or configured with a full executable path in `config.yaml`.

## Configuration

```bash
cp config.example.yaml config.yaml
```

On Windows PowerShell:

```powershell
Copy-Item config.example.yaml config.yaml
```

Then edit `config.yaml` for local paths such as `colmap_executable` and `gaussian_splatting_path`. The local `config.yaml` file is intentionally ignored by Git.

## Dataset Setup

The first supported public dataset is Mip-NeRF 360, starting with the `bonsai` scene.

```bash
python src/download_dataset.py --dataset mipnerf360 --scene bonsai
```

The official Mip-NeRF 360 project page provides Google Storage archives. The current downloader uses only those official archive URLs and extracts the requested scene into `data/public/mipnerf360/<scene>/`.

## COLMAP Reconstruction

Phase 3 adds COLMAP Structure-from-Motion orchestration for public dataset validation. For the Mip-NeRF 360 `bonsai` scene, use the downsampled `images_4` directory for the first practical smoke reconstruction unless you deliberately want full-resolution feature extraction.

```bash
python src/run_colmap.py \
  --image-dir data/public/mipnerf360/bonsai/images_4 \
  --output-dir outputs/mipnerf_bonsai_colmap \
  --config config.yaml \
  --backend auto \
  --input-type dataset \
  --overwrite
```

`colmap.backend: auto` uses PyCOLMAP when it is importable and falls back to the COLMAP CLI otherwise. The default `auto` matcher chooses exhaustive matching for public dataset image collections and sequential matching for smartphone video frames.

Outputs are written as `database/database.db`, `sparse/0`, `logs/`, and `summary.json` under the experiment directory.

To inspect an existing sparse model:

```bash
python src/convert_colmap.py --model-path outputs/mipnerf_bonsai_colmap/sparse/0
```

## Smartphone Video Capture Guidelines

- Capture a static scene while moving the camera slowly.
- Use stable, even lighting and avoid severe motion blur.
- Maintain strong overlap between neighboring views.
- Circle the object or scene where possible.
- Avoid moving people, reflective surfaces, and transparent objects for early tests.
- Start with a 30-60 second video before scaling up.

## Smartphone Video Preprocessing

Phase 2 implements smartphone-video preparation without invoking COLMAP or Gaussian Splatting. The pipeline validates the video, extracts candidate frames, scores each frame for blur and exposure, removes near-duplicate frames with feature matching, preprocesses selected RGB images, and writes `frame_selection.csv` plus `summary.json`.

Frame extraction supports exactly one strategy at a time:

```bash
python src/extract_frames.py --input data/smartphone/scene01.mp4 --output outputs/test/frames --frame-interval 10
python src/extract_frames.py --input data/smartphone/scene01.mp4 --output outputs/test/frames --time-interval 0.5
python src/extract_frames.py --input data/smartphone/scene01.mp4 --output outputs/test/frames --target-frames 100
```

The automatic selector uses configurable blur, brightness, dark-pixel, and bright-pixel thresholds. Redundancy removal uses ORB features by default, descriptor matching, an optional lightweight geometric check, and a small normalized-motion threshold. These defaults are initial experiment parameters, not experimentally proven optimum values.

Run the Phase 2 smartphone preprocessing pipeline:

```bash
python src/pipeline.py \
  --mode video \
  --input data/smartphone/scene01.mp4 \
  --experiment-name smartphone_scene01 \
  --stop-after preprocessing
```

Outputs are written under:

```text
outputs/<experiment_name>/
|-- frame_selection.csv
|-- summary.json
|-- logs/
`-- frames/
    |-- candidates/
    `-- selected/
```

## Planned Commands

Dataset mode:

```bash
python src/pipeline.py --mode dataset --dataset mipnerf360 --scene bonsai --experiment-name mipnerf_bonsai
```

Smartphone video mode:

```bash
python src/pipeline.py --mode video --input data/smartphone/scene01.mp4 --experiment-name smartphone_scene01
```

The full pipeline entry point is planned for a later phase. Phase 1 provides repository setup, configuration, logging utilities, and dataset acquisition support.

## Evaluation Metrics

- PSNR will measure pixel-level reconstruction fidelity.
- SSIM will measure structural similarity.
- LPIPS will measure perceptual similarity using a learned feature model.

Unavailable metrics will be stored as `null` in later experiment summaries rather than invented.

## Experiments

The target comparison is:

- 30-frame baseline
- 60-frame baseline
- 100-frame baseline
- automatic quality-based frame selection

Results will be added after completion of the corresponding experiment.

## Third-Party Acknowledgements

This project will integrate, rather than reimplement, a maintained 3D Gaussian Splatting implementation. Its repository, license, installation method, and required citation will be documented before integration.

Mip-NeRF 360:

```bibtex
@article{barron2022mipnerf360,
  title={Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields},
  author={Jonathan T. Barron and Ben Mildenhall and Dor Verbin and Pratul P. Srinivasan and Peter Hedman},
  journal={CVPR},
  year={2022}
}
```

## License

This repository's orchestration code is released under the MIT License. External datasets and third-party reconstruction/training implementations retain their own licenses and terms.
