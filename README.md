# 3D Scene Reconstruction from Smartphone Videos Using 3D Gaussian Splatting

Professional, reproducible Computer Vision pipeline for reconstructing a realistic 3D scene from public multi-view datasets and smartphone videos. The repository focuses on data handling, frame selection, preprocessing, COLMAP orchestration, Gaussian Splatting integration, rendering, evaluation, and experiment management.

## Project Status

| Phase | Status | Notes |
| --- | --- | --- |
| Phase 1 - Project structure | Complete | Repository, configuration, logging, and dataset acquisition scaffolding. |
| Phase 2 - Smartphone video preprocessing | Complete | Frame extraction, quality filtering, redundancy filtering, preprocessing, CSV, and summary output. |
| Phase 3 - COLMAP/SfM reconstruction | Complete | Mip-NeRF 360 `bonsai` `images_4` registered `292/292` images with PyCOLMAP. |
| Phase 4 - Kaggle 3DGS GPU smoke test | Complete | Official GraphDeco 3DGS ran `300` iterations on Kaggle Tesla T4 and rendered `292` training views. |
| Phase 5 - Full reconstruction experiments and evaluation | Preparation | Local manifests, split enforcement, scene-prep tooling, metrics, and Kaggle runner are prepared; GPU experiments are not run yet. |

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

## Kaggle 3DGS Smoke Test

Phase 4 was smoke-tested on Kaggle using only the verified Phase 3 bonsai inputs. To reproduce the private Kaggle input bundle:

```bash
python scripts/prepare_kaggle_input.py \
  --images-dir data/public/mipnerf360/bonsai/images_4 \
  --sparse-dir outputs/mipnerf_bonsai_colmap/sparse/0 \
  --output-dir outputs/kaggle/bonsai_3dgs_input \
  --archive-path outputs/kaggle/bonsai_3dgs_input.zip \
  --overwrite
```

Upload the zip as a private Kaggle dataset, attach it to a GPU notebook, and run `notebooks/phase4_kaggle.ipynb`. The notebook clones the official GraphDeco/Inria 3D Gaussian Splatting implementation at the pinned commit recorded in `config.example.yaml`, verifies the non-commercial research/evaluation license, copies the read-only Kaggle input scene to `/kaggle/working/bonsai_scene`, builds only the required CUDA extensions, runs a short smoke test, writes a checkpoint, renders all 292 training views, displays one render, and records `summary.json`.

Do not use this step for full 30,000-iteration training, evaluation metrics, smartphone reconstruction, or 30/60/100-frame experiments.

Verified smoke-test result: Kaggle Tesla T4, PyTorch `2.10.0+cu128`, CUDA `12.8`, `300` iterations, loss changed from about `0.2694682` to `0.1096482`, `100,730` Gaussians, `292` renders, status `SUCCESS`. This is an integration smoke test only, not a final-quality reconstruction. See `docs/phase4_results.md`.

## Phase 5 Experiment Preparation

Phase 5 studies how frame-selection strategy affects 3D Gaussian Splatting quality and cost on Mip-NeRF 360 `bonsai`. The prepared experiments are:

- `fixed_30`
- `fixed_60`
- `fixed_100`
- `automatic_60`
- `full` optional baseline using all non-held-out training candidates

All experiments share the same explicit held-out test views. The committed split manifest is `experiments/phase5/split_manifest.json`: from 292 ordered COLMAP images, every 8th image starting at offset 0 is held out, producing 37 test images and 255 training candidates. The 30/60/100 counts refer only to training images.

Generate or refresh the small manifests:

```bash
python scripts/prepare_phase5_experiment.py \
  --generate-manifests \
  --include-full \
  --manifest-dir experiments/phase5 \
  --source-images data/public/mipnerf360/bonsai/images_4 \
  --source-model outputs/mipnerf_bonsai_colmap/sparse/0
```

Prepare one experiment scene later without rerunning COLMAP:

```bash
python scripts/prepare_phase5_experiment.py \
  --experiment fixed_30 \
  --manifest-dir experiments/phase5 \
  --source-images data/public/mipnerf360/bonsai/images_4 \
  --source-model outputs/mipnerf_bonsai_colmap/sparse/0 \
  --output outputs/phase5/fixed_30 \
  --link-mode hardlink
```

For Kaggle GPU execution, use `notebooks/phase5_kaggle.ipynb` and change only `EXPERIMENT_NAME` for each run. The notebook trains from a train-only scene and renders a separate held-out scene, so held-out images are not used for training. No Phase 5 PSNR, SSIM, or LPIPS values have been obtained yet.

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

The Phase 5 target comparison is:

- `fixed_30`
- `fixed_60`
- `fixed_100`
- `automatic_60`
- optional `full`

Results will be added after completion of the corresponding experiment.

Phase 5 evaluation must use the shared held-out evaluation views rather than reporting metrics only on training views.

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
