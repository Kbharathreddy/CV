# Methodology

## Phase 2: Smartphone Video Preprocessing

Phase 2 prepares smartphone video frames for later Structure-from-Motion and 3D Gaussian Splatting. It does not run COLMAP, train a model, render views, or report reconstruction metrics.

```text
Smartphone Video
        |
Video Validation
        |
Candidate Frame Extraction
        |
Frame Quality Analysis
        |
Blur / Exposure Filtering
        |
Feature-Based Redundancy Removal
        |
Preprocessing
        |
Final Selected RGB Images
        |
frame_selection.csv
```

## Video Validation

The pipeline opens the MP4/MOV file with OpenCV and validates that the file exists, can be decoded, has positive FPS, has a positive frame count, and reports positive width and height. These metadata values are written into `summary.json`.

## Candidate Frame Extraction

Only one sampling strategy is active at a time:

- `frame_interval`: keep every Nth source frame.
- `time_interval`: keep frames separated by approximately N seconds.
- `target_frames`: sample approximately N frames evenly across the video duration.

Extracted candidates are written chronologically with deterministic names such as `frame_000001.jpg`. For every candidate the pipeline records the source video frame index, candidate frame ID, timestamp, filename, and file path.

## Quality Analysis

Each candidate image is analyzed independently before redundancy filtering.

Blur is measured with variance of Laplacian on the grayscale image. Lower values indicate less high-frequency detail and usually more blur.

Exposure is measured with grayscale mean brightness plus dark and bright pixel ratios. The intensity cutoffs and rejection thresholds are configuration values, not fixed claims about the best settings for every scene.

Feature content is measured with ORB by default. SIFT can be requested when the installed OpenCV build provides it, but ORB remains the default because it works with standard `opencv-python`.

## Quality Filtering

The quality gate rejects candidates with these reasons:

- `invalid_image`: image cannot be read or decoded.
- `blur`: Laplacian variance is below the configured blur threshold.
- `underexposed`: mean brightness is too low or too many pixels are darker than the configured cutoff.
- `overexposed`: mean brightness is too high or too many pixels are brighter than the configured cutoff.
- `selected`: the frame passed the independent quality checks.

These labels are written directly to `frame_selection.csv` so rejection behavior is auditable.

## Redundancy Removal

Redundancy filtering is feature-based rather than raw-pixel-based. Each quality-valid candidate is compared against the most recently retained frame.

The baseline algorithm is:

1. Detect ORB keypoints and descriptors in both frames.
2. Match descriptors with a brute-force matcher.
3. Apply Lowe's ratio test, with an ORB cross-check fallback for near-identical frames.
4. Optionally estimate a fundamental matrix, falling back to a homography, and count RANSAC inliers.
5. Estimate median matched-keypoint motion normalized by the image diagonal.
6. Reject the later frame as `redundant` only when reliable match overlap is high and geometric motion is very small.

This is intentionally conservative for SfM. Moderate viewpoint change with good feature overlap is retained, because neighboring images must overlap for robust camera registration. Extreme mismatch or insufficient features is not treated as proof of duplication, so the frame is kept if it passed quality checks.

## Preprocessing

Selected frames are opened with Pillow, corrected with EXIF orientation metadata when present, converted to RGB, resized to fit within the configured maximum width and height while preserving aspect ratio, and saved as JPEG. The preprocessing stage avoids histogram equalization, sharpening, denoising, and color changes because later photometric reconstruction benefits from consistent image appearance.

## Outputs

Each Phase 2 run creates:

```text
outputs/<experiment_name>/
|-- frame_selection.csv
|-- summary.json
|-- logs/
`-- frames/
    |-- candidates/
    `-- selected/
```

`frame_selection.csv` records per-frame quality and redundancy information. `summary.json` records video metadata, rejection counts, retained frame count, processing time, timestamp, and the runtime configuration used for the experiment.

## Phase 3: COLMAP Structure-from-Motion

Phase 3 validates camera pose estimation on the public Mip-NeRF 360 `bonsai` scene before using smartphone videos. Public multi-view images are not necessarily chronological video frames, so the pipeline selects exhaustive matching for dataset mode by default and reserves sequential matching for video mode.

PyCOLMAP is the preferred backend because it lets the project run COLMAP's reconstruction stages directly from Python and collect consistent metadata. The COLMAP command-line backend remains available as a fallback. With `colmap.backend: auto`, the runner uses PyCOLMAP when importable and otherwise requires a configured COLMAP CLI executable.

The COLMAP workflow is:

```text
images
        |
feature_extractor
        |
exhaustive_matcher or sequential_matcher
        |
mapper
        |
sparse/0
        |
cameras + image poses + sparse points
```

The runner validates the selected backend before reconstruction. Each major reconstruction stage writes logs under the experiment's `logs/` directory, records elapsed time, and stops immediately if a critical stage fails. Generated Phase 3 experiments are written under `outputs/<experiment_name>/` with `database/database.db`, `sparse/0`, `logs/`, and `summary.json`, and must not overwrite dataset-provided reference models such as `data/public/mipnerf360/bonsai/sparse/0`.

The parser in `src/convert_colmap.py` reads binary or text sparse models. It extracts camera intrinsics, registered image rotations and translations, camera centers, sparse point coordinates, colors, reprojection errors, and track lengths. These values are summarized in `outputs/<experiment_name>/summary.json` so later Gaussian Splatting integration can consume verified camera and sparse point metadata rather than assuming reconstruction succeeded.

## Phase 4: Gaussian Splatting GPU Smoke Test

Phase 4 is complete. It used a short GPU smoke test rather than full optimization. The training backend should be portable across AWS, Kaggle, and other CUDA machines, so project code prepares a standard COLMAP scene layout and command configuration while the heavy 3DGS implementation remains an external dependency.

While AWS G-family quota is pending, Kaggle can be used as an alternate validation backend. The required Kaggle input contains only:

```text
bonsai/
        images/
        sparse/
                0/
```

The `images` directory comes from `data/public/mipnerf360/bonsai/images_4`. The `sparse/0` directory must come from our generated Phase 3 reconstruction at `outputs/mipnerf_bonsai_colmap/sparse/0`, not from the Mip-NeRF download's reference `sparse/0`.

On Kaggle, the read-only input scene was copied to `/kaggle/working/bonsai_scene`. Both official 3DGS commands used that writable path as their scene source.

The selected external implementation is the official GraphDeco/Inria repository, pinned by commit in `config.example.yaml`. Its license is suitable for non-commercial research/evaluation use; external source is cloned in the GPU environment and is not copied into this repository.

The smoke-test sequence is:

```text
CUDA/GPU check
        |
input COLMAP scene validation
        |
official 3DGS clone + license/commit check
        |
CUDA extension build
        |
short training run, default 300 iterations
        |
checkpoint + point cloud check
        |
render command
        |
summary.json
```

Smoke-test success means CUDA is available, the COLMAP model and images load, Gaussian initialization succeeds, training starts, finite loss values are observed and change, a checkpoint is written, and at least one render is produced. The notebook records measured values only; it must not fabricate runtime, loss, GPU memory, or Gaussian counts.

The verified Kaggle run used a Tesla T4 with PyTorch `2.10.0+cu128`, ran 300 iterations, changed the main training loss from about `0.2694682` to `0.1096482`, wrote `chkpnt300.pth`, saved a `100,730`-Gaussian point cloud, and rendered 292 training views. The loss parser intentionally ignores `Depth Loss=0.0000000` so the reported final loss reflects the main training loss. These results are a smoke-test validation only, not final reconstruction quality evidence.

## Phase 5: Full Experiments and Evaluation Preparation

Phase 5 preparation defines the scientific experiment design before running expensive GPU jobs. It compares `fixed_30`, `fixed_60`, `fixed_100`, `automatic_60`, and an optional `full` baseline. No Phase 5 training or evaluation metrics have been run yet.

All Phase 5 experiments use the same explicit held-out test views. The split is generated from the ordered COLMAP image list in the Phase 3 sparse model. The default policy holds out every 8th registered image starting at offset 0, which produces 37 held-out test views from the 292-image bonsai scene. The remaining 255 images are training candidates.

Fixed-budget selectors uniformly sample the ordered training candidates instead of taking the first N images. This gives the 30, 60, and 100-frame baselines coverage across the camera trajectory.

The automatic selector is implemented as a first-class reusable component in `src/frame_selection.py`. It uses image-quality signals from the Phase 2 quality logic, feature counts, and Phase 3 camera-pose coverage. It filters clearly poor candidates when enough good candidates remain, anchors the first and last usable views, and greedily adds views that improve pose diversity while retaining quality and feature-richness. The prepared `automatic_60` manifest records computed quality and pose-signal counts.

Phase 5 scene preparation reuses the existing Phase 3 COLMAP model without rerunning feature extraction, matching, or mapping. `src/phase5_experiments.py` reads the COLMAP cameras, image poses, sparse points, 2D observations, and tracks, filters them to the manifest image lists, and writes train/test COLMAP scene folders:

```text
outputs/phase5/<experiment>/train_scene/
outputs/phase5/<experiment>/test_scene/
```

The train scene contains only training views. The test scene contains only held-out views. The Kaggle notebook trains Graphdeco 3DGS on `train_scene` and then renders `test_scene` from the trained checkpoint. This avoids relying on Graphdeco's implicit COLMAP evaluation split, because implicit splitting would change when the image subset changes.

Evaluation metrics must be computed on held-out views. `scripts/evaluate_phase5.py` prepares PSNR, SSIM, and optional LPIPS evaluation against the manifest's `test_images`, and `scripts/compare_phase5_results.py` aggregates completed experiment summaries into CSV. Training-view renders can be useful for sanity checks, but they are not sufficient evidence of generalization or final reconstruction quality.
