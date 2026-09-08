# Phase 5 Experiment Design

Phase 5 prepares the full reconstruction and evaluation experiments. It does
not run GPU training locally and does not report final reconstruction metrics
yet.

## Research Question

How does intelligent frame selection affect 3D Gaussian Splatting
reconstruction quality and computational efficiency?

## Variables

Independent variable:

- Frame-selection strategy and frame count.

Dependent measurements:

- PSNR on held-out test views.
- SSIM on held-out test views.
- LPIPS on held-out test views.
- Training time.
- Render time.
- Number of Gaussians.
- Reconstruction/training success.

Controlled variables:

- Same scene: Mip-NeRF 360 `bonsai`.
- Same held-out test-image filenames for every experiment.
- Same Phase 3 COLMAP sparse reconstruction as the source of poses.
- Same pinned Graphdeco/Inria 3DGS implementation.
- Same general training and evaluation procedure.

## Common Train/Test Split

The committed split lives at:

```text
experiments/phase5/split_manifest.json
```

The split is deterministic. Images are ordered by the registered COLMAP image
names from `outputs/mipnerf_bonsai_colmap/sparse/0`. Every 8th image starting
at offset 0 is held out for testing.

For the 292-image bonsai scene this creates:

- Total images: 292
- Held-out test views: 37
- Training candidates: 255

The held-out test images are fixed once in the split manifest and copied into
every experiment manifest. The 30/60/100 budgets refer to training views only,
not training plus test views.

The same held-out views are required because PSNR, SSIM, and LPIPS are only
comparable if every method is judged on the same camera viewpoints. If one
method is evaluated on easier or closer views than another, the metric
comparison is scientifically unfair.

## Fixed-Budget Selection

The fixed experiments are:

- `fixed_30`
- `fixed_60`
- `fixed_100`

They do not take the first N images. Each fixed selector uniformly samples from
the ordered training-candidate list so the selected frames cover the camera
trajectory. The first and last training candidates are included when the budget
is greater than one.

## Automatic Selection

The prepared automatic experiment is:

- `automatic_60`

It is designed for a matched-budget comparison against `fixed_60`.

The automatic selector is deterministic and explainable. It combines:

- Phase 2-style image quality signals: blur, exposure, and feature count.
- COLMAP pose coverage: camera centers and viewing directions from the Phase 3
  sparse model.
- Greedy diversity selection: after filtering invalid or very low-quality
  candidates when enough are available, it anchors trajectory endpoints and
  repeatedly adds the candidate with the best weighted coverage/quality score.

The default weights are:

- Pose diversity: 0.65
- Image quality: 0.25
- Feature richness: 0.10

The real `automatic_60` manifest was generated with local OpenCV quality
analysis available. Its selection details record that 255 pose signals and 255
quality signals were considered, and that 206 candidates passed the quality
filter before the 60-frame subset was chosen.

The same code can generate `automatic_30` or `automatic_100` later by changing
the budget.

## Reusing COLMAP Without Rerunning SfM

Phase 5 does not recompute features, matches, or mapping. It reads the existing
Phase 3 sparse model:

```text
outputs/mipnerf_bonsai_colmap/sparse/0
```

For each experiment, scene preparation filters the existing COLMAP cameras,
registered images, sparse points, 2D observations, and tracks to match the
selected image list. It then writes a new binary COLMAP model under:

```text
outputs/phase5/<experiment>/train_scene/sparse/0
outputs/phase5/<experiment>/test_scene/sparse/0
```

The train scene contains only training images. The test scene contains only
held-out images. Camera intrinsics and poses are preserved from the Phase 3
model; they are not fabricated or approximated.

## Explicit 3DGS Train/Test Enforcement

The official Graphdeco implementation has an implicit evaluation split for
COLMAP scenes. Phase 5 does not rely on that split because subset scenes would
otherwise produce different test images.

Instead, Phase 5 uses manifest-controlled scenes:

```text
train_scene/
  images/
  sparse/0/

test_scene/
  images/
  sparse/0/
```

The Kaggle notebook trains with `train_scene` only. It later renders
`test_scene` from the trained checkpoint. This keeps the shared held-out views
out of optimization while still using their Phase 3 camera poses for rendering.

## Evaluation

`scripts/evaluate_phase5.py` evaluates only the manifest's held-out
`test_images`.

Prepared metrics:

- PSNR
- SSIM
- LPIPS with the configurable `alex` network by default

PSNR and SSIM use CPU-safe established implementations. LPIPS is loaded lazily
and should be enabled in the Kaggle environment where the model/package is
available. Missing LPIPS dependencies should produce an error rather than a
fabricated value when LPIPS is requested.

Per-image metrics can be written to CSV, and experiment summaries use the
standard schema in `src/phase5_results.py`.

## Result Aggregation

`scripts/compare_phase5_results.py` combines completed `summary.json` files
into a comparison CSV. It should only aggregate real completed result files.
Unrun experiments should remain absent or have explicit `not_run` placeholders
with `null` metrics.

Target table:

```text
Experiment | Train Frames | PSNR | SSIM | LPIPS | Train Time | Gaussians
fixed_30
fixed_60
fixed_100
automatic_60
full
```

## Current Status

Phase 5 preparation is complete once the local code, manifests, notebook,
tests, and documentation are committed. No Phase 5 GPU training, held-out
rendering, PSNR, SSIM, or LPIPS results have been obtained yet.

Phase 4's 300-iteration Kaggle run remains only an integration smoke test and
must not be mixed with Phase 5 final experiment results.
