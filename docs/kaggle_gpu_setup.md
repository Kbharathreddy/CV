# Kaggle GPU Setup for Phase 4

Phase 4 used Kaggle as an alternate GPU backend while AWS G-family quota was
pending. Kaggle was used only for a small 3D Gaussian Splatting smoke test. Do
not rerun COLMAP and do not use the dataset-provided reference `sparse/0`.

## Inputs

Use only the data required for the bonsai smoke test:

```text
bonsai/
  images/
  sparse/
    0/
      cameras.bin
      images.bin
      points3D.bin
```

The source directories on this machine are:

```text
data/public/mipnerf360/bonsai/images_4
outputs/mipnerf_bonsai_colmap/sparse/0
```

The second path is our Phase 3 reconstruction. It must not be replaced by:

```text
data/public/mipnerf360/bonsai/sparse/0
```

Prepare the upload bundle locally:

```powershell
python scripts/prepare_kaggle_input.py `
  --images-dir data/public/mipnerf360/bonsai/images_4 `
  --sparse-dir outputs/mipnerf_bonsai_colmap/sparse/0 `
  --output-dir outputs/kaggle/bonsai_3dgs_input `
  --archive-path outputs/kaggle/bonsai_3dgs_input.zip `
  --overwrite
```

Upload `outputs/kaggle/bonsai_3dgs_input.zip` to Kaggle as a private dataset.
The completed Kaggle run used this mounted input path:

```text
/kaggle/input/datasets/kopperlabharathreddy/mip-nerf-360-bonsai-our-colmap-reconstruction/bonsai
```

If your Kaggle dataset slug is different, edit `SOURCE_PATH` in the first
configuration cell of `notebooks/phase4_kaggle.ipynb`.

Because Kaggle input datasets are read-only, the notebook copies the scene to:

```text
/kaggle/working/bonsai_scene
```

Both `train.py` and `render.py` use that writable scene path.

## Implementation Choice

Use the official GraphDeco/Inria implementation:

```text
Repository: https://github.com/graphdeco-inria/gaussian-splatting.git
Pinned commit: 54c035f7834b564019656c3e3fcc3646292f727d
License: non-commercial research/evaluation license from Inria and MPII
```

The implementation is cloned in the Kaggle working directory with submodules.
No external 3DGS source code is copied into this project repository.

The official optimizer expects a COLMAP-style scene containing `images` and
`sparse/0`. Our Kaggle input bundle uses exactly that layout.

## Kaggle Notebook Setup

Create a Kaggle notebook with:

- Accelerator: GPU.
- Internet: enabled, so the official implementation and submodules can be
  cloned from GitHub.
- Input dataset: the private bonsai input bundle created above.

Then upload or copy the cells from:

```text
notebooks/phase4_kaggle.ipynb
```

The notebook stages are:

1. GPU, CUDA, PyTorch, Python, and Git checks.
2. Input dataset validation.
3. Official 3DGS clone, commit checkout, submodule update, and license check.
4. Required CUDA extension installation.
5. Short training smoke test, default `300` iterations.
6. Render command and render-file verification.
7. Summary JSON writing.

## Smoke-Test Success Criteria

The smoke test is considered successful only if:

- `torch.cuda.is_available()` is true.
- `nvidia-smi` reports a CUDA-capable GPU.
- The Kaggle input contains 292 bonsai images.
- The COLMAP files load from our `sparse/0`.
- The official 3DGS training command starts on CUDA.
- Parsed loss values are finite and change during the short run.
- A checkpoint is written.
- A point cloud is written.
- At least one render image is produced.
- `summary.json` records measured runtime and environment metadata.

Verified Kaggle output root:

```text
/kaggle/working/outputs/kaggle/bonsai_smoke/
  checkpoints/
  renders/
  logs/
  summary.json
```

The training interface should remain independent of the GPU provider. AWS and
Kaggle runs should differ only in their execution environment and configured
paths.

## Verified Smoke-Test Result

The downloaded Kaggle notebook records a successful Phase 4 smoke test:

| Field | Value |
| --- | --- |
| Backend | Kaggle |
| Dataset | Mip-NeRF 360 Bonsai |
| Images | 292 |
| GPU | Tesla T4 |
| GPU VRAM | 15360 MB |
| NVIDIA driver | 580.159.04 |
| PyTorch | 2.10.0+cu128 |
| Torch CUDA | 12.8 |
| 3DGS implementation | graphdeco-inria/gaussian-splatting |
| Implementation commit | 54c035f7834b564019656c3e3fcc3646292f727d |
| Iterations | 300 |
| Initial loss | 0.2694682 |
| Final loss | 0.1096482 |
| Number of Gaussians | 100730 |
| Rendered images | 292 |
| Status | SUCCESS |

The 300-iteration run is only a GPU/integration smoke test, not a final-quality
reconstruction.

## Do Not Run Yet For Later Experiments

Do not use the Kaggle notebook for:

- Full 30,000-iteration training.
- PSNR, SSIM, or LPIPS evaluation.
- Smartphone reconstruction.
- 30-frame, 60-frame, or 100-frame frame-selection experiments.

Stop after the smoke test and record the generated `summary.json`.
