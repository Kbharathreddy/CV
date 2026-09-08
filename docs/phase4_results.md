# Phase 4 Results: Kaggle 3DGS GPU Smoke Test

Phase 4 is complete. The goal was to verify 3D Gaussian Splatting integration on
a CUDA GPU using the existing Phase 3 Mip-NeRF 360 `bonsai` COLMAP
reconstruction. This was not a final-quality reconstruction experiment.

## Input

| Field | Value |
| --- | --- |
| Dataset | Mip-NeRF 360 Bonsai |
| Images | 292 |
| Image source | `data/public/mipnerf360/bonsai/images_4` |
| COLMAP source | `outputs/mipnerf_bonsai_colmap/sparse/0` |
| Registered Phase 3 images | 292 / 292 |
| Sparse points used for initialization | 100,730 |

The Kaggle input dataset contained only `bonsai/images` and
`bonsai/sparse/0`. The sparse model came from our Phase 3 output, not the
dataset-provided reference reconstruction.

## Execution Environment

| Field | Value |
| --- | --- |
| Backend | Kaggle |
| GPU | Tesla T4 |
| GPU VRAM | 15360 MB |
| NVIDIA driver | 580.159.04 |
| PyTorch | 2.10.0+cu128 |
| Torch CUDA | 12.8 |
| Python | 3.12.13 |
| Git | 2.34.1 |

The notebook first copied the read-only Kaggle input scene to the writable path
`/kaggle/working/bonsai_scene`. Both `train.py` and `render.py` used that
writable scene path.

## 3DGS Implementation

| Field | Value |
| --- | --- |
| Repository | `https://github.com/graphdeco-inria/gaussian-splatting.git` |
| Commit | `54c035f7834b564019656c3e3fcc3646292f727d` |
| License | Non-commercial research/evaluation license from Inria and MPII |

The external implementation was cloned in Kaggle and was not vendored into this
repository.

## Smoke-Test Output

| Field | Value |
| --- | --- |
| Iterations | 300 |
| Initial loss | 0.2694682 |
| Final loss | 0.1096482 |
| Loss changed | true |
| Number of Gaussians | 100,730 |
| Checkpoint | `/kaggle/working/outputs/kaggle/bonsai_smoke/checkpoints/chkpnt300.pth` |
| Point cloud | `/kaggle/working/outputs/kaggle/bonsai_smoke/point_cloud/iteration_300/point_cloud.ply` |
| Rendered images | 292 |
| Example render | `/kaggle/working/outputs/kaggle/bonsai_smoke/renders/00000.png` |
| Summary | `/kaggle/working/outputs/kaggle/bonsai_smoke/summary.json` |
| Status | SUCCESS |

The downloaded notebook records `training_runtime_seconds = 51.79` and
`render_runtime_seconds = 109.96`. These are measured Kaggle notebook values and
may vary on reruns depending on the assigned GPU and notebook environment.

## Next Phase

Phase 5 should run full reconstruction experiments and evaluation. It should
compare 30-frame, 60-frame, 100-frame, automatic frame-selection, and optional
full 292-frame baselines. Evaluation should use held-out evaluation views rather
than reporting metrics only on training views.
