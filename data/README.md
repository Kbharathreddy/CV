# Data Directory

This directory is for local datasets, smartphone videos, extracted frames, and processed images. Dataset files are not committed to Git.

## Layout

```text
data/
|-- public/
|   |-- mipnerf360/
|   `-- tanks_and_temples/
|-- smartphone/
`-- processed/
```

## Mip-NeRF 360

Initial supported scene:

```bash
python src/download_dataset.py --dataset mipnerf360 --scene bonsai
```

The downloader uses official Google Storage archives linked from the Mip-NeRF 360 project page. Official scene-only archives are not currently exposed, so the downloader caches the relevant official archive and extracts only the requested scene.

Expected output:

```text
data/public/mipnerf360/bonsai/
|-- images/
|-- images_2/
|-- images_4/
|-- images_8/
`-- sparse/
```

If automatic download is unavailable in your environment, manually download the required archive and run:

```bash
python src/download_dataset.py --dataset mipnerf360 --scene bonsai --archive-path data/public/mipnerf360/_archives/360_v2.zip
```

## Smartphone Videos

Place personal smartphone videos under:

```text
data/smartphone/
```

Example:

```text
data/smartphone/scene01.mp4
```

Videos, extracted frames, COLMAP outputs, checkpoints, and experiment outputs are ignored by Git.
