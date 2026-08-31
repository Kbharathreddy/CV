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
