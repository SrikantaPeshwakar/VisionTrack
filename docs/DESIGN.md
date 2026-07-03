# VisionTrack — Design Document

This document explains the key architectural decisions made during the
design of VisionTrack, the trade-offs considered, and how the system can
be extended in the future.

---

## 1. Why YOLO?

### Decision
Use the Ultralytics YOLO family (v8/v9/v10/v11) as the sole detection
backbone.

### Alternatives Considered

| Detector          | Pros                                      | Cons                                           |
|-------------------|-------------------------------------------|------------------------------------------------|
| YOLOv8–v11        | Unified API, fast, accurate, active OSS   | Single-stage, less accurate on tiny objects    |
| RT-DETR           | Transformer-based, strong on dense scenes | Slower, larger memory footprint                |
| Faster R-CNN      | High accuracy, established baseline       | Two-stage, ~5× slower than YOLO                |
| Grounding DINO    | Open-vocabulary detection                 | Extremely slow, not suitable for real-time     |
| SSD / MobileNet   | Very fast, edge-friendly                  | Much lower accuracy on partially occluded persons |

### Rationale

- **Ultralytics ecosystem** provides a single API across all YOLO versions —
  switching from `yolov8n` to `yolo11m` requires only a config change.
- **Person class filtering** (COCO class 0) is built-in, avoiding unnecessary
  post-processing overhead.
- **Model size flexibility** lets the same codebase run on a laptop CPU
  (`yolov8n`, ~14 FPS) or a server GPU (`yolov8x`, 60+ FPS).
- **Active development** — Ultralytics releases new variants regularly;
  the `SUPPORTED_MODELS` list in `constants.py` can be extended without
  touching any pipeline code.

---

## 2. Why BoT-SORT?

### Decision
Use Ultralytics' built-in BoT-SORT tracker (`model.track(..., tracker="botsort.yaml")`).

### Alternatives Considered

| Tracker       | Strengths                                      | Weaknesses                                          |
|---------------|------------------------------------------------|-----------------------------------------------------|
| **BoT-SORT**  | Kalman + ReID + camera motion compensation     | Slightly slower than ByteTrack                      |
| ByteTrack     | Very fast, strong on crowded scenes            | No appearance features — more ID switches on re-entry|
| DeepSORT      | Pioneered appearance-based re-ID               | Slower, older codebase, harder to tune              |
| OC-SORT       | Observation-centric, handles occlusion well    | Less mature ecosystem integration                   |
| StrongSORT    | High accuracy on benchmarks                    | Significant overhead, slower than BoT-SORT          |

### Rationale

- **Identity persistence** across occlusions is the primary requirement.
  BoT-SORT's combination of Kalman motion prediction and ReID appearance
  embeddings handles the re-identification problem that pure IoU-based
  trackers (ByteTrack) miss.
- **Camera motion compensation** (GMC via sparse optical flow) handles
  handheld or panning cameras, which is common in real deployments.
- **Built-in Ultralytics integration** means `model.track(..., persist=True)`
  handles the entire tracker lifecycle — no custom Kalman filter or
  association matrix implementation required.
- **No reimplementation** — the assignment calls for using BoT-SORT, not
  building it. The `Tracker` class wraps the Ultralytics interface cleanly
  and adds track history management on top.

### BoT-SORT vs ByteTrack Trade-off

```
BoT-SORT  →  more accurate identity persistence, ~10% slower
ByteTrack →  faster, fewer computations, more ID switches on occlusion
```

For crowd analytics where unique visitor count accuracy matters more than
raw throughput, BoT-SORT is the correct default. For pure speed-critical
deployments, `config/bytetrack.yaml` can be swapped in via `tracker.config_file`.

---

## 3. Model Size Trade-offs

### Benchmark Results

Measured on Apple M-series (MPS), `sample.mp4` (361 frames, 480×848, 24 FPS):

| Model    | Avg FPS | Inf (ms) | Unique Visitors | Peak Concurrent | Speedup vs n |
|----------|---------|----------|-----------------|-----------------|--------------|
| yolov8n  | 14.9    | 66 ms    | 34              | 7               | 1.0×         |
| yolov8s  | 11.8    | 84 ms    | 38              | 8               | 0.79×        |
| yolov8m  | 9.8     | 101 ms   | 44              | 8               | 0.66×        |

**Key observations:**

1. **Accuracy scales with model size** — yolov8m found 44 unique persons vs
   34 for yolov8n, a 29% improvement in detection coverage on the same video.
   Larger models detect more partial/occluded persons that smaller models miss.

2. **FPS cost is approximately linear** — each step up the model ladder
   costs ~25–35% throughput. Going from n → m halves the FPS.

3. **Peak concurrent tracks** increase with larger models because more
   simultaneous partial occlusions are resolved correctly.

4. **Practical recommendation:**
   - `yolov8n` — real-time on CPU or MPS; acceptable for well-lit, low-density scenes
   - `yolov8s` — best balance for most deployments on Apple Silicon / entry GPU
   - `yolov8m` — recommended for GPU deployments where accuracy matters
   - `yolov8l/x` — research or forensic use; not suitable for real-time

### GPU vs CPU Comparison (approximate, based on community benchmarks)

| Device              | yolov8n FPS | yolov8m FPS |
|---------------------|-------------|-------------|
| Apple M2 (MPS)      | ~15         | ~10         |
| NVIDIA RTX 3080     | ~80         | ~45         |
| NVIDIA RTX 4090     | ~130        | ~75         |
| Intel i9 CPU only   | ~8          | ~3          |

---

## 4. Configuration-Driven Architecture

### Decision
All parameters (model type, thresholds, tracker config, output flags) live
in `config/config.yaml`. Nothing is hardcoded.

### Rationale

- **Zero re-deployment friction** — changing confidence threshold or model
  size for a new venue requires only a YAML edit, no code change.
- **CLI override chain** — `config.yaml` → environment variables →
  `--flag` arguments. Each layer overrides the previous, giving maximum
  flexibility without breaking defaults.
- **Config snapshot in exports** — every JSON output file embeds the exact
  config used for that run, making results reproducible.
- **Single source of truth** — `ConfigManager` validates all fields at load
  time, surfacing bad values immediately rather than failing mid-run.

---

## 5. Dependency Injection in VideoPipeline

### Decision
`VideoPipeline` receives all components (Detector, Tracker, Analytics,
Visualizer, Exporter) via its constructor rather than constructing them
internally.

### Rationale

- **Testability** — every component can be mocked independently. The
  362-test suite runs in under 1 second with no GPU, no model weights,
  and no video files.
- **Swappability** — replacing the detector with a future RT-DETR wrapper
  requires no changes to `pipeline.py`, only to the component itself.
- **Separation of concerns** — the pipeline orchestrates; it doesn't own
  the logic of any component.

---

## 6. fuse_score Clamping

### Note on BoT-SORT Confidence Values

BoT-SORT with `fuse_score: True` multiplies the detection confidence score
by the IoU overlap score during association. This can produce fused scores
above 1.0 (e.g., `conf=0.95 × IoU=1.0 → 0.95`, but edge cases produce
values like `2.0` on MPS).

`Tracker._parse_results()` and `Detector._parse_results()` both clamp the
raw float to `[0.0, 1.0]` before constructing `Track` or `Detection`
dataclasses. This is safe because the fused score is only used internally
by BoT-SORT for association; the stored value is a confidence indicator,
not an exact probability.

---

## 7. MOT17 Dataset Integration

### Recommended Test Sequences

The [MOT17 benchmark](https://motchallenge.net/data/MOT17/) provides
standardised sequences for evaluating multi-object trackers.

| Sequence   | Scene                          | Frames | Challenge                              |
|------------|-------------------------------|--------|----------------------------------------|
| MOT17-04   | Train station (static camera)  | 1050   | Dense crowds, heavy occlusion          |
| MOT17-09   | Pedestrian crossing (static)   | 525    | Crisscrossing paths, variable lighting |
| MOT17-11   | Outdoor scene (moving camera)  | 900    | Camera motion, moderate density        |

**To test with MOT17:**

```bash
# 1. Register and download from https://motchallenge.net/data/MOT17/
# 2. Extract to ~/datasets/MOT17/

# 3. Convert image sequences to MP4
python scripts/download_mot.py \
    --mot-root ~/datasets/MOT17 \
    --sequences MOT17-04 MOT17-09 \
    --output sample_videos/

# 4. Run evaluation
python evaluate.py \
    --video sample_videos/MOT17-04.mp4 \
    --models yolov8n yolov8s yolov8m \
    --device mps
```

### Expected Challenges on MOT17

- **MOT17-04** (train station): Extremely dense crowds will push ID switch
  rates higher on smaller models. Expect yolov8m to track 20–30% more
  persons per frame than yolov8n.
- **MOT17-09** (pedestrian crossing): Crisscrossing paths test the IoU
  association; increase `track_buffer` to 45+ in `config/botsort.yaml`
  for better occlusion recovery.
- **MOT17-11** (moving camera): GMC (`gmc_method: sparseOptFlow`) in
  `botsort.yaml` is critical here. Disabling it will cause significant
  ID drift.

---

## 8. Future Roadmap

The following enhancements are explicitly out of scope for this submission
but are straightforward extensions of the current architecture:

### Near-term
- **RTSP / webcam input** — replace `cv2.VideoCapture(path)` with
  `cv2.VideoCapture(rtsp_url)` or `cv2.VideoCapture(0)`. Add a
  `WebcamPipeline` subclass.
- **Zone-based counting** — add a `ZoneCounter` component that accepts
  polygon coordinates from config and counts entries/exits per zone.
- **Streamlit web dashboard** — wrap the CLI in a Streamlit UI with
  file upload, live frame preview, and interactive charts.
- **Heatmap generation** — accumulate bounding box centres across all
  frames and render a 2D density heatmap using OpenCV or seaborn.

### Medium-term
- **Multi-camera tracking** — shared track ID space across cameras with
  cross-camera ReID. Requires a global ID broker and shared appearance
  feature store.
- **Distributed inference** — split detection and tracking across multiple
  processes using a frame queue (e.g., Redis or ZeroMQ).
- **Alert system** — trigger notifications when unique visitor count
  exceeds a configurable threshold or dwell time exceeds a limit.
- **YOLOv8-pose** — replace the detection backbone with a pose-estimation
  model for activity recognition (running, falling, loitering).

### Long-term
- **Cross-camera re-identification** — match persons across non-overlapping
  camera views using global ReID embeddings.
- **MOTA/MOTP evaluation** — integrate ground-truth annotation parsing for
  MOT Challenge sequences to compute standard tracking metrics.
- **Edge deployment** — export YOLO to ONNX or TensorRT for deployment on
  Jetson devices without Python overhead.
