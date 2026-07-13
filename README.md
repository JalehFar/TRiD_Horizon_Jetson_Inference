# TRiD-Horizon Inference

Self-contained inference and benchmarking repository for the final horizon-line detection methods. The complete final test set is included under `samples/`.

## Run one method

```bash
python3 run_inference.py \
    --method trid \
    --input samples/TMD/TMD_annotated_15.avi \
    --output outputs/TMD_annotated_15_trid.mp4 \
    --device cuda \
    --fp16
```

## Run all methods

```bash
python3 run_inference.py \
    --method all \
    --input samples/TMD/TMD_annotated_15.avi \
    --output outputs/TMD_annotated_15_all_methods.mp4 \
    --device cuda \
    --fp16
```

## Smoke Test

```bash
python3 run_inference.py \
    --method all \
    --input samples/TMD/TMD_annotated_15.avi \
    --output outputs/smoke_test.mp4 \
    --device cuda \
    --fp16 \
    --max-frames 20 \
    --warmup 2
```

## Run the complete test benchmark

```bash
python3 run_benchmark.py \
    --manifest samples/test_manifest.csv \
    --device cuda \
    --fp16 \
    --output outputs/full_test_benchmark
```

## Methods

- `essld`
- `directreg`
- `wls`
- `dsac`
- `trid`
- `all`

## Main arguments

- `--method`: method to run, or `all` for the synchronized 2x3 comparison video.
- `--input`: image file, image folder, or video file.
- `--output`: output video path; a CSV with the same stem is also written by default.
- `--device`: `cuda`, `cpu`, or another PyTorch device string.
- `--fp16`: use FP16 on CUDA.
- `--no-fp16`: force FP32, including on CUDA.
- `--no-roi`: disable ROI refinement and keep the coarse prediction.
- `--no-roi-gate`: run ROI refinement without the conservative acceptance gate.
- `--roi-width`: width used for CPU ROI refinement; lower is faster on Jetson.
- `--roi-every`: run ROI refinement every N frames; skipped frames use the coarse line.
- `--trid-mode`: temporal execution for `trid`; default `chunk` matches the reported evaluation.
- `--profile-stages`: write a per-frame/per-method stage timing CSV.
- `--profile-output`: optional path for the stage timing CSV.
- `--max-frames`: stop after this many frames.
- `--warmup`: number of initial rows excluded from summary latency statistics.
- `--verbose`: print per-frame CSV rows to the terminal.

Defaults are CUDA if available, FP16 on CUDA, ROI refinement enabled, bounded ROI gate enabled, no GUI, and CSV saving when `--output` is provided.

## TRiD temporal modes

`TRiD-Horizon` defaults to `--trid-mode chunk`. This is the mode used by the final evaluation: frames are processed in non-overlapping clips of `CLIP_LENGTH=8`, the final short clip is padded by repeating its last frame, and model latency is reported as the clip forward time divided by 8.

Other modes are available for diagnostics:

- `--trid-mode rolling`: streaming-style rolling prefix. This recomputes up to 8 DCEUNet backbone passes for every output frame and is much slower.
- `--trid-mode single`: one-frame temporal input (`T=1`). This is useful for isolating backbone/head latency, but it is not the final evaluated temporal pipeline.

## Jetson speed options

ROI refinement is CPU-heavy. The default path now runs ROI on a 512-wide normalized strip instead of the full source width, then maps the accepted line back to original pixels.

For more speed, reduce the ROI width:

```bash
python3 run_inference.py \
    --method trid \
    --input samples/TMD/TMD_annotated_15.avi \
    --output outputs/TMD_annotated_15_trid_fast_roi.mp4 \
    --device cuda \
    --fp16 \
    --roi-width 256
```

For maximum speed, disable ROI refinement:

```bash
python3 run_inference.py \
    --method trid \
    --input samples/TMD/TMD_annotated_15.avi \
    --output outputs/TMD_annotated_15_trid_no_roi.mp4 \
    --device cuda \
    --fp16 \
    --no-roi
```

If ROI is still desired but not on every frame:

```bash
python3 run_inference.py \
    --method trid \
    --input samples/TMD/TMD_annotated_15.avi \
    --output outputs/TMD_annotated_15_trid_roi_every_5.mp4 \
    --device cuda \
    --fp16 \
    --roi-every 5
```

## Output colors

- green: GT
- yellow: coarse prediction
- magenta: ROI search region
- cyan: final accepted prediction

If the bounded gate rejects the ROI-refined line, the final cyan line coincides with the yellow coarse line.

## Performance output

The terminal summary and CSV report:

- model-only latency;
- full-pipeline latency;
- model FPS;
- full-pipeline FPS;
- throughput FPS;
- center error;
- endpoint error;
- angular error;
- ROI acceptance rate.

Model latency measures only neural-network forward time. Full-pipeline latency includes preprocessing, model forward, post-processing, and ROI refinement. Throughput FPS is measured from processed frames divided by elapsed wall time.

When `--profile-stages` is enabled, an additional CSV is written with decode, preprocess, model, postprocess, ROI, visualization, encode, and frame-wall timings. With `--method all`, the terminal summary prints a separate `Profile:` block for ESSLD, DirectReg-HL, WLS-HL, DSAC-HL, and TRiD-Horizon.

## Outputs

Generated videos and CSV files are written under `outputs/`. The repository keeps `outputs/.gitkeep` only; generated outputs are ignored by Git.
