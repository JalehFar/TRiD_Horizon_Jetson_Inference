# TRiD-Horizon Jetson Inference

Self-contained inference and benchmarking repository for the final horizon-line detection methods. The complete final test set is included under `samples/`, so no external dataset-preparation step is required.

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
- `--max-frames`: stop after this many frames.
- `--warmup`: number of initial rows excluded from summary latency statistics.
- `--verbose`: print per-frame CSV rows to the terminal.

Defaults are CUDA if available, FP16 on CUDA, ROI refinement enabled, bounded ROI gate enabled, no GUI, and CSV saving when `--output` is provided.

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

## Outputs

Generated videos and CSV files are written under `outputs/`. The repository keeps `outputs/.gitkeep` only; generated outputs are ignored by Git.
