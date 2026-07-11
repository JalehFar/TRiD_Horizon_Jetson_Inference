# TRiD-Horizon Jetson Inference

Minimal inference-only repository for final horizon-line detection methods:

- `ESSLD`: DCEUNet segmentation, mask-derived coarse line, bounded ROI refinement.
- `DirectReg-HL`: DCEUNet enhanced feature, global-pooling regression head, bounded ROI refinement.
- `WLS-HL`: DCEUNet bottle feature, column heatmap/confidence, weighted least-squares fit, bounded ROI refinement.
- `DSAC-HL`: DCEUNet bottle feature, column heatmap/confidence, DSAC fit, bounded ROI refinement.
- `TRiD-Horizon`: DCEUNet enhanced features over a frame sequence, ConvGRU, DSAC fit, bounded ROI refinement.

## Structure

```text
config.py
run_inference.py
run_benchmark.py
models/
inference/
weights/
samples/
outputs/
tools/
```

## Checkpoints

See `weights/checkpoint_manifest.csv`.

| method | checkpoint | temporal | ROI |
|---|---|---:|---:|
| essld | `weights/essld_dceunetex.pth` | no | yes |
| directreg | `weights/directreg_hl_best_full.pt` | no | yes |
| wls | `weights/wls_hl_best_full.pt` | no | yes |
| dsac | `weights/dsac_hl_best_full.pt` | no | yes |
| trid | `weights/trid_horizon_best_visible_y95.pt` | yes | yes |

At startup the scripts print checkpoint path, SHA256, device, precision, and input size. A missing or hash-mismatched checkpoint raises an error.

## Coordinates

Model input is always RGB `512 x 256`, float `[0,1]`, NCHW. Output CSV line endpoints are in original input-frame pixel coordinates:

- `y_left`: line y at `x=0`
- `y_right`: line y at `x=width-1`

GT endpoints in `samples/test_manifest.csv` use the same original image coordinate convention.

## Installation

Desktop:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install a PyTorch build matching your CUDA/CPU environment first.
pip install -r requirements.txt
```

Jetson:

```bash
# Install the NVIDIA JetPack-compatible PyTorch wheel first.
# Then install the remaining packages:
pip3 install numpy opencv-python pandas
```

Do not install a generic PyPI PyTorch wheel on Jetson unless it matches your JetPack/CUDA stack.

## Prepare Test Data

The repo includes complete test manifests and annotation sidecars, not the large videos.

```bash
python3 tools/prepare_test_data.py --source-root /path/to/HL --symlink
```

Expected test set: 8 videos, 2196 frames.

## Inference

Single method on one video:

```bash
python3 run_inference.py --method trid --input samples/TMD/TMD_annotated_15.avi --output outputs/TMD_annotated_15_trid.mp4
```

All methods in synchronized 2x3 panels:

```bash
python3 run_inference.py --method all --input samples/TMD/TMD_annotated_15.avi --output outputs/TMD_annotated_15_all_methods.mp4
```

Single image:

```bash
python3 run_inference.py --method wls --input samples/example.png --output outputs/example_wls.mp4
```

Folder of images:

```bash
python3 run_inference.py --method dsac --input samples/my_images --output outputs/my_images_dsac.mp4
```

Useful options:

```bash
--device cuda
--fp16
--no-fp16
--no-roi
--no-roi-gate
--max-frames 100
--warmup 5
--verbose
```

Defaults: CUDA if available, FP16 on CUDA, bounded ROI gate enabled, CSV saved when `--output` is provided, no GUI.

## Visualization Legend

- green: GT
- yellow: coarse prediction before ROI
- magenta: actual ROI search band
- cyan: final accepted output after bounded ROI gate

If the gate rejects refinement, cyan overlaps yellow because the coarse line is retained.

## Bounded ROI

Inference sequence:

1. compute coarse line;
2. compute existing ROI-refined line;
3. evaluate gate;
4. accept refined line or fall back to coarse.

The gate does not use GT. Thresholds live in `inference/roi_gate.py`.

The CSV logs gate accepted/rejected, reason, inside-ROI fraction, center correction, endpoint correction, angle correction, candidate count, and candidate span.

## TRiD Temporal State

`TRiD-Horizon` resets temporal state at the beginning of every video. The inference runner keeps a rolling history up to `CLIP_LENGTH=8`; first frames use the available shorter prefix and do not use future frames. No hidden state is carried across videos.

## Benchmark

Run all methods on the prepared complete test manifest:

```bash
python3 run_benchmark.py --manifest samples/test_manifest.csv --device cuda --fp16 --output outputs/full_test_benchmark
```

Outputs:

- `frame_level_results.csv`
- `method_summary.csv`
- `dataset_summary.csv`
- `video_summary.csv`
- `benchmark_environment.txt`
- `failures.csv`

Latency reporting separates model-only latency from full preprocessing + model + post-processing + ROI latency. CUDA model timing uses `torch.cuda.Event`; CPU timing uses `time.perf_counter()`.

## CSV Fields

Important fields include method, source, frame index, preprocessing/model/postprocess/ROI/full latency, model-only FPS, full-pipeline FPS, running throughput FPS, prediction-valid flag, ROI gate status/reason, coarse endpoints, existing refined endpoints, accepted final endpoints, GT endpoints when available, and center/endpoint/angular errors.

## FP16

`--fp16` is used only on CUDA. CPU inference remains FP32.

## Known Limitations

- TensorRT is not implemented here; it is future work.
- Jetson FPS must be measured on the target Jetson. This README does not claim Jetson FPS.
- The complete test videos are external to Git; use `tools/prepare_test_data.py`.
- DSAC uses stochastic sampling inside the model head; run-to-run exact equality may require explicit PyTorch RNG seeding in a deployment wrapper.
