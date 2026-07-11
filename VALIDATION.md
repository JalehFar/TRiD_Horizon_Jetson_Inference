# Validation

Validation was run on 2026-07-11 from the extracted inference-only repository.

## Environment

- Host GPU: NVIDIA RTX A5000
- CUDA available: yes
- Test precision: CUDA FP16 by default, with an additional CUDA FP32 parity run
- Jetson hardware: not available locally

## Commands Executed

```bash
python -m py_compile config.py run_inference.py run_benchmark.py models/*.py inference/*.py tools/*.py

python - <<'PY'
import torch
from inference.pipeline import MethodRunner
for m in ['essld','directreg','wls','dsac','trid']:
    r = MethodRunner(m, torch.device('cpu'), fp16=False)
    print(m, type(r.model).__name__, 'loaded')
PY

python tools/prepare_test_data.py --source-root /path/to/HL --symlink

python run_inference.py --method all \
  --input samples/TMD/TMD_annotated_15.avi \
  --output outputs/validation_tmd15_all_methods_20f.mp4 \
  --max-frames 20 --device cuda --fp16 --warmup 2

python run_inference.py --method all \
  --input samples/TMD/TMD_annotated_15.avi \
  --output outputs/validation_tmd15_all_methods_20f_fp32.mp4 \
  --max-frames 20 --device cuda --no-fp16 --warmup 2

python run_benchmark.py \
  --manifest samples/test_manifest.csv \
  --device cuda --fp16 \
  --output outputs/validation_benchmark_slice \
  --max-frames 2 --warmup 0

python run_inference.py --method all \
  --input samples/Buoy/buoyGT_2_5_3_5.avi \
  --output outputs/validation_buoy_full_all_methods.mp4 \
  --device cuda --fp16 --warmup 5
```

The repository was also copied to `/tmp/TRiD_Horizon_Jetson_Inference_validation` and run with an empty `PYTHONPATH`:

```bash
PYTHONPATH= python -m py_compile config.py run_inference.py run_benchmark.py models/*.py inference/*.py tools/*.py
PYTHONPATH= python run_inference.py --method essld \
  --input samples/TMD/TMD_annotated_15.avi \
  --output outputs/isolated_essld_2f.mp4 \
  --max-frames 2 --device cuda --fp16 --warmup 0
```

## Checkpoints

All checkpoints loaded successfully and SHA256 hashes matched `weights/checkpoint_manifest.csv`.

| method | checkpoint | SHA256 |
|---|---|---|
| ESSLD | `weights/essld_dceunetex.pth` | `4261c4e6d0f51b76f6101bcee336943994a614ae970cf8e57894266fb7d8da36` |
| DirectReg-HL | `weights/directreg_hl_best_full.pt` | `0e1dc48afd259cbfa52e4b2e0fa0da2c740e0c992208c6e058845b6824253a89` |
| WLS-HL | `weights/wls_hl_best_full.pt` | `b9e69b7a70b4c87caf0838ec69e35b286f77e5a0cdbeb3ec33f8adb802603a30` |
| DSAC-HL | `weights/dsac_hl_best_full.pt` | `17634e52931f02db3c2ffc1c5e8da8951570e3dfc2f8e8675d9d50dfaa2d4cba` |
| TRiD-Horizon | `weights/trid_horizon_best_visible_y95.pt` | `bd74a14fa9dddc410a61f1b974743d8c31d4d8a767040e8c94bf3e12cc9e6c71` |

## Test Manifest

- Manifest: `samples/test_manifest.csv`
- Test videos: 8
- Test frames: 2196
- Datasets covered: Buoy, SMD, TMD
- Video files are not tracked by Git; `tools/prepare_test_data.py` symlinks or copies them into the expected relative paths.

## Inference Checks

Short synchronized all-method TMD video:

- Output: `outputs/validation_tmd15_all_methods_20f.mp4`
- OpenCV properties: 20 frames, 29.607 FPS, 1920 x 720
- CSV rows: 100 rows, equal to 20 frames x 5 methods
- GT loaded from `samples/test_manifest.csv`: 371 frames
- All five methods produced valid predictions for all 20 frames

Full synchronized all-method Buoy video:

- Output: `outputs/validation_buoy_full_all_methods.mp4`
- OpenCV properties: 98 frames, 25.0 FPS, 1920 x 960
- CSV rows: 490 rows, equal to 98 frames x 5 methods
- Methods present: ESSLD, DirectReg-HL, WLS-HL, DSAC-HL, TRiD-Horizon
- All five methods produced valid predictions for all 98 frames

The all-method montage keeps one source frame synchronized across the original/GT panel and the five method panels.

## Benchmark Checks

`run_benchmark.py` was run over the complete manifest with `--max-frames 2`, which processes the first two frames of every test video for every method:

- Videos touched: 8
- Source frames touched: 16
- Method-frame rows: 80
- Failures: 0
- Output directory: `outputs/validation_benchmark_slice`
- Files written: `frame_level_results.csv`, `method_summary.csv`, `dataset_summary.csv`, `video_summary.csv`, `benchmark_environment.txt`, `failures.csv`

A second one-frame benchmark slice after the Git-output patch also completed with zero failures.

## Original-Artifact Parity Check

The packaged outputs were compared against the older original comparison artifact:

`TRiD_Horizon/publication/full_video_method_comparisons/_test_100_frames/TMD/TMD_annotated_15_all_methods_roi_comparison.csv`

Comparison used the first 20 frames of `TMD_annotated_15` in CUDA FP32. The older artifact appears to store the raw ROI-refined line, while this package stores both the raw refined line and the accepted final line after the bounded ROI gate. Therefore, this is a coordinate/preprocessing sanity check, not a strict final-output equality proof.

| method | frames | max coarse endpoint diff px | max raw refined endpoint diff px | package ROI acceptance |
|---|---:|---:|---:|---:|
| DSAC-HL | 20 | 5.883 | 10.934 | 1.00 |
| DirectReg-HL | 20 | 2.272 | 80.913 | 0.45 |
| ESSLD | 20 | 2.786 | 923.669 | 0.05 |
| TRiD-Horizon | 20 | 15.181 | 43.361 | 0.80 |
| WLS-HL | 20 | 2.198 | 8.728 | 1.00 |

Known interpretation: WLS-HL, DSAC-HL, and DirectReg-HL coarse predictions are close to the old artifact. TRiD-Horizon differs more because temporal prefix handling and DSAC sampling affect early frames. ESSLD raw ROI refinement differs strongly on several frames; the packaged final output falls back to the coarse line when the bounded gate rejects the refinement.

## Frame Indexing

The runtime frame index is one-based and matches `samples/test_manifest.csv`. The TMD validation video loaded 371 GT rows for 371 video frames.

## TRiD Temporal State

The runner calls `reset()` at the start of each video. For the first frames, TRiD-Horizon uses the available prefix up to `CLIP_LENGTH=8`; it does not use future frames and does not carry state across videos. The benchmark slice opened each video separately and reset the runner per video.

## Timing Checks

Per-frame CSV output separates:

- preprocessing latency;
- model-forward latency;
- post-processing latency;
- ROI latency;
- full-pipeline latency.

CUDA model timing uses `torch.cuda.Event` with synchronization. CPU timing uses `time.perf_counter()`.

## Isolated Repository Check

The extracted repository was copied to `/tmp/TRiD_Horizon_Jetson_Inference_validation` without the original project package. With `PYTHONPATH=` it compiled and ran ESSLD inference successfully. This confirms that no training file or original repository import is required for basic inference.

## Known Limitations

- Jetson FPS and JetPack behavior were not measured locally.
- TensorRT is not implemented.
- Full-manifest benchmarking without `--max-frames` is available but was not run in this validation pass because it is computationally heavier.
- Strict prediction equality against all original final evaluation scripts remains partially open for ESSLD raw ROI refinement and early TRiD temporal/DSAC outputs. The packaged code uses the final checkpoints, final preprocessing, and the bounded ROI acceptance gate, and records raw refined and accepted final endpoints separately.
