# Validation

Validation was refreshed on 2026-07-12 after copying the complete final test videos directly into the repository.

## Self-Contained Data Check

- Test manifest: `samples/test_manifest.csv`
- Manifest rows: 2196
- Unique videos: 8
- Datasets: Buoy, SMD, TMD
- Annotation sidecars: present
- Symlinks inside repository: none
- Video paths: all relative and all point to files inside `samples/`

Included videos:

| dataset | video | frames | resolution | FPS |
|---|---|---:|---|---:|
| Buoy | `buoyGT_2_5_3_5.avi` | 98 | 800 x 600 | 25.000 |
| Buoy | `buoyGT_2_6_3_1.avi` | 98 | 800 x 600 | 25.000 |
| SMD | `MVI_0788_VIS_OB.mp4` | 299 | 1920 x 1080 | 30.000 |
| SMD | `MVI_0790_VIS_OB.mp4` | 299 | 1920 x 1080 | 30.000 |
| TMD | `TMD_annotated_5.avi` | 346 | 1920 x 1080 | 29.612 |
| TMD | `TMD_annotated_15.avi` | 371 | 1920 x 1080 | 29.607 |
| TMD | `TMD_annotated_16.avi` | 322 | 1920 x 1080 | 29.607 |
| TMD | `TMD_annotated_17.avi` | 363 | 1920 x 1080 | 29.608 |

## Size Check

- Total size of 8 test videos: 240,632,568 bytes, 229.49 MiB
- Clean working-tree payload size, excluding `.git`: 255,622,860 bytes, 243.78 MiB
- Largest file: `samples/TMD/TMD_annotated_15.avi`, 47,006,770 bytes, 44.83 MiB
- Git LFS enabled: no

Rationale: `git-lfs` is not installed in this environment, and no individual tracked file exceeds 100 MB. The videos and checkpoints are kept as normal Git files.

## Checkpoints

All checkpoints loaded successfully and SHA256 hashes matched `weights/checkpoint_manifest.csv`.

| method | checkpoint | SHA256 |
|---|---|---|
| ESSLD | `weights/essld_dceunetex.pth` | `4261c4e6d0f51b76f6101bcee336943994a614ae970cf8e57894266fb7d8da36` |
| DirectReg-HL | `weights/directreg_hl_best_full.pt` | `0e1dc48afd259cbfa52e4b2e0fa0da2c740e0c992208c6e058845b6824253a89` |
| WLS-HL | `weights/wls_hl_best_full.pt` | `b9e69b7a70b4c87caf0838ec69e35b286f77e5a0cdbeb3ec33f8adb802603a30` |
| DSAC-HL | `weights/dsac_hl_best_full.pt` | `17634e52931f02db3c2ffc1c5e8da8951570e3dfc2f8e8675d9d50dfaa2d4cba` |
| TRiD-Horizon | `weights/trid_horizon_best_visible_y95.pt` | `bd74a14fa9dddc410a61f1b974743d8c31d4d8a767040e8c94bf3e12cc9e6c71` |

## Commands Executed

Syntax/import check:

```bash
python -m py_compile config.py run_inference.py run_benchmark.py models/*.py inference/*.py
```

Required self-contained all-method validation:

```bash
python run_inference.py \
  --method all \
  --input samples/TMD/TMD_annotated_15.avi \
  --output outputs/self_contained_validation.mp4 \
  --device cuda \
  --fp16 \
  --max-frames 20 \
  --warmup 2
```

Result:

- Checkpoints loaded: all five
- GT loaded: 371 frames
- Output video generated: yes
- Output CSV generated: yes
- Result rows: 100, equal to 20 frames x 5 methods
- Valid predictions: 100 / 100
- Latency and FPS printed: yes
- ROI acceptance rate printed: yes

Required benchmark validation:

```bash
python run_benchmark.py \
  --manifest samples/test_manifest.csv \
  --device cuda \
  --fp16 \
  --output outputs/self_contained_benchmark_validation \
  --max-frames 2 \
  --warmup 0
```

Result:

- Videos touched: all 8
- Source frames touched: 16
- Method-frame rows: 80
- Failures: 0
- CSV files generated: yes
- Environment file generated: yes

## Isolated Copy Check

The repository was copied to a temporary isolated directory without the original research project on `PYTHONPATH`.

Commands run there:

```bash
PYTHONPATH= python -m py_compile config.py run_inference.py run_benchmark.py models/*.py inference/*.py

PYTHONPATH= python run_inference.py \
  --method essld \
  --input samples/TMD/TMD_annotated_15.avi \
  --output outputs/isolated_self_contained_essld_2f.mp4 \
  --device cuda \
  --fp16 \
  --max-frames 2 \
  --warmup 0

PYTHONPATH= python run_benchmark.py \
  --manifest samples/test_manifest.csv \
  --device cuda \
  --fp16 \
  --output outputs/isolated_self_contained_benchmark \
  --max-frames 1 \
  --warmup 0
```

Result:

- Original research project imports required: no
- Symlinks found: no
- All five checkpoints loaded in isolated benchmark: yes
- All 8 test videos found in isolated benchmark: yes
- Failures: 0

## Cleanliness

- `outputs/` is empty except `outputs/.gitkeep`
- No dataset preparation script is required
- No machine-specific dataset path is required
- No training code, publication scripts, debug artifacts, or generated validation outputs are needed for normal use

## Known Limitations

- Jetson FPS was not measured locally.
- TensorRT is not implemented.
- The complete full-test benchmark without `--max-frames` is supported but was not rerun during this cleanup pass because the required validation used the requested benchmark slice.
