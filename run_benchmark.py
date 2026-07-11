from __future__ import annotations

import argparse
import csv
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from run_inference import METHODS, result_row, summarize
from inference.geometry import HorizonLine
from inference.pipeline import MethodRunner


def environment_text(device: torch.device, precision: str, warmup: int) -> str:
    lines = [
        f"Python: {sys.version}",
        f"Platform: {platform.platform()}",
        f"PyTorch: {torch.__version__}",
        f"OpenCV: {cv2.__version__}",
        f"NumPy: {np.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"CUDA version: {torch.version.cuda}",
        f"cuDNN version: {torch.backends.cudnn.version()}",
        f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}",
        f"Precision: {precision}",
        f"Input dimensions: 512x256",
        f"Warmup frames: {warmup}",
    ]
    try:
        lines.append("Git commit: " + subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:
        lines.append("Git commit: unavailable")
    for cmd, label in [(["dpkg-query", "-W", "nvidia-l4t-core"], "JetPack/L4T"), ([sys.executable, "-c", "import tensorrt as trt; print(trt.__version__)"], "TensorRT")]:
        try:
            lines.append(f"{label}: " + subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip())
        except Exception:
            lines.append(f"{label}: unavailable")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("samples/test_manifest.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/full_test_benchmark"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    precision = "fp16" if args.fp16 and device.type == "cuda" else "fp32"
    (args.output / "benchmark_environment.txt").write_text(environment_text(device, precision, args.warmup))
    with args.manifest.open() as f:
        manifest_rows = list(csv.DictReader(f))
    by_video: dict[str, list[dict]] = {}
    for row in manifest_rows:
        by_video.setdefault(row["relative_input_path"], []).append(row)

    failures = []
    all_rows = []
    for method in METHODS:
        runner = MethodRunner(method, device, args.fp16, roi=True, roi_gate=True)
        for rel_path, rows in by_video.items():
            path = Path(rel_path)
            runner.reset()
            try:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    raise FileNotFoundError(path)
                rows_sorted = sorted(rows, key=lambda r: int(r["frame_index"]))
                start = time.perf_counter()
                for i, row in enumerate(rows_sorted):
                    if args.max_frames and i >= args.max_frames:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Could not read frame {row['frame_index']}")
                    gt = HorizonLine(float(row["gt_y_left"]), float(row["gt_y_right"]), int(row["original_width"]), int(row["original_height"]))
                    res = runner.predict(frame, int(row["frame_index"]), gt)
                    out = result_row(rel_path, res, gt, (i + 1) / max(time.perf_counter() - start, 1e-9))
                    out.update({"dataset": row["dataset"], "video": row["video"]})
                    all_rows.append(out)
                cap.release()
            except Exception as exc:
                failures.append({"method": method, "input": rel_path, "error": repr(exc)})
    frame_csv = args.output / "frame_level_results.csv"
    if all_rows:
        with frame_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
    with (args.output / "failures.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "input", "error"])
        w.writeheader()
        w.writerows(failures)
    import pandas as pd
    if all_rows:
        df = pd.DataFrame(all_rows)
        df.groupby("method").agg(processed_frames=("frame_index", "count"), valid_predictions=("prediction_valid", "sum"), mean_model_latency_ms=("model_forward_ms", "mean"), mean_full_latency_ms=("full_pipeline_ms", "mean"), mean_center_error=("center_y_abs_error", "mean"), roi_acceptance=("roi_accepted", "mean")).to_csv(args.output / "method_summary.csv")
        df.groupby(["method", "dataset"]).agg(processed_frames=("frame_index", "count"), mean_center_error=("center_y_abs_error", "mean")).to_csv(args.output / "dataset_summary.csv")
        df.groupby(["method", "dataset", "video"]).agg(processed_frames=("frame_index", "count"), mean_center_error=("center_y_abs_error", "mean")).to_csv(args.output / "video_summary.csv")
    print(f"Wrote benchmark outputs to {args.output}; failures={len(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
