#!/usr/bin/env python3
"""Same-server GPU benchmark for TRiD-Horizon, ESSLD, and Fast Horizon."""

import contextlib
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
FAST_HORIZON_DIR = REPO / "external" / "fast_horizon"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(FAST_HORIZON_DIR / "original"))

from inference.pipeline import MethodRunner
from FastHorizonAlg import FastHorizon


WARMUP_FRAMES = 5
DEVICE = torch.device("cuda")
FP16 = False
ROI_ENABLED = True
ROI_GATE_ENABLED = False
ROI_WIDTH = None
ROI_EVERY = 1
TRID_MODE = "streaming"

VIDEOS = [
    ("Buoy_buoyGT_2_5_3_5", REPO / "samples/Buoy/buoyGT_2_5_3_5.avi"),
    ("Buoy_buoyGT_2_6_3_1", REPO / "samples/Buoy/buoyGT_2_6_3_1.avi"),
    ("SMD_MVI_0788", REPO / "samples/SMD/MVI_0788_VIS_OB.mp4"),
    ("SMD_MVI_0790", REPO / "samples/SMD/MVI_0790_VIS_OB.mp4"),
    ("TMD_TMD_annotated_15", REPO / "samples/TMD/TMD_annotated_15.avi"),
    ("TMD_TMD_annotated_16", REPO / "samples/TMD/TMD_annotated_16.avi"),
    ("TMD_TMD_annotated_17", REPO / "samples/TMD/TMD_annotated_17.avi"),
    ("TMD_TMD_annotated_5", REPO / "samples/TMD/TMD_annotated_5.avi"),
]
METHODS = ["trid", "essld", "fast_horizon"]
DISPLAY_NAMES = {
    "trid": "TRiD-Horizon",
    "essld": "ESSLD",
    "fast_horizon": "Fast Horizon",
}


def read_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) <= WARMUP_FRAMES:
        raise RuntimeError(f"{path}: decoded {len(frames)} frames; need more than {WARMUP_FRAMES}")
    return frames


def new_fast_detector(frame: np.ndarray) -> FastHorizon:
    height, width = frame.shape[:2]
    detector = FastHorizon()
    detector.org_width = width
    detector.org_height = height
    detector.res_width = int(width * detector.resize_factor)
    detector.res_height = int(height * detector.resize_factor)
    detector.reset_for_new_video()
    return detector


def quiet_call(function):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function()


def timed_neural_call(runner, frame: np.ndarray, frame_index: int) -> float:
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sink = io.StringIO()
    sys.stdout, sys.stderr = sink, sink
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        runner.predict(frame, frame_index, None)
        torch.cuda.synchronize()
        return time.perf_counter() - start
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def timed_fast_call(detector, frame: np.ndarray) -> float:
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sink = io.StringIO()
    sys.stdout, sys.stderr = sink, sink
    try:
        start = time.perf_counter()
        detector.get_horizon(frame, get_image=False)
        return time.perf_counter() - start
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def summarize(method: str, video: str, durations: list[float]) -> dict:
    values = np.asarray(durations, dtype=float)
    total = float(values.sum())
    return {
        "method": DISPLAY_NAMES[method],
        "video": video,
        "timed_frames": int(values.size),
        "mean_latency_ms": float(values.mean() * 1000.0),
        "median_latency_ms": float(np.median(values) * 1000.0),
        "p95_latency_ms": float(np.percentile(values, 95) * 1000.0),
        "total_timed_processing_seconds": total,
        "aggregate_fps": float(values.size / total),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; torch.cuda.is_available() is False")

    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    print("repository:", REPO)
    print("git_commit:", git_hash)
    print("gpu:", torch.cuda.get_device_name(0))
    print("device:", DEVICE)
    print("precision: fp32")
    print("trid_mode:", TRID_MODE)
    print("roi_enabled:", ROI_ENABLED)
    print("roi_gate_enabled:", ROI_GATE_ENABLED)
    print("roi_width:", ROI_WIDTH)
    print("roi_every:", ROI_EVERY)
    print("warmup_frames:", WARMUP_FRAMES)
    print("timed_frames_per_video: all frames after warm-up")
    print("timed_neural_region: MethodRunner.predict including preprocessing, transfer, model, postprocessing, and ROI")
    print("timed_fast_region: original FastHorizon.get_horizon(frame, get_image=False)")
    print("decode/output/visualization/printing: excluded")

    source = FAST_HORIZON_DIR / "original" / "FastHorizonAlg.py"
    print("fast_horizon_source_sha256:", hashlib.sha256(source.read_bytes()).hexdigest())

    runners = {
        "trid": MethodRunner("trid", DEVICE, FP16, ROI_ENABLED, ROI_GATE_ENABLED, ROI_WIDTH, ROI_EVERY, TRID_MODE),
        "essld": MethodRunner("essld", DEVICE, FP16, ROI_ENABLED, ROI_GATE_ENABLED, ROI_WIDTH, ROI_EVERY, TRID_MODE),
    }
    pooled = {method: [] for method in METHODS}
    per_video = []

    for video_name, path in VIDEOS:
        frames = read_frames(path)
        print(f"loaded {video_name}: {len(frames)} frames")
        for method in METHODS:
            if method in runners:
                runner = runners[method]
                runner.reset()
                for index, frame in enumerate(frames[:WARMUP_FRAMES], start=1):
                    quiet_call(lambda frame=frame, index=index: runner.predict(frame, index, None))
                runner.reset()
                for index, frame in enumerate(frames[:WARMUP_FRAMES], start=1):
                    quiet_call(lambda frame=frame, index=index: runner.predict(frame, index, None))
                durations = [
                    timed_neural_call(runner, frame, index)
                    for index, frame in enumerate(frames[WARMUP_FRAMES:], start=WARMUP_FRAMES + 1)
                ]
            else:
                detector = new_fast_detector(frames[0])
                for frame in frames[:WARMUP_FRAMES]:
                    quiet_call(lambda frame=frame: detector.get_horizon(frame, get_image=False))
                detector = new_fast_detector(frames[0])
                for frame in frames[:WARMUP_FRAMES]:
                    quiet_call(lambda frame=frame: detector.get_horizon(frame, get_image=False))
                durations = [
                    timed_fast_call(detector, frame)
                    for frame in frames[WARMUP_FRAMES:]
                ]

            pooled[method].extend(durations)
            row = summarize(method, video_name, durations)
            per_video.append(row)
            print(json.dumps(row))

    print("=== per-video results ===")
    for row in per_video:
        print(json.dumps(row))

    print("=== pooled results ===")
    pooled_rows = {}
    for method in METHODS:
        row = summarize(method, "all_videos", pooled[method])
        pooled_rows[method] = row
        print(json.dumps(row))

    order = sorted(METHODS, key=lambda method: np.mean(pooled[method]))
    print("FASTEST -> SLOWEST: " + " -> ".join(DISPLAY_NAMES[method] for method in order))


if __name__ == "__main__":
    main()
