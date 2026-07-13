from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from config import DISPLAY_NAMES
from inference.geometry import HorizonLine, is_valid_line
from inference.pipeline import MethodRunner
from inference.visualization import make_all_methods_montage, render_method_panel


METHODS = ["essld", "directreg", "wls", "dsac", "trid"]


def choose_device(arg: str | None) -> torch.device:
    if arg:
        return torch.device(arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def open_input(path: Path):
    if path.is_dir():
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
        return "images", files
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        return "images", [path]
    return "video", path


def _norm_rel(path: Path) -> str:
    text = str(path).replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def load_manifest_gt(manifest_path: Path, input_path: Path) -> dict[int, HorizonLine]:
    if not manifest_path.exists():
        return {}
    input_keys = {_norm_rel(input_path)}
    try:
        input_keys.add(_norm_rel(input_path.resolve().relative_to(Path.cwd().resolve())))
    except Exception:
        pass
    gt_by_frame: dict[int, HorizonLine] = {}
    with manifest_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if _norm_rel(Path(row["relative_input_path"])) not in input_keys:
                continue
            gt_by_frame[int(row["frame_index"])] = HorizonLine(
                float(row["gt_y_left"]),
                float(row["gt_y_right"]),
                int(row["original_width"]),
                int(row["original_height"]),
            )
    return gt_by_frame


def result_row(source: str, result, gt: HorizonLine | None, running_fps: float) -> dict:
    err = result.errors()
    def endpoints(line):
        return (np.nan, np.nan) if not is_valid_line(line) else (line.y_left, line.y_right)
    c_l, c_r = endpoints(result.coarse)
    e_l, e_r = endpoints(result.existing_refined)
    f_l, f_r = endpoints(result.final)
    g_l, g_r = endpoints(gt)
    return {
        "method": result.method,
        "source": source,
        "frame_index": result.frame_index,
        "preprocess_ms": result.timing.get("preprocess_ms", np.nan),
        "model_forward_ms": result.timing.get("model_ms", np.nan),
        "postprocess_ms": result.timing.get("postprocess_ms", np.nan),
        "roi_ms": result.timing.get("roi_ms", np.nan),
        "full_pipeline_ms": result.timing.get("full_ms", np.nan),
        "model_only_fps": 1000.0 / max(result.timing.get("model_ms", 1e-6), 1e-6),
        "full_pipeline_instant_fps": 1000.0 / max(result.timing.get("full_ms", 1e-6), 1e-6),
        "running_average_throughput_fps": running_fps,
        "prediction_valid": int(result.prediction_valid),
        "roi_accepted": int(result.roi_accepted),
        "roi_rejection_reason": result.roi_reason,
        "coarse_y_left": c_l,
        "coarse_y_right": c_r,
        "existing_refined_y_left": e_l,
        "existing_refined_y_right": e_r,
        "accepted_final_y_left": f_l,
        "accepted_final_y_right": f_r,
        "gt_y_left": g_l,
        "gt_y_right": g_r,
        "center_y_abs_error": err["center_error"],
        "mean_endpoint_abs_error": err["endpoint_error"],
        "angular_error": err["angular_error"],
        "inside_roi_fraction": result.roi_log.get("inside_roi_fraction", np.nan),
        "center_correction": result.roi_log.get("center_correction", np.nan),
        "maximum_endpoint_correction": result.roi_log.get("endpoint_correction", np.nan),
        "angle_correction": result.roi_log.get("angle_correction", np.nan),
        "candidate_count": result.roi_log.get("candidate_count", np.nan),
        "candidate_horizontal_span": result.roi_log.get("candidate_span", np.nan),
    }


def summarize(rows: list[dict], method: str, device: torch.device, precision: str, warmup: int) -> None:
    timed = rows[warmup:] if len(rows) > warmup else rows
    model = np.array([r["model_forward_ms"] for r in timed], dtype=float)
    full = np.array([r["full_pipeline_ms"] for r in timed], dtype=float)
    center = np.array([r["center_y_abs_error"] for r in timed], dtype=float)
    endpoint = np.array([r["mean_endpoint_abs_error"] for r in timed], dtype=float)
    angle = np.array([r["angular_error"] for r in timed], dtype=float)
    print("\nSummary")
    print(f"method={method} device={device} precision={precision} frames={len(rows)} valid={sum(r['prediction_valid'] for r in rows)} warmup={warmup}")
    if len(timed):
        print(f"model latency ms mean/median/p95: {np.nanmean(model):.3f} / {np.nanmedian(model):.3f} / {np.nanpercentile(model,95):.3f}")
        print(f"full latency ms mean/median/p95: {np.nanmean(full):.3f} / {np.nanmedian(full):.3f} / {np.nanpercentile(full,95):.3f}")
        print(f"latency-derived FPS model/full: {1000/np.nanmean(model):.2f} / {1000/np.nanmean(full):.2f}")
        print(f"center error mean/median/p95: {np.nanmean(center):.3f} / {np.nanmedian(center):.3f} / {np.nanpercentile(center,95):.3f}")
        print(f"endpoint mean error: {np.nanmean(endpoint):.3f}; angular mean error: {np.nanmean(angle):.3f}")
        print(f"ROI acceptance rate: {100*np.mean([r['roi_accepted'] for r in timed]):.2f}%")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS + ["all"])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--roi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--roi-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--roi-width", type=int, default=None, help="Width used for CPU ROI refinement; lower values are faster on Jetson.")
    parser.add_argument("--roi-every", type=int, default=1, help="Run ROI refinement every N frames; skipped frames use the coarse line.")
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("samples/test_manifest.csv"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    fp16 = (device.type == "cuda") if args.fp16 is None else args.fp16
    methods = METHODS if args.method == "all" else [args.method]
    runners = [MethodRunner(m, device, fp16, args.roi, args.roi_gate, args.roi_width, args.roi_every) for m in methods]
    kind, source = open_input(args.input)
    gt_by_frame = load_manifest_gt(args.manifest, args.input)
    if gt_by_frame:
        print(f"Loaded GT for {len(gt_by_frame)} frames from {args.manifest}")
    rows: list[dict] = []
    writer = None
    video_writer = None
    start = time.perf_counter()

    frames = []
    fps = 25.0
    if kind == "images":
        frames = [(i + 1, cv2.imread(str(p)), str(p)) for i, p in enumerate(source)]
    else:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(source)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and idx >= args.max_frames):
                break
            idx += 1
            frames.append((idx, frame, str(source)))
        cap.release()
    if args.max_frames:
        frames = frames[: args.max_frames]
    for r in runners:
        r.reset()

    csv_handle = None
    try:
        if args.output and args.save_csv:
            csv_path = args.output.with_suffix(".csv") if args.output.suffix.lower() != ".csv" else args.output
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_handle = csv_path.open("w", newline="")
            writer = None
        for frame_index, frame, src in frames:
            gt = gt_by_frame.get(frame_index)
            results = [r.predict(frame, frame_index, gt) for r in runners]
            elapsed = time.perf_counter() - start
            running_fps = frame_index / max(elapsed, 1e-9)
            for res in results:
                row = result_row(src, res, gt, running_fps)
                rows.append(row)
                if writer is None and csv_handle is not None:
                    writer = csv.DictWriter(csv_handle, fieldnames=list(row.keys()))
                    writer.writeheader()
                if writer:
                    writer.writerow(row)
                if args.verbose:
                    print(row)
            if args.output and args.save_video:
                if args.method == "all":
                    vis = make_all_methods_montage(frame, results, gt, frame_index)
                else:
                    vis = render_method_panel(frame, results[0], 1000.0 / max(results[0].timing["full_ms"], 1e-6))
                if video_writer is None:
                    out_video = args.output if args.output.suffix.lower() != ".csv" else args.output.with_suffix(".mp4")
                    out_video.parent.mkdir(parents=True, exist_ok=True)
                    video_writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vis.shape[1], vis.shape[0]))
                video_writer.write(vis)
                if args.show:
                    cv2.imshow("horizon", vis)
                    if cv2.waitKey(1) == 27:
                        break
    finally:
        if video_writer:
            video_writer.release()
        if csv_handle:
            csv_handle.close()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    summarize(rows, args.method, device, "fp16" if fp16 else "fp32", args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
