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
        "trid_mode": result.roi_log.get("trid_mode", ""),
        "trid_chunk_length": result.roi_log.get("trid_chunk_length", np.nan),
        "trid_chunk_position": result.roi_log.get("trid_chunk_position", np.nan),
        "trid_actual_chunk_frames": result.roi_log.get("trid_actual_chunk_frames", np.nan),
        "trid_backbone_evals_total": result.roi_log.get("trid_backbone_evals_total", np.nan),
        "trid_backbone_evals_amortized": result.roi_log.get("trid_backbone_evals_amortized", np.nan),
    }


def _summarize_block(rows: list[dict], label: str, warmup: int) -> None:
    timed = rows[warmup:] if len(rows) > warmup else rows
    print(f"\n{label}")
    print(f"frames={len(rows)} valid={sum(r['prediction_valid'] for r in rows)} warmup={warmup}")
    if not len(timed):
        return
    model = np.array([r["model_forward_ms"] for r in timed], dtype=float)
    full = np.array([r["full_pipeline_ms"] for r in timed], dtype=float)
    center = np.array([r["center_y_abs_error"] for r in timed], dtype=float)
    endpoint = np.array([r["mean_endpoint_abs_error"] for r in timed], dtype=float)
    angle = np.array([r["angular_error"] for r in timed], dtype=float)
    print(f"model latency ms mean/median/p95: {np.nanmean(model):.3f} / {np.nanmedian(model):.3f} / {np.nanpercentile(model,95):.3f}")
    print(f"full latency ms mean/median/p95: {np.nanmean(full):.3f} / {np.nanmedian(full):.3f} / {np.nanpercentile(full,95):.3f}")
    print(f"latency-derived FPS model/full: {1000/np.nanmean(model):.2f} / {1000/np.nanmean(full):.2f}")
    print(f"center error mean/median/p95: {np.nanmean(center):.3f} / {np.nanmedian(center):.3f} / {np.nanpercentile(center,95):.3f}")
    print(f"endpoint mean error: {np.nanmean(endpoint):.3f}; angular mean error: {np.nanmean(angle):.3f}")
    print(f"ROI acceptance rate: {100*np.mean([r['roi_accepted'] for r in timed]):.2f}%")


def summarize(rows: list[dict], method: str, device: torch.device, precision: str, warmup: int) -> None:
    print("\nSummary")
    print(f"method={method} device={device} precision={precision} frames={len(rows)} valid={sum(r['prediction_valid'] for r in rows)} warmup={warmup}")
    if method == "all":
        for method_name in sorted({r["method"] for r in rows}):
            _summarize_block([r for r in rows if r["method"] == method_name], f"Profile: {method_name}", warmup)
        _summarize_block(rows, "Overall all-method row profile", warmup)
    else:
        _summarize_block(rows, f"Profile: {method}", warmup)


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
    parser.add_argument(
        "--trid-mode",
        choices=["chunk", "rolling", "single"],
        default="chunk",
        help="TRiD temporal execution. chunk matches the reported evaluation; rolling is streaming-style and recomputes the clip each frame; single is T=1.",
    )
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--profile-stages", action="store_true", help="Write a per-frame/per-method stage timing CSV.")
    parser.add_argument("--profile-output", type=Path, default=None, help="Optional path for --profile-stages CSV.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("samples/test_manifest.csv"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = choose_device(args.device)
    fp16 = (device.type == "cuda") if args.fp16 is None else args.fp16
    methods = METHODS if args.method == "all" else [args.method]
    runners = [MethodRunner(m, device, fp16, args.roi, args.roi_gate, args.roi_width, args.roi_every, args.trid_mode) for m in methods]
    kind, source = open_input(args.input)
    gt_by_frame = load_manifest_gt(args.manifest, args.input)
    if gt_by_frame:
        print(f"Loaded GT for {len(gt_by_frame)} frames from {args.manifest}")
    rows: list[dict] = []
    stage_rows: list[dict] = []
    writer = None
    video_writer = None
    start = time.perf_counter()

    frames = []
    fps = 25.0
    if kind == "images":
        for i, p in enumerate(source):
            t_decode = time.perf_counter()
            frame = cv2.imread(str(p))
            decode_ms = (time.perf_counter() - t_decode) * 1000.0
            frames.append((i + 1, frame, str(p), decode_ms))
    else:
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise FileNotFoundError(source)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        idx = 0
        while True:
            t_decode = time.perf_counter()
            ok, frame = cap.read()
            decode_ms = (time.perf_counter() - t_decode) * 1000.0
            if not ok or (args.max_frames and idx >= args.max_frames):
                break
            idx += 1
            frames.append((idx, frame, str(source), decode_ms))
        cap.release()
    if args.max_frames:
        frames = frames[: args.max_frames]
    for r in runners:
        r.reset()

    trid_chunk_results: dict[int, object] = {}
    if args.trid_mode == "chunk":
        trid_runner = next((r for r in runners if r.method == "trid"), None)
        if trid_runner is not None:
            import config

            print(f"Precomputing TRiD-Horizon in non-overlapping chunks of {config.CLIP_LENGTH} frames")
            for start_i in range(0, len(frames), config.CLIP_LENGTH):
                chunk = frames[start_i : start_i + config.CLIP_LENGTH]
                items = [(frame_index, frame, gt_by_frame.get(frame_index)) for frame_index, frame, _, _ in chunk]
                for res in trid_runner.predict_trid_chunk(items):
                    trid_chunk_results[res.frame_index] = res

    csv_handle = None
    try:
        if args.output and args.save_csv:
            csv_path = args.output.with_suffix(".csv") if args.output.suffix.lower() != ".csv" else args.output
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_handle = csv_path.open("w", newline="")
            writer = None
        for frame_index, frame, src, decode_ms in frames:
            frame_wall_start = time.perf_counter()
            gt = gt_by_frame.get(frame_index)
            results = [
                trid_chunk_results[frame_index] if (r.method == "trid" and args.trid_mode == "chunk") else r.predict(frame, frame_index, gt)
                for r in runners
            ]
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
            visualization_ms = 0.0
            encode_ms = 0.0
            if args.output and args.save_video:
                t_vis = time.perf_counter()
                if args.method == "all":
                    vis = make_all_methods_montage(frame, results, gt, frame_index)
                else:
                    vis = render_method_panel(frame, results[0], 1000.0 / max(results[0].timing["full_ms"], 1e-6))
                visualization_ms = (time.perf_counter() - t_vis) * 1000.0
                if video_writer is None:
                    out_video = args.output if args.output.suffix.lower() != ".csv" else args.output.with_suffix(".mp4")
                    out_video.parent.mkdir(parents=True, exist_ok=True)
                    video_writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vis.shape[1], vis.shape[0]))
                t_encode = time.perf_counter()
                video_writer.write(vis)
                encode_ms = (time.perf_counter() - t_encode) * 1000.0
                if args.show:
                    cv2.imshow("horizon", vis)
                    if cv2.waitKey(1) == 27:
                        break
            if args.profile_stages:
                frame_wall_ms = (time.perf_counter() - frame_wall_start) * 1000.0
                for res in results:
                    stage_rows.append(
                        {
                            "source": src,
                            "frame_index": frame_index,
                            "method": res.method,
                            "decode_ms": decode_ms,
                            "preprocess_ms": res.timing.get("preprocess_ms", np.nan),
                            "model_forward_ms": res.timing.get("model_ms", np.nan),
                            "postprocess_ms": res.timing.get("postprocess_ms", np.nan),
                            "roi_ms": res.timing.get("roi_ms", np.nan),
                            "visualization_ms": visualization_ms,
                            "encode_ms": encode_ms,
                            "method_full_pipeline_ms": res.timing.get("full_ms", np.nan),
                            "frame_wall_ms": frame_wall_ms,
                            "prediction_valid": int(res.prediction_valid),
                            "roi_accepted": int(res.roi_accepted),
                            "roi_reason": res.roi_reason,
                            "trid_mode": res.roi_log.get("trid_mode", ""),
                            "trid_chunk_length": res.roi_log.get("trid_chunk_length", np.nan),
                            "trid_chunk_position": res.roi_log.get("trid_chunk_position", np.nan),
                            "trid_backbone_evals_amortized": res.roi_log.get("trid_backbone_evals_amortized", np.nan),
                        }
                    )
    finally:
        if video_writer:
            video_writer.release()
        if csv_handle:
            csv_handle.close()
        if args.profile_stages and stage_rows:
            if args.profile_output is not None:
                profile_path = args.profile_output
            elif args.output is not None:
                profile_path = args.output.with_name(args.output.stem + "_stage_profile.csv")
            else:
                profile_path = Path("outputs/run_inference_stage_profile.csv")
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            with profile_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(stage_rows[0].keys()))
                w.writeheader()
                w.writerows(stage_rows)
            print(f"Wrote stage profile to {profile_path}")
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
    summarize(rows, args.method, device, "fp16" if fp16 else "fp32", args.warmup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
