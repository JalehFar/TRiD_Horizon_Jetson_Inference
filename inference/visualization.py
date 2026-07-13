from __future__ import annotations

import cv2
import numpy as np

from inference.geometry import HorizonLine, draw_line, is_valid_line

GREEN = (0, 220, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_roi(frame: np.ndarray, pts: np.ndarray | None, thickness: int = 2) -> None:
    if pts is not None and np.isfinite(pts).all():
        cv2.polylines(frame, [np.round(pts).astype(np.int32)], True, MAGENTA, thickness, cv2.LINE_AA)


def status_box(frame: np.ndarray, lines: list[str]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, frame.shape[1] / 1800.0)
    thickness = max(1, int(round(scale * 2)))
    line_h = int(24 * scale)
    pad = int(7 * scale)
    max_w = max(cv2.getTextSize(s, font, scale, thickness)[0][0] for s in lines)
    cv2.rectangle(frame, (0, 0), (max_w + pad * 2, pad * 2 + line_h * len(lines)), BLACK, -1)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + line_h * (i + 1) - 4), font, scale, WHITE, thickness, cv2.LINE_AA)


def render_method_panel(frame: np.ndarray, result, latency_fps: float) -> np.ndarray:
    out = frame.copy()
    thickness = max(2, int(round(frame.shape[1] / 900.0)))
    if is_valid_line(result.gt):
        out = draw_line(out, result.gt, GREEN, thickness)
    draw_roi(out, result.roi_pts, thickness)
    if is_valid_line(result.coarse):
        out = draw_line(out, result.coarse, YELLOW, thickness)
    if is_valid_line(result.final):
        out = draw_line(out, result.final, CYAN, thickness)
    status = "valid" if result.prediction_valid else "invalid"
    roi = "accepted" if result.roi_accepted else f"rejected:{result.roi_reason}"
    status_box(out, [f"{result.method} frame {result.frame_index}", f"{status} ROI {roi}", f"{result.timing.get('full_ms', 0):.1f} ms {latency_fps:.1f} FPS"])
    return out


def render_original_panel(frame: np.ndarray, gt: HorizonLine | None, frame_index: int) -> np.ndarray:
    out = frame.copy()
    if is_valid_line(gt):
        out = draw_line(out, gt, GREEN, max(2, int(round(frame.shape[1] / 900.0))))
    status_box(out, ["Original + GT" if gt is not None else "Original", f"frame {frame_index}"])
    return out


def make_all_methods_montage(frame: np.ndarray, results: list, gt: HorizonLine | None, frame_index: int, panel_width: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_h = int(round(panel_width * h / w))
    panels = [render_original_panel(frame, gt, frame_index)]
    for result in results:
        fps = 1000.0 / max(result.timing.get("full_ms", 1e-6), 1e-6)
        panels.append(render_method_panel(frame, result, fps))
    resized = [cv2.resize(p, (panel_width, panel_h), interpolation=cv2.INTER_AREA) for p in panels]
    return np.concatenate([np.concatenate(resized[:3], axis=1), np.concatenate(resized[3:], axis=1)], axis=0)
