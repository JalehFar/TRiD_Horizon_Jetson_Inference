from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inference.geometry import HorizonLine
from inference.postprocess import refine_existing


@dataclass(frozen=True)
class ROIGateConfig:
    min_inside_roi_fraction: float = 0.98
    max_center_shift_half_height_frac: float = 0.80
    max_endpoint_shift_roi_height_frac: float = 1.00
    max_angle_change_deg: float = 4.0
    min_candidate_count: int = 10
    min_candidate_span_frac: float = 0.20


DEFAULT_GATE = ROIGateConfig()


def line_y(line: HorizonLine, xs: np.ndarray) -> np.ndarray:
    return line.slope * xs + line.intercept


def roi_top_bottom(roi_pts: np.ndarray, xs: np.ndarray):
    lt, rt, rb, lb = roi_pts
    alpha = (xs - lt[0]) / max(float(rt[0] - lt[0]), 1.0)
    top = lt[1] + alpha * (rt[1] - lt[1])
    bottom = lb[1] + alpha * (rb[1] - lb[1])
    return np.minimum(top, bottom), np.maximum(top, bottom)


def inside_fraction(line: HorizonLine, roi_pts: np.ndarray, samples: int = 128) -> float:
    xs = np.linspace(0, line.width - 1, samples)
    top, bottom = roi_top_bottom(roi_pts, xs)
    ys = line_y(line, xs)
    return float(np.mean((ys >= top) & (ys <= bottom)))


def apply_bounded_roi(frame, coarse: HorizonLine, enable_roi: bool = True, enable_gate: bool = True, cfg: ROIGateConfig = DEFAULT_GATE):
    if not enable_roi:
        return {"existing_refined": None, "final": coarse, "accepted": False, "reason": "roi_disabled", "roi_pts": None, "padding": None, "roi_height": None, "inside_roi_fraction": np.nan, "center_correction": np.nan, "endpoint_correction": np.nan, "angle_correction": np.nan, "candidate_count": 0, "candidate_span": 0.0}
    refined, roi_pts, padding, roi_h, _, points = refine_existing(frame, coarse)
    candidate_count = 0 if points is None else int(len(points))
    candidate_span = 0.0 if points is None or len(points) == 0 else float(points[:, 1].max() - points[:, 1].min())
    if refined is None:
        return {"existing_refined": None, "final": coarse, "accepted": False, "reason": "invalid_refined_fit", "roi_pts": roi_pts, "padding": padding, "roi_height": roi_h, "inside_roi_fraction": 0.0, "center_correction": np.nan, "endpoint_correction": np.nan, "angle_correction": np.nan, "candidate_count": candidate_count, "candidate_span": candidate_span}
    inside = inside_fraction(refined, roi_pts)
    center_corr = abs(refined.y_center - coarse.y_center)
    endpoint_corr = max(abs(refined.y_left - coarse.y_left), abs(refined.y_right - coarse.y_right))
    angle_corr = abs(refined.theta_deg - coarse.theta_deg)
    reason = "accepted"
    if enable_gate:
        if inside < cfg.min_inside_roi_fraction:
            reason = "refined_outside_roi"
        elif center_corr > cfg.max_center_shift_half_height_frac * float(padding):
            reason = "center_shift_too_large"
        elif endpoint_corr > cfg.max_endpoint_shift_roi_height_frac * float(roi_h):
            reason = "endpoint_shift_too_large"
        elif angle_corr > cfg.max_angle_change_deg:
            reason = "angle_change_too_large"
        elif candidate_count < cfg.min_candidate_count:
            reason = "insufficient_candidates"
        elif candidate_span < cfg.min_candidate_span_frac * float(coarse.width):
            reason = "candidate_span_too_small"
    final = refined if reason == "accepted" or not enable_gate else coarse
    return {"existing_refined": refined, "final": final, "accepted": reason == "accepted" or not enable_gate, "reason": "gate_disabled" if not enable_gate else reason, "roi_pts": roi_pts, "padding": padding, "roi_height": roi_h, "inside_roi_fraction": inside, "center_correction": center_corr, "endpoint_correction": endpoint_corr, "angle_correction": angle_corr, "candidate_count": candidate_count, "candidate_span": candidate_span}
