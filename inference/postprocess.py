from __future__ import annotations

import cv2
import numpy as np

import config
from inference.geometry import HorizonLine, line_from_center_angle


def robust_fit_line(points, dist_type=cv2.DIST_HUBER):
    if len(points) < 2:
        return None, None
    line = cv2.fitLine(np.asarray(points, dtype=np.float32), dist_type, 0, 0.01, 0.01).flatten()
    vx, vy, x0, y0 = line[0], line[1], line[2], line[3]
    if abs(vx) < 1e-5:
        vx = 1e-5
    k = vy / vx
    return float(k), float(y0 - k * x0)


def get_coarse_line_from_mask(mask: np.ndarray, original_frame: np.ndarray):
    h, w = mask.shape[:2]
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    if num_labels < 2:
        return None
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[largest, cv2.CC_STAT_AREA] < 100:
        return None
    clean = np.zeros_like(mask)
    clean[labels == largest] = 255
    repaired = np.zeros_like(clean)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.fillPoly(repaired, [cv2.convexHull(max(contours, key=cv2.contourArea))], 255)
    else:
        repaired = clean
    points = []
    mask_i = repaired.astype(np.int16)
    for col in range(0, w, 4):
        transitions = np.where(np.abs(np.diff(mask_i[:, col])) > 100)[0]
        if len(transitions) > 0 and 5 < transitions[0] < h - 5:
            points.append([col, int(transitions[0])])
    if len(points) < 10:
        return None
    pts = np.asarray(points)
    k, b = robust_fit_line(pts, cv2.DIST_HUBER)
    if k is None:
        return None
    pred_y = k * pts[:, 0] + b
    inliers = pts[np.abs(pts[:, 1] - pred_y) < 5.0]
    if len(inliers) > 10:
        k, b = np.polyfit(inliers[:, 0], inliers[:, 1], 1)
    return float(k * (w / 2) + b), float(np.rad2deg(np.arctan(k)))


def non_max_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Vectorized non-maximum suppression for the thin ROI confidence map."""

    mag = magnitude.astype(np.float32, copy=False)
    deg = np.rad2deg(angle).astype(np.float32, copy=False)
    deg[deg < 0] += 180.0

    padded = np.pad(mag, 1, mode="constant")
    center = padded[1:-1, 1:-1]
    east, west = padded[1:-1, 2:], padded[1:-1, :-2]
    south, north = padded[2:, 1:-1], padded[:-2, 1:-1]
    sw, ne = padded[2:, :-2], padded[:-2, 2:]
    nw, se = padded[:-2, :-2], padded[2:, 2:]

    out = np.zeros_like(mag, dtype=np.float32)
    mask0 = ((deg < 22.5) | (deg >= 157.5)) & (center >= east) & (center >= west)
    mask45 = (deg >= 22.5) & (deg < 67.5) & (center >= sw) & (center >= ne)
    mask90 = (deg >= 67.5) & (deg < 112.5) & (center >= south) & (center >= north)
    mask135 = (deg >= 112.5) & (deg < 157.5) & (center >= nw) & (center >= se)
    keep = mask0 | mask45 | mask90 | mask135
    out[keep] = center[keep]
    return out


def calculate_dynamic_padding(grad_score: float = 80) -> int:
    return int(40 - np.clip((grad_score - 25.0) / (200.0 - 25.0), 0, 1) * 20)


def create_roi(frame: np.ndarray, coarse: HorizonLine, padding: int, target_width: int | None = None):
    h, w = frame.shape[:2]
    angle = np.deg2rad(np.clip(coarse.theta_deg, -89.9, 89.9))
    slope = np.tan(angle)
    intercept = coarse.y_center - slope * (w / 2)
    y_pad = padding / (abs(np.cos(angle)) + 1e-6)
    src = np.float32([[0, intercept - y_pad], [w - 1, slope * (w - 1) + intercept - y_pad], [w - 1, slope * (w - 1) + intercept + y_pad], [0, intercept + y_pad]])
    roi_h = int(2 * padding)
    roi_w = int(target_width or w)
    roi_w = max(32, min(roi_w, w))
    dst = np.float32([[0, 0], [roi_w - 1, 0], [roi_w - 1, roi_h - 1], [0, roi_h - 1]])
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    return cv2.warpPerspective(frame, m, (roi_w, roi_h)), m_inv, src, roi_h


def dual_fusion_pipeline(roi_image: np.ndarray):
    if roi_image is None:
        return None, None, None
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(gray)
    edge_accum = np.zeros_like(gray, dtype=np.float32)
    for k, weight in zip(config.MEDIAN_FILTER_SIZES, config.CANNY_FUSION_WEIGHTS):
        edge_accum += (cv2.Canny(cv2.medianBlur(gray, k), 50, 150) / 255.0) * weight
    edge_norm = cv2.normalize(edge_accum, None, 0, 1, cv2.NORM_MINMAX)
    h, w = gray.shape
    dist = np.abs(np.arange(h).reshape(-1, 1) - h // 2)
    conf_accum = np.zeros_like(gray, dtype=np.float32)
    for sigma, weight in zip(config.CONFIDENCE_MAP_SIGMAS, config.CONFIDENCE_MAP_WEIGHTS):
        conf_accum += edge_norm * np.tile(np.exp(-0.5 * (dist / sigma) ** 2), (1, w)) * weight
    nms = non_max_suppression(conf_accum, np.arctan2(cv2.Sobel(conf_accum, cv2.CV_64F, 0, 1, ksize=5), cv2.Sobel(conf_accum, cv2.CV_64F, 1, 0, ksize=5)))
    binary = (nms * 255 > config.FUSED_MAP_FINAL_THRESHOLD).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 4, cv2.CV_32S)
    if num_labels > 1:
        cleaned = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 50:
                cleaned[labels == i] = 255
        binary = cleaned
    lines = cv2.HoughLinesP(binary, 1, np.pi / 180, config.HOUGH_THRESHOLD, config.HOUGH_MIN_LINE_LENGTH, config.HOUGH_MAX_LINE_GAP)
    if lines is not None:
        mask = np.zeros_like(binary)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
        binary = cv2.bitwise_and(binary, binary, mask=mask)
    pts_yx = np.argwhere(binary > 0)
    if len(pts_yx) < 10:
        return None, binary, pts_yx
    pts = np.column_stack((pts_yx[:, 1], pts_yx[:, 0]))
    weights = conf_accum[pts_yx[:, 0], pts_yx[:, 1]]
    if np.sum(weights) > 1e-6:
        k, b = np.polyfit(pts[:, 0], pts[:, 1], 1, w=weights)
    else:
        k, b = robust_fit_line(pts, cv2.DIST_L12)
    if k is None:
        return None, binary, pts_yx
    return (float(k * (w / 2) + b), float(np.rad2deg(np.arctan(k)))), binary, pts_yx


def refine_existing(frame: np.ndarray, coarse: HorizonLine, roi_width: int | None = None):
    padding = calculate_dynamic_padding(80)
    roi, m_inv, roi_pts, roi_h = create_roi(frame, coarse, padding, target_width=roi_width or config.ROI_PROCESS_WIDTH)
    local, binary, points = dual_fusion_pipeline(roi)
    if local is None:
        return None, roi_pts, padding, roi_h, binary, points
    local_y, local_angle = local
    local_slope = float(np.tan(np.deg2rad(local_angle)))
    local_intercept = float(local_y) - local_slope * (float(roi.shape[1]) / 2.0)
    local_pts = np.array(
        [
            [[0.0, local_intercept]],
            [[float(roi.shape[1] - 1), local_slope * float(roi.shape[1] - 1) + local_intercept]],
        ],
        dtype=np.float32,
    )
    global_pts = cv2.perspectiveTransform(local_pts, m_inv).reshape(2, 2)
    dx = float(global_pts[1, 0] - global_pts[0, 0])
    if abs(dx) < 1e-6:
        return None, roi_pts, padding, roi_h, binary, points
    slope = float((global_pts[1, 1] - global_pts[0, 1]) / dx)
    intercept = float(global_pts[0, 1] - slope * global_pts[0, 0])
    refined = HorizonLine(intercept, slope * float(coarse.width - 1) + intercept, coarse.width, coarse.height)
    return refined, roi_pts, padding, roi_h, binary, points
