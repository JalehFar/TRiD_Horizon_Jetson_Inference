from __future__ import annotations

import cv2
import numpy as np

import config
from inference.geometry import HorizonLine, line_from_center_angle

try:
    import numba
except Exception:  # pragma: no cover - optional deployment acceleration
    numba = None

ESSLD_ORIGINAL_HOUGH_THRESHOLD = 50
ESSLD_ORIGINAL_HOUGH_MIN_LINE_LENGTH = 60
ESSLD_ORIGINAL_HOUGH_MAX_LINE_GAP = 100


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
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    if num_labels < 2:
        return None

    largest = 1
    max_area = 0
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > max_area:
            max_area = stats[i, cv2.CC_STAT_AREA]
            largest = i
    if max_area < 100:
        return None

    clean = np.zeros_like(mask)
    clean[labels == largest] = 255
    repaired = np.zeros_like(clean)

    try:
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest_contour)
            cv2.fillPoly(repaired, [hull], 255)
        else:
            repaired = clean
    except Exception:
        repaired = clean

    points = []
    mask_i = repaired.astype(np.int16)
    for col in range(0, w, 4):
        column = mask_i[:, col]
        diff = np.diff(column)
        transitions = np.where(np.abs(diff) > 100)[0]
        if len(transitions) > 0:
            y_pos = transitions[0]
            if 5 < y_pos < h - 5:
                points.append([col, y_pos])
    if len(points) < 10:
        return None
    pts = np.asarray(points)

    try:
        k, b = robust_fit_line(pts, cv2.DIST_HUBER)
        if k is None:
            return None
        pred_y = k * pts[:, 0] + b
        diff = np.abs(pts[:, 1] - pred_y)
        inliers = pts[diff < 5.0]
        if len(inliers) > 10:
            k, b = np.polyfit(inliers[:, 0], inliers[:, 1], 1)
        return float(k * (w / 2) + b), float(np.rad2deg(np.arctan(k)))
    except Exception:
        return None


def _non_max_suppression_original_core(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    height, width = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    angle = angle * 180.0 / np.pi
    for i in range(height):
        for j in range(width):
            if angle[i, j] < 0:
                angle[i, j] += 180.0
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            q, r = 255.0, 255.0
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q, r = magnitude[i, j + 1], magnitude[i, j - 1]
            elif 22.5 <= a < 67.5:
                q, r = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif 67.5 <= a < 112.5:
                q, r = magnitude[i + 1, j], magnitude[i - 1, j]
            elif 112.5 <= a < 157.5:
                q, r = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0
    return suppressed


if numba is not None:
    _non_max_suppression_original_core = numba.jit(nopython=True)(_non_max_suppression_original_core)


def non_max_suppression_original(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    return _non_max_suppression_original_core(magnitude, angle)


def dual_fusion_pipeline_original(roi_image: np.ndarray):
    if roi_image is None:
        return None
    gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    edge_accum = np.zeros_like(gray, dtype=np.float32)
    for k, weight in zip(config.MEDIAN_FILTER_SIZES, config.CANNY_FUSION_WEIGHTS):
        blurred = cv2.medianBlur(gray, k)
        edge = cv2.Canny(blurred, 50, 150)
        edge_accum += (edge / 255.0) * weight
    edge_norm = cv2.normalize(edge_accum, None, 0, 1, cv2.NORM_MINMAX)

    h, w = roi_image.shape[:2]
    center_y = h // 2
    y_grid = np.arange(h).reshape(-1, 1)
    dist_map = np.abs(y_grid - center_y)

    conf_accum = np.zeros_like(gray, dtype=np.float32)
    for sigma, weight in zip(config.CONFIDENCE_MAP_SIGMAS, config.CONFIDENCE_MAP_WEIGHTS):
        conf = np.exp(-0.5 * (dist_map / sigma) ** 2)
        conf = np.tile(conf, (1, w))
        conf_accum += (edge_norm * conf) * weight

    sobelx = cv2.Sobel(conf_accum, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(conf_accum, cv2.CV_64F, 0, 1, ksize=5)
    angle = np.arctan2(sobely, sobelx)
    nms_map = non_max_suppression_original(conf_accum, angle)

    binary_map = (nms_map * 255 > config.FUSED_MAP_FINAL_THRESHOLD).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, 4, cv2.CV_32S)
    if num_labels > 1:
        cleaned_map = np.zeros_like(binary_map)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= 50:
                cleaned_map[labels == i] = 255
        binary_map = cleaned_map

    lines = cv2.HoughLinesP(
        binary_map,
        1,
        np.pi / 180,
        ESSLD_ORIGINAL_HOUGH_THRESHOLD,
        ESSLD_ORIGINAL_HOUGH_MIN_LINE_LENGTH,
        ESSLD_ORIGINAL_HOUGH_MAX_LINE_GAP,
    )
    if lines is not None:
        mask = np.zeros_like(binary_map)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
        binary_map = cv2.bitwise_and(binary_map, binary_map, mask=mask)

    points = np.argwhere(binary_map > 0)
    if len(points) < 10:
        return None

    pts_xy = np.column_stack((points[:, 1], points[:, 0]))

    try:
        weights = conf_accum[points[:, 0], points[:, 1]]
        if np.sum(weights) > 1e-6:
            coeffs = np.polyfit(pts_xy[:, 0], pts_xy[:, 1], 1, w=weights)
            k, b = coeffs[0], coeffs[1]
        else:
            k, b = robust_fit_line(pts_xy, cv2.DIST_L12)
        if k is None:
            return None
        angle = np.rad2deg(np.arctan(k))
        y_mid = k * (w / 2) + b
        return y_mid, angle
    except Exception:
        return None


def create_roi_original(frame: np.ndarray, y_coarse: float, angle_coarse: float, padding: int):
    h, w = frame.shape[:2]
    angle_rad = np.deg2rad(np.clip(angle_coarse, -89.9, 89.9))
    slope = np.tan(angle_rad)
    intercept = y_coarse - slope * (w / 2)
    y_pad = padding / (abs(np.cos(angle_rad)) + 1e-6)
    p1 = [0, intercept - y_pad]
    p2 = [w - 1, slope * (w - 1) + intercept - y_pad]
    p3 = [w - 1, slope * (w - 1) + intercept + y_pad]
    p4 = [0, intercept + y_pad]
    src_pts = np.float32([p1, p2, p3, p4])
    roi_h = int(2 * padding)
    if roi_h <= 0:
        return None, None, None, None
    dst_pts = np.float32([[0, 0], [w - 1, 0], [w - 1, roi_h - 1], [0, roi_h - 1]])
    m = cv2.getPerspectiveTransform(src_pts, dst_pts)
    m_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    roi = cv2.warpPerspective(frame, m, (w, roi_h))
    return roi, m, m_inv, (roi_h, w), src_pts


def refine_essld_original(frame: np.ndarray, coarse: HorizonLine, grad_score: float = 80):
    padding = calculate_dynamic_padding(grad_score)
    roi_img, _m, m_inv, roi_dims, roi_pts = create_roi_original(frame, coarse.y_center, coarse.theta_deg, padding)
    if roi_img is None:
        return None, roi_pts, padding, None
    refined_local = dual_fusion_pipeline_original(roi_img)
    if refined_local is None:
        return None, roi_pts, padding, roi_dims[0]
    local_y, local_angle = refined_local
    final_angle = coarse.theta_deg + local_angle
    pt_roi = np.array([[[roi_dims[1] / 2, local_y]]], dtype=np.float32)
    pt_global = cv2.perspectiveTransform(pt_roi, m_inv)
    final_y = pt_global[0][0][1]
    return line_from_center_angle(float(final_y), float(final_angle), coarse.width, coarse.height), roi_pts, padding, roi_dims[0]


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
