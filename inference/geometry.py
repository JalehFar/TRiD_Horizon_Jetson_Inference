from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HorizonLine:
    y_left: float
    y_right: float
    width: int
    height: int

    @property
    def slope(self) -> float:
        return (float(self.y_right) - float(self.y_left)) / max(float(self.width - 1), 1.0)

    @property
    def intercept(self) -> float:
        return float(self.y_left)

    @property
    def y_center(self) -> float:
        return self.slope * (float(self.width) / 2.0) + self.intercept

    @property
    def theta_deg(self) -> float:
        return math.degrees(math.atan(self.slope))

    def scaled(self, out_width: int, out_height: int) -> "HorizonLine":
        scale = float(out_height) / float(self.height)
        return HorizonLine(self.y_left * scale, self.y_right * scale, out_width, out_height)


def line_from_center_angle(y_center: float, theta_deg: float, width: int, height: int) -> HorizonLine:
    slope = math.tan(math.radians(float(theta_deg)))
    intercept = float(y_center) - slope * (float(width) / 2.0)
    return HorizonLine(intercept, slope * float(width - 1) + intercept, int(width), int(height))


def draw_line(frame: np.ndarray, line: HorizonLine, color: tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    out = frame.copy()
    cv2.line(out, (0, int(round(line.y_left))), (line.width - 1, int(round(line.y_right))), color, thickness, cv2.LINE_AA)
    return out


def line_errors(pred: HorizonLine, gt: HorizonLine) -> dict[str, float]:
    left = abs(pred.y_left - gt.y_left)
    right = abs(pred.y_right - gt.y_right)
    return {"center_error": abs(pred.y_center - gt.y_center), "endpoint_error": 0.5 * (left + right), "angular_error": abs(pred.theta_deg - gt.theta_deg)}
