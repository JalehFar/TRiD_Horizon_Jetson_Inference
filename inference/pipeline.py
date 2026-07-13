from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

import config
from inference.geometry import HorizonLine, is_valid_line, line_errors, line_from_center_angle
from inference.postprocess import get_coarse_line_from_mask
from inference.roi_gate import apply_bounded_roi
from models.dceunet import DCEUNet
from models.heads import DirectRegHL, DSACHL, TRiDHorizon, WLSHL


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def cuda_time_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def preprocess(frame_bgr: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(frame_bgr, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0


def endpoints_norm_to_original(endpoints, width: int, height: int) -> HorizonLine:
    return HorizonLine(float(endpoints[0]) * float(height - 1), float(endpoints[1]) * float(height - 1), width, height)


def invalid_result(method: str, frame_index: int, timings: dict[str, float], gt: HorizonLine | None, reason: str) -> "FrameResult":
    timings.setdefault("postprocess_ms", 0.0)
    timings.setdefault("roi_ms", 0.0)
    timings["full_ms"] = timings.get("preprocess_ms", 0.0) + timings.get("model_ms", 0.0) + timings.get("postprocess_ms", 0.0) + timings.get("roi_ms", 0.0)
    return FrameResult(config.DISPLAY_NAMES[method], frame_index, None, None, None, False, False, reason, None, timings, {}, gt)


@dataclass
class FrameResult:
    method: str
    frame_index: int
    coarse: HorizonLine | None
    existing_refined: HorizonLine | None
    final: HorizonLine | None
    prediction_valid: bool
    roi_accepted: bool
    roi_reason: str
    roi_pts: np.ndarray | None
    timing: dict[str, float]
    roi_log: dict
    gt: HorizonLine | None = None

    def errors(self) -> dict[str, float]:
        if self.gt is None or self.final is None:
            return {"center_error": np.nan, "endpoint_error": np.nan, "angular_error": np.nan}
        return line_errors(self.final, self.gt)


class MethodRunner:
    def __init__(
        self,
        method: str,
        device: torch.device,
        fp16: bool = True,
        roi: bool = True,
        roi_gate: bool = True,
        roi_width: int | None = None,
        roi_every: int = 1,
    ):
        self.method = method
        self.device = device
        self.fp16 = bool(fp16 and device.type == "cuda")
        self.roi = roi
        self.roi_gate = roi_gate
        self.roi_width = int(roi_width or config.ROI_PROCESS_WIDTH)
        self.roi_every = max(1, int(roi_every))
        self.model = self._load_model()
        self.model.eval()
        if self.fp16:
            self.model.half()
        self.clip_buffer: list[torch.Tensor] = []
        meta = config.CHECKPOINTS[method]
        print(f"method={method} checkpoint={meta['path']} sha256={sha256_file(meta['path'])} device={device} precision={'fp16' if self.fp16 else 'fp32'} input={config.IMAGE_WIDTH}x{config.IMAGE_HEIGHT}")

    def reset(self) -> None:
        self.clip_buffer = []

    def _load_model(self):
        meta = config.CHECKPOINTS[self.method]
        path = meta["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        digest = sha256_file(path)
        if digest != meta["sha256"]:
            raise RuntimeError(f"Checkpoint SHA256 mismatch for {self.method}: {digest} != {meta['sha256']}")
        if self.method == "essld":
            model = DCEUNet(3, 1)
            state = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(state)
        else:
            cls = {"directreg": DirectRegHL, "wls": WLSHL, "dsac": lambda: DSACHL(64), "trid": lambda: TRiDHorizon(64, 64)}[self.method]
            model = cls()
            state = torch.load(path, map_location=self.device, weights_only=False)
            missing, unexpected = model.load_state_dict(state["model"], strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Incompatible checkpoint {path}: missing={missing}, unexpected={unexpected}")
        return model.to(self.device)

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray, frame_index: int, gt: HorizonLine | None = None) -> FrameResult:
        timings = {}
        t0 = time.perf_counter()
        tensor = preprocess(frame_bgr)
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0
        x = tensor.to(self.device)
        if self.fp16:
            x = x.half()
        model_start_wall = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
        if self.method == "essld":
            logits = self.model(x)
            if self.device.type == "cuda":
                ev1.record()
            prob = torch.sigmoid(logits).float().cpu().numpy()[0, 0]
            prob_orig = cv2.resize(prob, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = (prob_orig > 0.5).astype(np.uint8) * 255
            coarse_tuple = get_coarse_line_from_mask(mask, frame_bgr)
            coarse = None if coarse_tuple is None else line_from_center_angle(coarse_tuple[0], coarse_tuple[1], frame_bgr.shape[1], frame_bgr.shape[0])
        elif self.method == "trid":
            self.clip_buffer.append(x[0].detach().cpu())
            if len(self.clip_buffer) > config.CLIP_LENGTH:
                self.clip_buffer = self.clip_buffer[-config.CLIP_LENGTH :]
            seq = torch.stack(self.clip_buffer, dim=0).unsqueeze(0).to(self.device)
            if self.fp16:
                seq = seq.half()
            out = self.model(seq)
            if self.device.type == "cuda":
                ev1.record()
            endpoints = out["endpoints_norm01"][0, -1].float().cpu().numpy()
            coarse = endpoints_norm_to_original(endpoints, frame_bgr.shape[1], frame_bgr.shape[0])
        else:
            out = self.model(x)
            if self.device.type == "cuda":
                ev1.record()
            endpoints = out["endpoints_norm01"][0].float().cpu().numpy()
            coarse = endpoints_norm_to_original(endpoints, frame_bgr.shape[1], frame_bgr.shape[0])
        if self.device.type == "cuda":
            timings["model_ms"] = cuda_time_ms(ev0, ev1)
        else:
            timings["model_ms"] = (time.perf_counter() - model_start_wall) * 1000.0
        t_post = time.perf_counter()
        if not is_valid_line(coarse):
            timings["postprocess_ms"] = (time.perf_counter() - t_post) * 1000.0
            reason = "nonfinite_or_missing_coarse_line"
            return invalid_result(self.method, frame_index, timings, gt, reason)
        timings["postprocess_ms"] = (time.perf_counter() - t_post) * 1000.0
        t_roi = time.perf_counter()
        run_roi = self.roi and ((frame_index - 1) % self.roi_every == 0)
        roi_log = apply_bounded_roi(frame_bgr, coarse, run_roi, self.roi_gate, roi_width=self.roi_width)
        if self.roi and not run_roi:
            roi_log["reason"] = f"roi_skipped_every_{self.roi_every}"
        timings["roi_ms"] = (time.perf_counter() - t_roi) * 1000.0
        timings["full_ms"] = timings["preprocess_ms"] + timings["model_ms"] + timings["postprocess_ms"] + timings["roi_ms"]
        final = roi_log.get("final")
        if not is_valid_line(final):
            roi_log["final"] = coarse
            roi_log["accepted"] = False
            roi_log["reason"] = "nonfinite_final_fallback_to_coarse"
            final = coarse
        return FrameResult(config.DISPLAY_NAMES[self.method], frame_index, coarse, roi_log["existing_refined"], final, True, bool(roi_log["accepted"]), roi_log["reason"], roi_log["roi_pts"], timings, roi_log, gt)
