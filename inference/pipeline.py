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
from models.heads import DirectRegHL, DSACHL, TRiDHorizon, WLSHL, heatmap_to_points


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
    timings["full_ms"] = timings.get("preprocess_ms", 0.0) + timings.get("h2d_ms", 0.0) + timings.get("model_ms", 0.0) + timings.get("postprocess_ms", 0.0) + timings.get("roi_ms", 0.0)
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
        trid_mode: str = "chunk",
    ):
        self.method = method
        self.device = device
        self.fp16 = bool(fp16 and device.type == "cuda")
        self.roi = roi
        self.roi_gate = roi_gate
        self.roi_width = int(roi_width or config.ROI_PROCESS_WIDTH)
        self.roi_every = max(1, int(roi_every))
        self.trid_mode = trid_mode
        self.model = self._load_model()
        self.model.eval()
        if self.fp16:
            self.model.half()
        self.clip_buffer: list[torch.Tensor] = []
        self.stream_hidden: torch.Tensor | None = None
        self.stream_history_length = 0
        self.stream_pending_reset = True
        meta = config.CHECKPOINTS[method]
        extra = f" trid_mode={self.trid_mode}" if method == "trid" else ""
        print(f"method={method} checkpoint={meta['path']} sha256={sha256_file(meta['path'])} device={device} precision={'fp16' if self.fp16 else 'fp32'} input={config.IMAGE_WIDTH}x{config.IMAGE_HEIGHT}{extra}")

    def reset(self) -> None:
        self.clip_buffer = []
        self.reset_stream_state()

    def reset_stream_state(self) -> None:
        self.stream_hidden = None
        self.stream_history_length = 0
        self.stream_pending_reset = True

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

    def _time_op(self, fn):
        start_wall = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
            out = fn()
            ev1.record()
            return out, cuda_time_ms(ev0, ev1)
        return fn(), (time.perf_counter() - start_wall) * 1000.0

    def _finish_from_coarse(
        self,
        frame_bgr: np.ndarray,
        frame_index: int,
        gt: HorizonLine | None,
        coarse: HorizonLine | None,
        timings: dict[str, float],
        roi_log_extra: dict | None = None,
    ) -> FrameResult:
        t_post = time.perf_counter()
        if not is_valid_line(coarse):
            timings["postprocess_ms"] = (time.perf_counter() - t_post) * 1000.0
            reason = "nonfinite_or_missing_coarse_line"
            return invalid_result(self.method, frame_index, timings, gt, reason)
        timings["postprocess_ms"] = (time.perf_counter() - t_post) * 1000.0
        t_roi = time.perf_counter()
        run_roi = self.roi and ((frame_index - 1) % self.roi_every == 0)
        roi_log = apply_bounded_roi(frame_bgr, coarse, run_roi, self.roi_gate, roi_width=self.roi_width)
        if roi_log_extra:
            roi_log.update(roi_log_extra)
        if self.roi and not run_roi:
            roi_log["reason"] = f"roi_skipped_every_{self.roi_every}"
        timings["roi_ms"] = (time.perf_counter() - t_roi) * 1000.0
        timings["full_ms"] = timings["preprocess_ms"] + timings.get("h2d_ms", 0.0) + timings["model_ms"] + timings["postprocess_ms"] + timings["roi_ms"]
        final = roi_log.get("final")
        if not is_valid_line(final):
            roi_log["final"] = coarse
            roi_log["accepted"] = False
            roi_log["reason"] = "nonfinite_final_fallback_to_coarse"
            final = coarse
        return FrameResult(config.DISPLAY_NAMES[self.method], frame_index, coarse, roi_log["existing_refined"], final, True, bool(roi_log["accepted"]), roi_log["reason"], roi_log["roi_pts"], timings, roi_log, gt)

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray, frame_index: int, gt: HorizonLine | None = None) -> FrameResult:
        timings = {}
        t0 = time.perf_counter()
        tensor = preprocess(frame_bgr)
        timings["preprocess_ms"] = (time.perf_counter() - t0) * 1000.0
        x, h2d_ms = self._time_op(lambda: tensor.to(self.device))
        timings["h2d_ms"] = h2d_ms
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
            if self.trid_mode == "streaming":
                coarse, stream_timings, stream_extra = self.forward_stream_step(x, frame_bgr.shape[1], frame_bgr.shape[0])
                timings.update(stream_timings)
                timings["model_ms"] = (
                    timings.get("dce_backbone_ms", 0.0)
                    + timings.get("convgru_step_ms", 0.0)
                    + timings.get("reduce_ms", 0.0)
                    + timings.get("column_head_ms", 0.0)
                    + timings.get("dsac_ms", 0.0)
                )
                return self._finish_from_coarse(frame_bgr, frame_index, gt, coarse, timings, stream_extra)
            if self.trid_mode == "single":
                seq_cpu = [x[0].detach().cpu()]
            else:
                self.clip_buffer.append(x[0].detach().cpu())
                if len(self.clip_buffer) > config.CLIP_LENGTH:
                    self.clip_buffer = self.clip_buffer[-config.CLIP_LENGTH :]
                seq_cpu = self.clip_buffer
            seq = torch.stack(seq_cpu, dim=0).unsqueeze(0).to(self.device)
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
        return self._finish_from_coarse(frame_bgr, frame_index, gt, coarse, timings)

    @torch.no_grad()
    def forward_stream_step(self, x: torch.Tensor, width: int, height: int) -> tuple[HorizonLine, dict[str, float], dict]:
        hidden_was_reset = self.stream_hidden is None
        feats, dce_ms = self._time_op(lambda: self.model.backbone(x))
        enhanced = feats["enhanced"]

        def convgru_step():
            self.stream_hidden = self.model.temporal.cell(enhanced, self.stream_hidden)
            return self.stream_hidden

        hidden, convgru_ms = self._time_op(convgru_step)
        reduced, reduce_ms = self._time_op(lambda: self.model.reduce(hidden))
        cols, column_ms = self._time_op(lambda: self.model.column_head(reduced))

        def dsac_step():
            x_cols, y_cols, conf = heatmap_to_points(cols["heat_logits"], cols["confidence_logits"])
            fit = self.model.dsac(x_cols, y_cols, conf)
            return ((fit["endpoints_norm"] + 1.0) * 0.5).clamp(0.0, 1.0)

        endpoints_tensor, dsac_ms = self._time_op(dsac_step)
        endpoints = endpoints_tensor[0].float().cpu().numpy()
        self.stream_hidden = hidden.detach()
        self.stream_history_length += 1
        self.stream_pending_reset = False
        timings = {
            "dce_backbone_ms": dce_ms,
            "convgru_step_ms": convgru_ms,
            "reduce_ms": reduce_ms,
            "column_head_ms": column_ms,
            "dsac_ms": dsac_ms,
        }
        extra = {
            "trid_mode": "streaming",
            "effective_history_length": self.stream_history_length,
            "backbone_evaluations_this_frame": 1,
            "hidden_state_reset": int(hidden_was_reset),
            "cache_hit": 0,
            "clip_id": "",
        }
        return endpoints_norm_to_original(endpoints, width, height), timings, extra

    @torch.no_grad()
    def predict_trid_chunk(self, items: list[tuple[int, np.ndarray, HorizonLine | None]]) -> list[FrameResult]:
        if self.method != "trid":
            raise RuntimeError("predict_trid_chunk is only valid for TRiD-Horizon")
        if not items:
            return []

        frame_indices = [item[0] for item in items]
        frames = [item[1] for item in items]
        gts = [item[2] for item in items]
        actual_count = len(items)

        t0 = time.perf_counter()
        tensors = [preprocess(frame)[0] for frame in frames]
        while len(tensors) < config.CLIP_LENGTH:
            tensors.append(tensors[-1].clone())
        seq_cpu = torch.stack(tensors[: config.CLIP_LENGTH], dim=0)
        preprocess_ms_total = (time.perf_counter() - t0) * 1000.0

        seq, h2d_ms_total = self._time_op(lambda: seq_cpu.unsqueeze(0).to(self.device))
        if self.fp16:
            seq = seq.half()
        model_start_wall = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
        out = self.model(seq)
        if self.device.type == "cuda":
            ev1.record()
            model_ms_total = cuda_time_ms(ev0, ev1)
        else:
            model_ms_total = (time.perf_counter() - model_start_wall) * 1000.0

        endpoints_all = out["endpoints_norm01"][0, :actual_count].float().cpu().numpy()
        results: list[FrameResult] = []
        preprocess_each = preprocess_ms_total / float(actual_count)
        h2d_each = h2d_ms_total / float(actual_count)
        model_each = model_ms_total / float(config.CLIP_LENGTH)
        for local_i, (frame_index, frame, gt, endpoints) in enumerate(zip(frame_indices, frames, gts, endpoints_all)):
            timings = {"preprocess_ms": preprocess_each, "h2d_ms": h2d_each, "model_ms": model_each}
            coarse = endpoints_norm_to_original(endpoints, frame.shape[1], frame.shape[0])
            extra = {
                "trid_mode": "chunk",
                "trid_chunk_length": config.CLIP_LENGTH,
                "trid_chunk_position": local_i,
                "trid_actual_chunk_frames": actual_count,
                "trid_backbone_evals_total": config.CLIP_LENGTH,
                "trid_backbone_evals_amortized": 1.0,
                "chunk_model_latency_ms": model_ms_total,
                "amortized_chunk_compute_per_frame_ms": model_each,
                "backbone_evaluations_this_frame": "",
                "hidden_state_reset": 1 if local_i == 0 else 0,
                "cache_hit": 0,
                "clip_id": frame_indices[0],
            }
            results.append(self._finish_from_coarse(frame, frame_index, gt, coarse, timings, extra))
        return results
