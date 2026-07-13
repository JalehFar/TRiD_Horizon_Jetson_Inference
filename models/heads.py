from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.dceunet import DCEUNet


class DCEBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = DCEUNet(input_channels=3, num_classes=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net.forward_features(x)


class DirectRegHead(nn.Module):
    def __init__(self, in_channels: int = 104, hidden_channels: int = 128):
        super().__init__()
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_channels, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, 2), nn.Sigmoid())

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)


class VisibilityHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_channels, hidden_channels), nn.ReLU(inplace=True), nn.Linear(hidden_channels, 1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat).squeeze(-1)


class HorizonColumnHead(nn.Module):
    def __init__(self, in_channels: int = 24, hidden_channels: int = 32, out_height: int = 256, out_width: int = 512):
        super().__init__()
        self.out_height = out_height
        self.out_width = out_width
        self.heat = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, 1, 1))
        self.conf = nn.Sequential(nn.Conv2d(in_channels, hidden_channels, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden_channels, 1, 1))

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        heat_logits = F.interpolate(self.heat(feat), size=(self.out_height, self.out_width), mode="bilinear", align_corners=False)
        conf_map = self.conf(feat).mean(dim=2, keepdim=True)
        conf_logits = F.interpolate(conf_map, size=(1, self.out_width), mode="bilinear", align_corners=False).squeeze(2)
        return {"heat_logits": heat_logits, "confidence_logits": conf_logits}


def endpoints_from_mb(m: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack([b - m, b + m], dim=-1)


def weighted_line_fit(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, ridge: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.float()
    y = y.float()
    weights = weights.float()
    if x.ndim == 1:
        x = x.unsqueeze(0).expand_as(y)
    w = weights.clamp_min(1e-6)
    ones = torch.ones_like(x)
    a00 = torch.sum(w * x * x, dim=-1) + ridge
    a01 = torch.sum(w * x, dim=-1)
    a11 = torch.sum(w * ones, dim=-1) + ridge
    rhs0 = torch.sum(w * x * y, dim=-1)
    rhs1 = torch.sum(w * y, dim=-1)
    det = (a00 * a11 - a01 * a01).clamp_min(1e-8)
    return (rhs0 * a11 - rhs1 * a01) / det, (a00 * rhs1 - a01 * rhs0) / det


def heatmap_to_points(logits: torch.Tensor, confidence_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = logits.float()
    confidence_logits = confidence_logits.float()
    bsz, _, height, width = logits.shape
    probs = torch.softmax(logits.squeeze(1), dim=1)
    y_grid = torch.linspace(-1.0, 1.0, height, device=logits.device, dtype=logits.dtype).view(1, height, 1)
    y = torch.sum(probs * y_grid, dim=1)
    conf = torch.sigmoid(confidence_logits.squeeze(1))
    if conf.shape[-1] != width:
        conf = F.interpolate(conf.unsqueeze(1), size=(width,), mode="linear", align_corners=False).squeeze(1)
    x = torch.linspace(-1.0, 1.0, width, device=logits.device, dtype=logits.dtype).view(1, width).expand(bsz, width)
    return x, y, conf


class DSACLineFit(nn.Module):
    def __init__(self, hypotheses: int = 64, temperature: float = 0.1, inlier_threshold: float = 0.05, min_column_delta: float = 0.05):
        super().__init__()
        self.hypotheses = int(hypotheses)
        self.temperature = float(temperature)
        self.inlier_threshold = float(inlier_threshold)
        self.min_column_delta = float(min_column_delta)

    def forward(self, x: torch.Tensor, y: torch.Tensor, confidence: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.float()
        y = y.float()
        confidence = confidence.float()
        bsz, width = x.shape
        probs = confidence.clamp_min(1e-6)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        idx1 = torch.multinomial(probs, self.hypotheses, replacement=True)
        idx2 = torch.multinomial(probs, self.hypotheses, replacement=True)
        x1, y1 = torch.gather(x, 1, idx1), torch.gather(y, 1, idx1)
        x2, y2 = torch.gather(x, 1, idx2), torch.gather(y, 1, idx2)
        too_close = (x2 - x1).abs() < self.min_column_delta
        idx2 = torch.where(too_close, (idx1 + max(2, width // 4)) % width, idx2)
        x2, y2 = torch.gather(x, 1, idx2), torch.gather(y, 1, idx2)
        denom = (x2 - x1).clamp_min(1e-6)
        denom = torch.where((x2 - x1) < 0, (x2 - x1).clamp_max(-1e-6), denom)
        m = (y2 - y1) / denom
        b = y1 - m * x1
        residual = (y.unsqueeze(1) - (m.unsqueeze(-1) * x.unsqueeze(1) + b.unsqueeze(-1))).abs()
        soft_inlier = torch.sigmoid((self.inlier_threshold - residual) / max(self.temperature, 1e-6))
        conf_score = torch.gather(confidence, 1, idx1) * torch.gather(confidence, 1, idx2)
        consensus = torch.sum(soft_inlier * confidence.unsqueeze(1), dim=-1) / confidence.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        scores = consensus + 0.25 * conf_score + 0.1 * torch.exp(-0.5 * (m / 0.75) ** 2)
        hyp_probs = torch.softmax(scores / max(self.temperature, 1e-6), dim=-1)
        m_final = torch.sum(hyp_probs * m, dim=-1)
        b_final = torch.sum(hyp_probs * b, dim=-1)
        return {"m": m_final, "b": b_final, "endpoints_norm": endpoints_from_mb(m_final, b_final)}


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(input_channels + hidden_channels, 2 * hidden_channels, kernel_size, padding=padding)
        self.candidate = nn.Conv2d(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> torch.Tensor:
        if h is None:
            h = torch.zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        combined = torch.cat([x, h], dim=1)
        z, r = torch.chunk(torch.sigmoid(self.gates(combined)), 2, dim=1)
        cand = torch.tanh(self.candidate(torch.cat([x, r * h], dim=1)))
        return (1.0 - z) * h + z * cand


class ConvGRU(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int):
        super().__init__()
        self.cell = ConvGRUCell(input_channels, hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = None
        outs = []
        for step in range(x.shape[1]):
            h = self.cell(x[:, step], h)
            outs.append(h)
        return torch.stack(outs, dim=1)


class DirectRegHL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DCEBackbone()
        self.reg_head = DirectRegHead(104)
        self.visibility_head = VisibilityHead(104)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        return {"endpoints_norm01": self.reg_head(feats["enhanced"]), "features": feats, "seg_logits": feats["seg_logits"]}


class WLSHL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = DCEBackbone()
        self.column_head = HorizonColumnHead(24)
        self.visibility_head = VisibilityHead(104)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        cols = self.column_head(feats["bottle"])
        x_cols, y_cols, conf = heatmap_to_points(cols["heat_logits"], cols["confidence_logits"])
        m, b = weighted_line_fit(x_cols, y_cols, conf)
        return {"endpoints_norm01": ((endpoints_from_mb(m, b) + 1.0) * 0.5).clamp(0.0, 1.0), "features": feats, "column_confidence": conf, "seg_logits": feats["seg_logits"]}


class DSACHL(nn.Module):
    def __init__(self, hypotheses: int = 64):
        super().__init__()
        self.backbone = DCEBackbone()
        self.column_head = HorizonColumnHead(24)
        self.dsac = DSACLineFit(hypotheses=hypotheses)
        self.visibility_head = VisibilityHead(104)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        cols = self.column_head(feats["bottle"])
        x_cols, y_cols, conf = heatmap_to_points(cols["heat_logits"], cols["confidence_logits"])
        fit = self.dsac(x_cols, y_cols, conf)
        return {"endpoints_norm01": ((fit["endpoints_norm"] + 1.0) * 0.5).clamp(0.0, 1.0), "features": feats, "column_confidence": conf, "seg_logits": feats["seg_logits"]}


class TRiDHorizon(nn.Module):
    def __init__(self, hidden_channels: int = 64, hypotheses: int = 64):
        super().__init__()
        self.backbone = DCEBackbone()
        self.temporal = ConvGRU(104, hidden_channels)
        self.reduce = nn.Conv2d(hidden_channels, 24, 1)
        self.column_head = HorizonColumnHead(24)
        self.dsac = DSACLineFit(hypotheses=hypotheses)
        self.visibility_head = VisibilityHead(hidden_channels)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bsz, steps, channels, height, width = x.shape
        flat = x.reshape(bsz * steps, channels, height, width)
        feats = self.backbone(flat)
        enhanced = feats["enhanced"].reshape(bsz, steps, *feats["enhanced"].shape[1:])
        temporal = self.temporal(enhanced)
        flat_temporal = temporal.reshape(bsz * steps, *temporal.shape[2:])
        cols = self.column_head(self.reduce(flat_temporal))
        x_cols, y_cols, conf = heatmap_to_points(cols["heat_logits"], cols["confidence_logits"])
        fit = self.dsac(x_cols, y_cols, conf)
        return {"endpoints_norm01": ((fit["endpoints_norm"] + 1.0) * 0.5).clamp(0.0, 1.0).reshape(bsz, steps, 2), "column_confidence": conf.reshape(bsz, steps, -1), "seg_logits": feats["seg_logits"].reshape(bsz, steps, *feats["seg_logits"].shape[1:])}
