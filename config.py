from __future__ import annotations

from pathlib import Path

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512
CLIP_LENGTH = 8

MEDIAN_FILTER_SIZES = [1, 5, 7]
CANNY_FUSION_WEIGHTS = [0.5, 0.3, 0.2]
CONFIDENCE_MAP_SIGMAS = [10, 15, 20]
CONFIDENCE_MAP_WEIGHTS = [0.5, 0.3, 0.2]
FUSED_MAP_FINAL_THRESHOLD = 60
HOUGH_THRESHOLD = 30
HOUGH_MIN_LINE_LENGTH = 50
HOUGH_MAX_LINE_GAP = 20

CHECKPOINTS = {
    "essld": {
        "path": Path("weights/essld_dceunetex.pth"),
        "sha256": "4261c4e6d0f51b76f6101bcee336943994a614ae970cf8e57894266fb7d8da36",
        "class": "DCEUNet",
        "fine_tuned_backbone": False,
        "temporal": False,
    },
    "directreg": {
        "path": Path("weights/directreg_hl_best_full.pt"),
        "sha256": "0e1dc48afd259cbfa52e4b2e0fa0da2c740e0c992208c6e058845b6824253a89",
        "class": "DirectRegHL",
        "fine_tuned_backbone": True,
        "temporal": False,
    },
    "wls": {
        "path": Path("weights/wls_hl_best_full.pt"),
        "sha256": "b9e69b7a70b4c87caf0838ec69e35b286f77e5a0cdbeb3ec33f8adb802603a30",
        "class": "WLSHL",
        "fine_tuned_backbone": True,
        "temporal": False,
    },
    "dsac": {
        "path": Path("weights/dsac_hl_best_full.pt"),
        "sha256": "17634e52931f02db3c2ffc1c5e8da8951570e3dfc2f8e8675d9d50dfaa2d4cba",
        "class": "DSACHL",
        "fine_tuned_backbone": True,
        "temporal": False,
    },
    "trid": {
        "path": Path("weights/trid_horizon_best_visible_y95.pt"),
        "sha256": "bd74a14fa9dddc410a61f1b974743d8c31d4d8a767040e8c94bf3e12cc9e6c71",
        "class": "TRiDHorizon",
        "fine_tuned_backbone": True,
        "temporal": True,
    },
}

DISPLAY_NAMES = {
    "essld": "ESSLD",
    "directreg": "DirectReg-HL",
    "wls": "WLS-HL",
    "dsac": "DSAC-HL",
    "trid": "TRiD-Horizon",
}
