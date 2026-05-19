"""Shared adapters for feeding non-ultralytics detection outputs into the
ultralytics BYTETracker and back out into the Results-like shape that
``CropwiseMoondreamPipeline`` consumes.
"""

from __future__ import annotations

import numpy as np


class DetInput:
    """Minimal indexable namespace that ultralytics BYTETracker accepts as `results`."""
    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return DetInput(self.xywh[idx], self.conf[idx], self.cls[idx])


class Boxes:
    """Subset of ultralytics Results.boxes that the cropwise pipeline reads."""
    def __init__(self, tracks: np.ndarray):
        import torch
        if tracks.size == 0:
            self.id = None
            self.xyxy = torch.zeros((0, 4))
            self.cls = torch.zeros((0,))
            self.conf = torch.zeros((0,))
        else:
            self.id = torch.from_numpy(tracks[:, 4].astype(np.int64))
            self.xyxy = torch.from_numpy(tracks[:, :4].astype(np.float32))
            self.conf = torch.from_numpy(tracks[:, 5].astype(np.float32))
            self.cls = torch.from_numpy(tracks[:, 6].astype(np.float32))


class ResultsLike:
    def __init__(self, tracks: np.ndarray, names: dict):
        self.boxes = Boxes(tracks)
        self.names = names


def xyxy_to_xywh(xyxy: np.ndarray) -> np.ndarray:
    """Convert (N, 4) xyxy to (N, 4) xywh (center-x, center-y, w, h)."""
    if not len(xyxy):
        return np.zeros((0, 4), dtype=np.float32)
    xywh = np.zeros_like(xyxy, dtype=np.float32)
    xywh[:, 0] = (xyxy[:, 0] + xyxy[:, 2]) / 2
    xywh[:, 1] = (xyxy[:, 1] + xyxy[:, 3]) / 2
    xywh[:, 2] = xyxy[:, 2] - xyxy[:, 0]
    xywh[:, 3] = xyxy[:, 3] - xyxy[:, 1]
    return xywh


def load_tracker(cfg_path: str, frame_rate: int = 30):
    """Load an ultralytics tracker (ByteTrack or BoT-SORT) from a YAML path.

    Dispatch is by ``tracker_type`` inside the yaml.
    """
    import yaml
    from ultralytics.utils import IterableSimpleNamespace
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    args = IterableSimpleNamespace(**raw)
    ttype = raw.get("tracker_type", "bytetrack").lower()
    if ttype == "botsort":
        from ultralytics.trackers.bot_sort import BOTSORT
        return BOTSORT(args, frame_rate=frame_rate)
    from ultralytics.trackers.byte_tracker import BYTETracker
    return BYTETracker(args, frame_rate=frame_rate)


# Backwards-compatible alias (older code paths)
load_bytetracker = load_tracker
