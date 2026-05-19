"""Multi-detector ensemble with per-detector class allowlists.

Use case: the fine-tuned RT-DETR knows the ``crashed car`` semantic but has a
weak ``person`` head (drone-altitude OOD); a COCO-trained YOLO has a strong
``person`` head but no notion of ``crashed car``. Combining them with a
role-split (each detector contributes only the classes it's good at) gives a
single detection stream calibrated enough to feed an unchanged tracker.

Detections from each sub-detector are filtered to its allowed class set + min
conf, remapped to a unified class id space, concatenated, NMS-deduplicated
per class, and finally tracked by ultralytics BYTETracker — exposing the same
``.track()`` facade ``CropwiseMoondreamPipeline`` already uses.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torchvision.ops import nms

from vista.models._track_io import (
    DetInput, ResultsLike, load_bytetracker, xyxy_to_xywh,
)


class _SubDetector:
    """One detector inside an EnsembleDetector."""

    def __init__(
        self,
        weights: str,
        det_type: str,
        allowed_classes: list[str],
        min_conf: float,
        imgsz: int | None = None,
        device: str = "cuda:0",
    ) -> None:
        from ultralytics import RTDETR, YOLO
        cls = RTDETR if det_type.lower() == "rtdetr" else YOLO
        self.model = cls(weights)
        self.allowed = set(allowed_classes)
        self.min_conf = min_conf
        self.imgsz = imgsz
        self.device = device
        # name -> local class id, frozen once at construction
        self.names: dict = dict(self.model.names)
        self.allowed_ids = {i for i, n in self.names.items() if n in self.allowed}

    def predict(self, image: np.ndarray):
        """Return (xyxy, conf, names) numpy arrays after class+conf filtering."""
        kwargs: dict[str, Any] = dict(verbose=False, conf=self.min_conf,
                                      device=self.device)
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz
        r = self.model.predict(image, **kwargs)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.array([], dtype=object))
        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = r.boxes.conf.cpu().numpy().astype(np.float32)
        cls_local = r.boxes.cls.cpu().numpy().astype(np.int64)
        mask = np.array([cid in self.allowed_ids for cid in cls_local])
        if not mask.any():
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.array([], dtype=object))
        xyxy, conf, cls_local = xyxy[mask], conf[mask], cls_local[mask]
        names = np.array([self.names[int(c)] for c in cls_local], dtype=object)
        return xyxy, conf, names


class EnsembleDetector:
    """Combine several detectors with class-role splits, then track.

    Args:
        detectors:  list of dicts, each with keys
                    ``weights``, ``type``, ``classes`` (allowlist),
                    ``conf`` (min), optional ``imgsz``.
        unified_names: dict mapping unified-class-name -> unified-class-id.
                       Detections whose normalized name is not in this map are
                       dropped.
        device:     CUDA device.
        nms_iou:    IoU threshold for per-class NMS across detectors.
    """

    DEFAULT_UNIFIED = {"person": 0, "car": 1, "crashed car": 2,
                       "emergency_vehicle": 3}

    def __init__(
        self,
        detectors: list[dict],
        unified_names: dict[str, int] | None = None,
        device: str = "cuda:0",
        nms_iou: float = 0.5,
    ) -> None:
        self.unified_names = unified_names or self.DEFAULT_UNIFIED
        self.id_to_name = {v: k for k, v in self.unified_names.items()}
        self.nms_iou = nms_iou
        self.subs = [
            _SubDetector(
                weights=d["weights"],
                det_type=d.get("type", "yolo"),
                allowed_classes=d["classes"],
                min_conf=d.get("conf", 0.30),
                imgsz=d.get("imgsz"),
                device=device,
            )
            for d in detectors
        ]

        self._tracker = None
        self._tracker_cfg_path = None

    # ── tracker plumbing ────────────────────────────────────────────────────

    def _ensure_tracker(self, cfg_path: str):
        if self._tracker is None or self._tracker_cfg_path != cfg_path:
            self._tracker = load_bytetracker(cfg_path)
            self._tracker_cfg_path = cfg_path

    def _reset_tracker(self):
        self._tracker = None
        self._tracker_cfg_path = None

    # ── public interface ────────────────────────────────────────────────────

    @property
    def names(self):
        return self.id_to_name

    @property
    def predictor(self):
        return SimpleNamespace(trackers=None)

    def track(
        self,
        image: np.ndarray,
        persist: bool = True,
        conf: float | None = None,
        imgsz: int | None = None,
        tracker: str | None = None,
        verbose: bool = False,
        **_: Any,
    ):
        if not persist:
            self._reset_tracker()
        if tracker is None:
            raise ValueError(
                "EnsembleDetector needs a tracker config path "
                "(detector.tracker: config/trackers/bytetrack_long.yaml)"
            )
        self._ensure_tracker(tracker)

        # 1. Run each sub-detector, collect (xyxy, conf, unified_cls_id)
        xyxy_all, conf_all, cls_all = [], [], []
        for sub in self.subs:
            sxyxy, sconf, snames = sub.predict(image)
            if not len(sxyxy):
                continue
            unified = np.array(
                [self.unified_names.get(n, -1) for n in snames], dtype=np.int64
            )
            keep = unified >= 0
            if not keep.any():
                continue
            xyxy_all.append(sxyxy[keep])
            conf_all.append(sconf[keep])
            cls_all.append(unified[keep])

        if not xyxy_all:
            tracks = self._tracker.update(
                DetInput(np.zeros((0, 4), dtype=np.float32),
                         np.zeros((0,), dtype=np.float32),
                         np.zeros((0,), dtype=np.float32)),
                img=image,
            )
            return [ResultsLike(tracks, self.id_to_name)]

        xyxy = np.concatenate(xyxy_all, axis=0)
        conf_arr = np.concatenate(conf_all, axis=0)
        cls_arr = np.concatenate(cls_all, axis=0).astype(np.float32)

        # 2. Per-class NMS to dedupe between detectors (and within)
        keep_idx = []
        for cid in np.unique(cls_arr):
            mask = cls_arr == cid
            if not mask.any():
                continue
            idxs = np.flatnonzero(mask)
            t_box = torch.from_numpy(xyxy[idxs])
            t_scr = torch.from_numpy(conf_arr[idxs])
            k = nms(t_box, t_scr, self.nms_iou).cpu().numpy()
            keep_idx.append(idxs[k])
        keep_idx = np.concatenate(keep_idx) if keep_idx else np.zeros((0,), dtype=np.int64)
        xyxy = xyxy[keep_idx]
        conf_arr = conf_arr[keep_idx]
        cls_arr = cls_arr[keep_idx]

        # 3. Optional global conf gate
        if conf is not None and len(conf_arr):
            m = conf_arr >= conf
            xyxy, conf_arr, cls_arr = xyxy[m], conf_arr[m], cls_arr[m]

        det_ns = DetInput(xywh=xyxy_to_xywh(xyxy), conf=conf_arr, cls=cls_arr)
        tracks = self._tracker.update(det_ns, img=image)
        return [ResultsLike(tracks, self.id_to_name)]
