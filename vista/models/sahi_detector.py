"""SAHI-sliced detector + ultralytics BYTETracker, exposing the ultralytics
``.track()`` interface so it slots into CropwiseMoondreamPipeline as a drop-in
detector replacement.

Motivation: at drone altitude, persons occupy ~30 px in a 1920-wide frame.
Both the fine-tuned RT-DETR and COCO RT-DETR-L give them very low confidence
(median ~0.13, max ~0.7) in this regime. SAHI tiles the frame into smaller
crops so the detector sees each person at a much larger relative scale, which
typically lifts confidence into the well-calibrated band (>0.7).

The class wraps a SAHI ``AutoDetectionModel`` and an ultralytics ``BYTETracker``
so we can keep the existing pipeline + tracker config unchanged.
"""

from __future__ import annotations

from typing import Any
from types import SimpleNamespace

import numpy as np

from vista.models._track_io import (
    DetInput, ResultsLike, load_tracker, xyxy_to_xywh,
)


class SahiTrackingDetector:
    """SAHI sliced inference + ultralytics BYTETracker behind a ``.track()`` facade.

    Args:
        weights:        Path to an ultralytics-compatible checkpoint (.pt).
        model_type:     SAHI model type. "ultralytics" supports both YOLO and
                        RT-DETR in modern SAHI; falls back to "rtdetr"/"yolov8".
        slice_height:   Tile height in pixels.
        slice_width:    Tile width in pixels.
        overlap:        Overlap fraction between tiles (0.2 = 20%).
        device:         "cuda:0", "cpu", etc.
        sahi_conf:      SAHI's internal conf threshold. Keep low so the tracker
                        can use its own thresholds.
        postprocess_iou: IoU threshold for SAHI's post-slice merge (NMS).
    """

    def __init__(
        self,
        weights: str,
        model_type: str = "ultralytics",
        slice_height: int = 640,
        slice_width: int = 640,
        overlap: float = 0.2,
        device: str = "cuda:0",
        sahi_conf: float = 0.05,
        postprocess_iou: float = 0.5,
    ) -> None:
        from sahi import AutoDetectionModel

        self.sahi_model = AutoDetectionModel.from_pretrained(
            model_type=model_type,
            model_path=weights,
            confidence_threshold=sahi_conf,
            device=device,
        )
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap = overlap
        self.postprocess_iou = postprocess_iou

        # Pull class-name mapping from the underlying ultralytics model
        underlying = getattr(self.sahi_model, "model", None)
        self.names: dict = getattr(underlying, "names", {}) or {}

        self._tracker = None
        self._tracker_cfg_path = None

    # ── tracker plumbing ────────────────────────────────────────────────────

    def _ensure_tracker(self, cfg_path: str):
        if self._tracker is None or self._tracker_cfg_path != cfg_path:
            self._tracker = load_tracker(cfg_path)
            self._tracker_cfg_path = cfg_path

    def _reset_tracker(self):
        self._tracker = None
        self._tracker_cfg_path = None

    # ── ultralytics-compatible interface ────────────────────────────────────

    def track(
        self,
        image: np.ndarray,
        persist: bool = True,
        conf: float | None = None,
        imgsz: int | None = None,    # ignored: SAHI controls tile size
        tracker: str | None = None,
        verbose: bool = False,
        **_: Any,
    ):
        from sahi.predict import get_sliced_prediction

        if not persist:
            self._reset_tracker()
        if tracker is None:
            raise ValueError(
                "SahiTrackingDetector needs a tracker config path "
                "(e.g. detector.tracker: config/trackers/bytetrack_long.yaml)"
            )
        self._ensure_tracker(tracker)

        result = get_sliced_prediction(
            image,
            self.sahi_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap,
            overlap_width_ratio=self.overlap,
            perform_standard_pred=True,
            postprocess_type="GREEDYNMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=self.postprocess_iou,
            verbose=0,
        )

        preds = result.object_prediction_list
        if preds:
            xyxy = np.array(
                [[p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy] for p in preds],
                dtype=np.float32,
            )
            conf_arr = np.array([p.score.value for p in preds], dtype=np.float32)
            cls_arr = np.array([p.category.id for p in preds], dtype=np.float32)
            if conf is not None:
                mask = conf_arr >= conf
                xyxy, conf_arr, cls_arr = xyxy[mask], conf_arr[mask], cls_arr[mask]
        else:
            xyxy = np.zeros((0, 4), dtype=np.float32)
            conf_arr = np.zeros((0,), dtype=np.float32)
            cls_arr = np.zeros((0,), dtype=np.float32)

        det_ns = DetInput(xywh=xyxy_to_xywh(xyxy), conf=conf_arr, cls=cls_arr)
        tracks = self._tracker.update(det_ns, img=image)
        return [ResultsLike(tracks, self.names)]

    # ``CropwiseMoondreamPipeline.reset()`` peeks at ``predictor.trackers``; we
    # expose a no-op predictor so that branch is harmless.
    @property
    def predictor(self):  # noqa: D401 - simple property
        return SimpleNamespace(trackers=None)
