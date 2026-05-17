"""Crop-wise captioning pipeline: detector → tracker → per-crop Moondream caption.

Architecturally different from QwenYoloPipeline: the VLM never sees the full
frame and never produces bboxes. The detector is the sole source of localisation;
Moondream is queried once per active track at a configurable stride and only
has to describe the contents of a single crop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from vista.pipeline.base import Detection, FrameResult, VistaPipeline


# VistaCrash class name → canonical submission category
# (submission spec: car / emergency_vehicle / person).
_CATEGORY_MAP = {
    "crashed car": "car",
    "crashed_car": "car",
    "car":         "car",
    "person":      "person",
    # leave room for future detectors that include an emergency_vehicle class
    "emergency_vehicle": "emergency_vehicle",
    "ambulance":         "emergency_vehicle",
    "fire truck":        "emergency_vehicle",
    "police":            "emergency_vehicle",
}


def _normalize_category(raw: str) -> str:
    return _CATEGORY_MAP.get(raw.lower().strip(), raw)


def _pad_crop(bbox, W: int, H: int, pad: float) -> tuple[int, int, int, int]:
    """Pad a bbox by `pad` fraction (e.g. 0.1 = 10% on each side), clipped to image."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    x1 -= w * pad
    x2 += w * pad
    y1 -= h * pad
    y2 += h * pad
    return (max(0, int(x1)), max(0, int(y1)),
            min(W, int(x2)), min(H, int(y2)))


class MoondreamCaptioner:
    """Minimal wrapper around Moondream's caption / query HF API.

    Loads ``AutoModelForCausalLM`` with ``trust_remote_code=True`` and exposes
    a single ``caption(pil_crop) -> str`` method used by the pipeline.
    """

    def __init__(
        self,
        model_id: str = "vikhyatk/moondream2",
        revision: str | None = None,
        device: str = "cuda",
        prompt: str | None = None,
        max_new_tokens: int = 48,
    ) -> None:
        kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": {"": device}}
        if revision is not None:
            kwargs["revision"] = revision
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.device = device
        self.prompt = prompt or (
            "In one short phrase (under 10 words), describe the state and role "
            "of this object in an accident scene. Use terms like 'crashed', "
            "'injured, sitting', 'running', 'helping', 'emergency vehicle' "
            "when applicable."
        )
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def caption(self, crop: Image.Image, prompt: str | None = None) -> str:
        out = self.model.query(crop, prompt or self.prompt)
        # moondream returns {"answer": "..."} for query, {"caption": "..."} for caption()
        text = out.get("answer") or out.get("caption") or ""
        return text.strip()


class CropwiseMoondreamPipeline(VistaPipeline):
    """Detector + tracker (Ultralytics) + Moondream per-crop captioner.

    Args:
        detector:           An Ultralytics-style model with ``.track(img, persist=True)``.
                            Typically an RT-DETR or YOLO checkpoint fine-tuned on VistaCrash.
        captioner:          Object with ``.caption(pil_crop) -> str``.
        caption_stride:     Run the captioner every N frames. Captions are
                            propagated between calls via ``_track_db``.
        caption_padding:    Bbox padding fraction when cropping (e.g. 0.1 = 10%).
        det_conf:           Detector confidence threshold (passed to ``track``).
        recaption_every:    If set, force re-captioning a track at most every
                            ``recaption_every`` calls even if it already has one.
                            None means caption only when missing.
    """

    def __init__(
        self,
        detector: Any,
        captioner: Any,
        caption_stride: int = 30,
        caption_padding: float = 0.1,
        det_conf: float | None = 0.25,
        recaption_every: int | None = None,
        tracker: str | None = None,
        prompts: dict | None = None,
        imgsz: int | None = None,
    ) -> None:
        self.detector = detector
        self.captioner = captioner
        self.caption_stride = caption_stride
        self.caption_padding = caption_padding
        self.det_conf = det_conf
        self.recaption_every = recaption_every
        self.tracker = tracker
        self.prompts = prompts or {}
        self.imgsz = imgsz

        self._track_db: dict[int, dict] = {}
        self._caption_call_idx: dict[int, int] = {}
        self._caption_best_conf: dict[int, float] = {}   # approach 2: best-conf tracking
        self._stride_counter = 0

    def reset(self) -> None:
        self._track_db.clear()
        self._caption_call_idx.clear()
        self._caption_best_conf.clear()
        self._stride_counter = 0
        if hasattr(self.detector, "predictor") and self.detector.predictor is not None:
            # ultralytics keeps tracker state on the predictor across calls;
            # nulling it triggers a fresh tracker on the next .track() call.
            try:
                self.detector.predictor.trackers = None
            except Exception:
                pass

    def forward(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        W, H = frame.size

        track_kwargs: dict[str, Any] = dict(persist=True, verbose=False, conf=self.det_conf)
        if self.tracker is not None:
            track_kwargs["tracker"] = self.tracker
        if self.imgsz is not None:
            track_kwargs["imgsz"] = self.imgsz
        results = self.detector.track(np.array(frame), **track_kwargs)[0]

        # ── 1. Active tracks from detector ────────────────────────────────
        active: dict[int, dict] = {}
        if results.boxes.id is not None:
            for box, tid, cls, conf in zip(
                results.boxes.xyxy,
                results.boxes.id,
                results.boxes.cls,
                results.boxes.conf,
            ):
                tid = int(tid.item())
                raw_cat = results.names.get(int(cls.item()), "unknown")
                cat = _normalize_category(raw_cat)
                prev = self._track_db.get(tid, {})
                active[tid] = {
                    "bbox":    box.cpu().numpy().tolist(),
                    "category": cat,
                    "caption":  prev.get("caption"),
                    "conf":    float(conf.item()),
                }

        # remove stale tracks
        for tid in set(self._track_db) - set(active):
            self._track_db.pop(tid, None)
            self._caption_call_idx.pop(tid, None)

        # ── 2a. Caption new tracks immediately (approach 1) ──────────────
        new_tids = set(active) - set(self._track_db)
        for tid in new_tids:
            tr = active[tid]
            x1, y1, x2, y2 = _pad_crop(tr["bbox"], W, H, self.caption_padding)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = frame.crop((x1, y1, x2, y2))
            try:
                prompt = self.prompts.get(tr["category"]) or None
                caption = self.captioner.caption(crop, prompt=prompt)
            except Exception:
                caption = None
            if caption:
                tr["caption"] = caption
                self._caption_best_conf[tid] = tr["conf"]

        # ── 2b. Stride-based recaptioning with best-frame selection ──────
        if frame_idx % self.caption_stride == 0 and active:
            self._stride_counter += 1
            for tid, tr in active.items():
                if tid in new_tids:
                    continue  # already captioned above
                needs = tr["caption"] is None
                if not needs and self.recaption_every is not None:
                    last = self._caption_call_idx.get(tid, -10**9)
                    needs = (self._stride_counter - last) >= self.recaption_every
                if not needs:
                    continue

                x1, y1, x2, y2 = _pad_crop(tr["bbox"], W, H, self.caption_padding)
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                crop = frame.crop((x1, y1, x2, y2))
                try:
                    prompt = self.prompts.get(tr["category"]) or None
                    caption = self.captioner.caption(crop, prompt=prompt)
                except Exception:
                    caption = None
                if caption:
                    # approach 2: only update if this frame has higher detector conf
                    current_conf = tr["conf"]
                    if current_conf >= self._caption_best_conf.get(tid, -1.0):
                        tr["caption"] = caption
                        self._caption_best_conf[tid] = current_conf
                    self._caption_call_idx[tid] = self._stride_counter

        # ── 3. Persist + emit ────────────────────────────────────────────
        for tid, tr in active.items():
            self._track_db[tid] = tr

        detections = [
            Detection(
                bbox=tuple(tr["bbox"]),
                category=tr["category"],
                confidence=tr.get("conf", 1.0),
                track_id=tid,
                caption=tr.get("caption"),
            )
            for tid, tr in self._track_db.items()
        ]
        return FrameResult(detections=detections, frame_idx=frame_idx)
