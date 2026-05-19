"""Render an annotated mp4 from a saved detections.json + source video.

Reads <out>/detections.json (produced by diagnose_detector.py or diagnose_sahi.py)
and writes <out>/annotated.mp4 with boxes colored by category.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


_COLORS = {
    "person":      (0, 255, 0),
    "crashed car": (0, 0, 255),
    "car":         (255, 128, 0),
}
_DEFAULT = (200, 200, 200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--detections", required=True, help="detections.json path")
    ap.add_argument("--out", required=True, help="output mp4 path")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="only draw boxes with conf >= this")
    args = ap.parse_args()

    with open(args.detections) as f:
        dets = json.load(f)
    frame_to_dets = {d["frame"]: d["detections"] for d in dets}
    start_frame = min(frame_to_dets) if frame_to_dets else 0
    end_frame = max(frame_to_dets) + 1 if frame_to_dets else 0

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H))

    idx = start_frame
    while idx < end_frame:
        ok, bgr = cap.read()
        if not ok:
            break
        for d in frame_to_dets.get(idx, []):
            if d["conf"] < args.min_conf:
                continue
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            color = _COLORS.get(d["category"], _DEFAULT)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{d['category']} {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(bgr, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        writer.write(bgr)
        idx += 1

    cap.release()
    writer.release()
    print(f"[render] wrote {out_path}  ({idx - start_frame} frames)")


if __name__ == "__main__":
    main()
