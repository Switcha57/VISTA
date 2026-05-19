"""SAHI detection-only diagnostic, mirroring scripts/diagnose_detector.py.

Runs SAHI sliced inference over the same video with no tracker, dumps the
same detections.json layout so we can compare confidence histograms directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--slice", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=None)
    ap.add_argument("--model-type", default="ultralytics")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sahi] loading {args.weights} via {args.model_type}")
    model = AutoDetectionModel.from_pretrained(
        model_type=args.model_type,
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
    )
    names = getattr(getattr(model, "model", None), "names", {}) or {}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    per_frame, summary = [], []
    frame_idx = args.start_frame
    while True:
        if args.end_frame is not None and frame_idx >= args.end_frame:
            break
        ok, bgr = cap.read()
        if not ok:
            break

        res = get_sliced_prediction(
            bgr, model,
            slice_height=args.slice, slice_width=args.slice,
            overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
            perform_standard_pred=True,
            postprocess_type="GREEDYNMM",
            postprocess_match_metric="IOS",
            postprocess_match_threshold=0.5,
            verbose=0,
        )
        dets = []
        for p in res.object_prediction_list:
            dets.append({
                "bbox": [float(p.bbox.minx), float(p.bbox.miny),
                         float(p.bbox.maxx), float(p.bbox.maxy)],
                "category": p.category.name,
                "conf": float(p.score.value),
            })
        per_frame.append({"frame": frame_idx, "detections": dets})

        counts = {"person": 0, "crashed car": 0, "car": 0, "other": 0}
        max_conf = {"person": 0.0, "crashed car": 0.0, "car": 0.0}
        for d in dets:
            key = d["category"] if d["category"] in counts else "other"
            counts[key] += 1
            if d["category"] in max_conf:
                max_conf[d["category"]] = max(max_conf[d["category"]], d["conf"])
        summary.append({
            "frame": frame_idx,
            **{f"n_{k.replace(' ', '_')}": v for k, v in counts.items()},
            **{f"maxconf_{k.replace(' ', '_')}": v for k, v in max_conf.items()},
        })
        if frame_idx % 30 == 0:
            print(f"[sahi] frame {frame_idx}: {counts}  max_person_conf={max_conf['person']:.3f}")
        frame_idx += 1

    cap.release()

    with open(out_dir / "detections.json", "w") as f:
        json.dump(per_frame, f)
    with open(out_dir / "summary.csv", "w") as f:
        if summary:
            cols = list(summary[0].keys())
            f.write(",".join(cols) + "\n")
            for r in summary:
                f.write(",".join(str(r[c]) for c in cols) + "\n")

    print(f"[sahi] done, {len(summary)} frames -> {out_dir}")


if __name__ == "__main__":
    main()
