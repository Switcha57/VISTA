"""Detector-only diagnostic: did the fine-tuned RT-DETR ever see the fallen person?

Runs the fine-tuned RT-DETR on a video at a low confidence threshold with NO
tracker and NO captioner, then writes:

  * <out>/annotated.mp4 — green = person, red = crashed car, blue = car
  * <out>/detections.json — per-frame list of {bbox, category, conf}
  * <out>/summary.csv — per-frame counts by category (quick scan for misses)

Usage:
    python scripts/diagnose_detector.py \
        --weights out/rtdetr_vistacrash/weights/best.pt \
        --video data/_smoke/DJI_20251120172410_0001_S.mp4 \
        --out out/diag/smoke_detector_only \
        --conf 0.05

Interpreting the output for the bike-fall case:
  * If 'person' detections never appear in frames where the person is clearly
    on the ground -> detector miss (out-of-distribution prone pose).
  * If 'person' detections appear sporadically with conf < new_track_thresh
    (0.50 in bytetrack_long.yaml) -> tracker drops them. Tune the tracker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import RTDETR


_COLORS = {
    "person":      (0, 255, 0),    # green
    "crashed car": (0, 0, 255),    # red
    "car":         (255, 128, 0),  # blue-ish
}
_DEFAULT_COLOR = (200, 200, 200)


def _draw(frame_bgr, dets):
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        color = _COLORS.get(d["category"], _DEFAULT_COLOR)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{d['category']} {d['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return frame_bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="RT-DETR .pt checkpoint")
    ap.add_argument("--video", required=True, help="Input video path")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--conf", type=float, default=0.05,
                    help="Detection confidence threshold (low on purpose)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diag] loading {args.weights}")
    model = RTDETR(args.weights)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[diag] video {W}x{H} @ {fps:.2f} fps")

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    writer = cv2.VideoWriter(
        str(out_dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H),
    )

    per_frame = []
    summary_rows = []
    frame_idx = args.start_frame
    while True:
        if args.end_frame is not None and frame_idx >= args.end_frame:
            break
        ok, bgr = cap.read()
        if not ok:
            break

        res = model.predict(
            bgr, conf=args.conf, imgsz=args.imgsz,
            device=args.device, verbose=False,
        )[0]

        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            for box, cls, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                dets.append({
                    "bbox": [float(v) for v in box.cpu().numpy().tolist()],
                    "category": res.names.get(int(cls.item()), "unknown"),
                    "conf": float(conf.item()),
                })

        per_frame.append({"frame": frame_idx, "detections": dets})

        counts = {"person": 0, "crashed car": 0, "car": 0, "other": 0}
        max_conf = {"person": 0.0, "crashed car": 0.0, "car": 0.0}
        for d in dets:
            key = d["category"] if d["category"] in counts else "other"
            counts[key] += 1
            if d["category"] in max_conf:
                max_conf[d["category"]] = max(max_conf[d["category"]], d["conf"])
        summary_rows.append({
            "frame": frame_idx,
            **{f"n_{k.replace(' ', '_')}": v for k, v in counts.items()},
            **{f"maxconf_{k.replace(' ', '_')}": v for k, v in max_conf.items()},
        })

        writer.write(_draw(bgr, dets))
        if frame_idx % 30 == 0:
            print(f"[diag] frame {frame_idx}: {counts}")
        frame_idx += 1

    cap.release()
    writer.release()

    with open(out_dir / "detections.json", "w") as f:
        json.dump(per_frame, f)

    with open(out_dir / "summary.csv", "w") as f:
        if summary_rows:
            cols = list(summary_rows[0].keys())
            f.write(",".join(cols) + "\n")
            for r in summary_rows:
                f.write(",".join(str(r[c]) for c in cols) + "\n")

    n_person_frames = sum(1 for r in summary_rows if r["n_person"] > 0)
    print(f"[diag] done. {len(summary_rows)} frames, "
          f"person detected in {n_person_frames} ({100*n_person_frames/max(1,len(summary_rows)):.1f}%)")
    print(f"[diag] outputs in {out_dir}")


if __name__ == "__main__":
    main()
