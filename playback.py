"""
playback.py — Replay a video with bounding boxes from detections.json

Usage
    python playback.py # detections.json + source path stored inside
    python playback.py --json detections.json # explicit JSON path
    python playback.py --json detections.json --video path/to/video.mp4 # override video path
    python playback.py --json detections.json --speed 2.0 # 2x playback speed

Controls
    q / ESC   → quit
    SPACE     → pause / resume
    d         → step forward one frame (while paused)
"""
import cv2
import json
import argparse
from pathlib import Path

_PALETTE = [
    (0, 255, 200), (255, 100,   0), (  0, 100, 255), (200, 255,   0),
    (255,   0, 200), (  0, 200, 255), (255, 200,   0), (100,   0, 255),
    (  0, 255, 100), (255,  50,  50),
]

def _color(track_id: int):
    return _PALETTE[track_id % len(_PALETTE)]

def draw_bbox(frame, box, track_id, class_name, conf):
    x1, y1, x2, y2 = map(int, box)
    color = _color(track_id)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{class_name}:{track_id} {conf:.2f}"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    padding    = 6

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    bg_y1 = max(y1 - th - 2 * padding, 0)
    bg_y2 = y1
    bg_x2 = x1 + tw + 2 * padding

    cv2.rectangle(frame, (x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.putText(
        frame, label,
        (x1 + padding, bg_y2 - padding // 2 - baseline // 2 + th // 2),
        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
    )


def draw_hud(frame, frame_idx, total_frames, paused, class_counts):
    """Top-left semi-transparent overlay with counts + playback info."""
    lines = [f"Frame {frame_idx}/{total_frames}  {'[PAUSED]' if paused else ''}"]
    lines += [f"  {name}: {cnt}" for name, cnt in sorted(class_counts.items())]

    line_h  = 28
    padding = 8
    width   = 230
    height  = padding + len(lines) * line_h + padding

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        y = 8 + padding + i * line_h + line_h // 2 + 6
        cv2.putText(frame, line, (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 1, cv2.LINE_AA)

def main():
    ap = argparse.ArgumentParser(description="Replay video with saved bounding boxes.")
    ap.add_argument("-j", "--json",  default="data/detections.json", help="Path to detections JSON")
    ap.add_argument("-v","--video", default=None, help="Override source video path")
    ap.add_argument("-s", "--speed", default = 1, help="Playback speed")
    args = ap.parse_args()

    # Load JSON
    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    # Support both the new format (with "meta") and the old flat format
    if "meta" in data:
        meta        = data["meta"]
        detections  = data["detections"]
        source_path = args.video or meta["source"]
        fps         = meta["fps"]
        start_frame = meta["start_frame"]
        end_frame   = meta["end_frame"]
        class_names = {int(k): v for k, v in meta["class_names"].items()}
    else:
        # dict : {frame_str: [detection, ...]}
        detections  = data
        source_path = args.video
        if source_path is None:
            raise ValueError(
                "JSON has no 'meta' block. Pass --video path/to/video.mp4"
            )
        fps         = None
        start_frame = 0
        end_frame   = None
        class_names = {}  

    if source_path is None:
        raise ValueError("No video path. Use --video path/to/video.mp4")

    # Open video
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source_path}")

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
    if end_frame is None:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_analyzed = end_frame - start_frame
    delay_ms = max(1, int(1000 / (fps * args.speed)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    seen_ids     = set()
    class_counts = {}

    print(f"Playing back {source_path}")
    print(f"Frames {start_frame} to {end_frame}  |  {fps:.1f} FPS  |  speed × {args.speed}")
    print("Controls: SPACE=pause  d=step  q/ESC=quit")

    paused = False
    current_frame = start_frame

    while cap.isOpened() and current_frame < end_frame:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1

        # Draw detections for this frame (key : int or str)
        frame_key = str(current_frame)
        frame_dets = detections.get(frame_key, detections.get(current_frame, []))

        for det in frame_dets:
            track_id   = det.get("id", 0) or 0
            cls        = det.get("cls", 0)
            conf       = det.get("conf", 0.0)
            bbox       = det.get("bbox", [0, 0, 0, 0])
            class_name = class_names.get(cls, str(cls))

            # Update counts
            uid = (int(track_id), int(cls))
            if uid not in seen_ids:
                seen_ids.add(uid)
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

            draw_bbox(frame, bbox, int(track_id), class_name, conf) # type: ignore

        draw_hud(frame, current_frame - start_frame, total_analyzed, paused, class_counts) # type: ignore

        cv2.imshow("Playback", frame) # type: ignore

        key = cv2.waitKey(1 if paused else delay_ms) & 0xFF
        if key in (ord("q"), 27):          # q or ESC
            break
        elif key == ord(" "):              # SPACE : pause vid
            paused = not paused
        elif key == ord("d") and paused:   # d → step one frame while paused
            ret, frame = cap.read()
            if not ret:
                break
            current_frame += 1
            paused = True

    cap.release()
    cv2.destroyAllWindows()
    print("Playback finished.")


if __name__ == "__main__":
    main()
