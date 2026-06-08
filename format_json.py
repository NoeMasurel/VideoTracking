import cv2
import argparse
import json
import os

def format_seconds(total_seconds):
    """Convert seconds to mm:ss string."""
    m = total_seconds // 60
    s = total_seconds % 60
    return f"{m:02d}:{s:02d}"

def format_timestamps(timestamps):
    """Returns a string of timestamps in the format xx:xx-xx:xx separated by spaces"""
    segments = []
    for i in range(0, len(timestamps), 2):
        start = format_seconds(timestamps[i])
        end = format_seconds(timestamps[i + 1]) if i + 1 < len(timestamps) else "??:??"
        segments.append(f"{start}-{end}")
    return " ".join(segments)

def build_clips(annotations, timestamps):
    """Return a list of {annotation, start, end, annotation} dicts from a flat list of seconds."""
    clips = []
    for i in range(0, len(timestamps) - 1, 2):
        clip_index = i // 2

        clip = {
            "street" : annotations[clip_index],
            "start": format_seconds(timestamps[i]),
            "end": format_seconds(timestamps[i + 1]),
        }
        clips.append(clip)
    return clips

def draw_overlay(frame, timestamps, annotations, current_annotation, waiting_for_annotation):
    out = frame.copy()

    is_recording = len(timestamps) % 2 == 1

    if timestamps:
        seg_str = format_timestamps(timestamps)
        # Append annotations to completed segments
        parts = seg_str.split(" ")
        labeled = []
        for idx, part in enumerate(parts):
            ann = annotations[idx] if idx < len(annotations) else None
            labeled.append(f"{part}[{ann}]" if ann else part)
        seg_str = " ".join(labeled)

        (tw, th), _ = cv2.getTextSize(seg_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (6, 4), (14 + tw, 14 + th), (20, 20, 20), -1)
        color = (0, 100, 230) if is_recording else (0, 230, 100)
        cv2.putText(out, seg_str, (10, 10 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Annotation input prompt
    if waiting_for_annotation:
        prompt = f"Annotation: {current_annotation}_"
        (pw, ph), _ = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        h, w = out.shape[:2]
        x, y = (w - pw) // 2, h // 2
        cv2.rectangle(out, (x - 10, y - ph - 10), (x + pw + 10, y + 10), (20, 20, 20), -1)
        cv2.rectangle(out, (x - 10, y - ph - 10), (x + pw + 10, y + 10), (0, 200, 255), 1)
        cv2.putText(out, prompt, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

    return out

def get_timestamps(source_path, output):
    timestamps_json = output
    start_frame = 0
    timestamps  = []
    annotations = []          # one entry per completed clip
    current_annotation = ""   # being typed right now
    waiting_for_annotation = False

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"Playing back {source_path}")
    print("Controls: S/Space=start  E=end  B=back 5s  N=forward 5s  Q/ESC=quit")

    current_frame = start_frame
    is_recording  = False

    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        display = draw_overlay(frame, timestamps, annotations, current_annotation, waiting_for_annotation)
        cv2.imshow("Playback", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue

        # ── Annotation input mode ────────────────────────────────────
        if waiting_for_annotation:
            if key in (13, 10):  # Enter — only accept if input is valid
                if len(current_annotation) == 2 and current_annotation[0].isalpha() and current_annotation[1].isdigit():
                    annotations.append(current_annotation.upper())
                    current_annotation = ""
                    waiting_for_annotation = False
            elif key == 8:  # Backspace
                current_annotation = current_annotation[:-1]
            elif len(current_annotation) < 2 and 32 <= key <= 126:
                ch = chr(key)
                if len(current_annotation) == 0 and ch.isalpha():
                    current_annotation += ch
                elif len(current_annotation) == 1 and ch.isdigit():
                    current_annotation += ch
            continue  # block all other controls while waiting

        # ── Normal playback controls ─────────────────────────────────
        match key:
            case 113 | 27:  # q or ESC
                break

            case 98:  # b : reculer 5 s
                new_pos = max(start_frame, current_frame - int(5 * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                current_frame = new_pos

            case 110:  # n : avancer de 5 s
                new_pos = min(end_frame, current_frame + int(5 * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                current_frame = new_pos

            case 115 | 32:  # s or space : start / restart segment
                if is_recording:
                    timestamps.append(int(current_frame / fps))
                    timestamps.append(int(current_frame / fps))
                    waiting_for_annotation = True   # closed a segment, need annotation
                else:
                    is_recording = True
                    timestamps.append(int(current_frame / fps))

            case 101:  # e : end segment
                if is_recording:
                    is_recording = False
                    timestamps.append(int(current_frame / fps))
                    waiting_for_annotation = True   # need annotation before continuing
                else:
                    if timestamps:
                        timestamps.append(timestamps[-1])
                        timestamps.append(int(current_frame / fps))
                        waiting_for_annotation = True

            case _:
                pass

    cap.release()
    cv2.destroyAllWindows()

    if timestamps:
        if len(timestamps) % 2 == 1:  # close last open segment
            timestamps.append(int(current_frame / fps))

        # If the last clip still has no annotation, ask in the terminal
        while len(annotations) < len(timestamps) // 2:
            raw = input(f"Annotation for clip {len(annotations) + 1} (letter+digit, e.g. A1): ").strip()
            if len(raw) == 2 and raw[0].isalpha() and raw[1].isdigit():
                annotations.append(raw.upper())
            else:
                print("  → Must be one letter followed by one digit (e.g. A1, B3).")

        result = format_timestamps(timestamps)
        print("\nTimestamps :", result)

        pl = {
            "file": os.path.basename(source_path),
            "clips": build_clips(annotations, timestamps)
        }
        with open(timestamps_json, "w") as f:
            json.dump(pl, f, indent=2)

        print(f"Saved → {timestamps_json}")
        return result

    else:
        print("\nAucun timestamp enregistré.")
        return []

def main():
    ap = argparse.ArgumentParser(description="Outputs a list of timestamps")
    ap.add_argument("-v", "--video", default="videos/2025_08_11/20250811.mp4", help="Source video path")
    ap.add_argument("-o", "--output", default="data/timestamps.json", help="Output file path")

    args = ap.parse_args()
    input_file = args.video or input("Input file (path/to/vid.mp4): ")
    output_file = args.output
    get_timestamps(input_file, output_file)


if __name__ == "__main__":
    main()