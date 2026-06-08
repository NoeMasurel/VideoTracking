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
    """Return a list of {street, start, end} dicts from a flat list of seconds."""
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

def draw_overlay(frame, timestamps, annotations, current_annotation, waiting_for_annotation, is_paused):
    out = frame.copy()

    is_recording = len(timestamps) % 2 == 1

    if timestamps:
        seg_str = format_timestamps(timestamps)
        
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


    if is_paused and not waiting_for_annotation:
        h, w = out.shape[:2]
        pause_str = "|| PAUSED"
        (pw, ph), _ = cv2.getTextSize(pause_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        x = w - pw - 14
        cv2.rectangle(out, (x - 6, 4), (x + pw + 6, 14 + ph), (20, 20, 20), -1)
        cv2.putText(out, pause_str, (x, 10 + ph),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2, cv2.LINE_AA)

    # Annotation input prompt
    if waiting_for_annotation:
        prompt = f"Street name : {current_annotation}_"
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
    is_paused = False

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"Playing back {source_path}")
    print("Controls: S/Space=start  E=end  P=pause/play  B=back 5s  N=forward 5s  Q/ESC=quit")

    current_frame = start_frame
    is_recording  = False
    last_display  = None   # hold the last rendered frame while paused

    while cap.isOpened() and current_frame < end_frame:

        # Paused (or waiting for annotation): redisplay last frame, poll keys ──
        if is_paused or waiting_for_annotation:
            if last_display is not None:
                display = draw_overlay(last_display, timestamps, annotations,
                                       current_annotation, waiting_for_annotation, is_paused)
                cv2.imshow("Playback", display)

            key = cv2.waitKey(30) & 0xFF  # 30 ms poll so the window stays responsive

            # Annotation input mode
            if waiting_for_annotation:
                if key == 255:
                    continue
                if key in (13, 10):  # Enter
                    if len(current_annotation) == 2 and current_annotation[0].isalpha() and current_annotation[1].isdigit():
                        annotations.append(current_annotation.upper())
                        current_annotation = ""
                        waiting_for_annotation = False
                        # Stay paused after annotation so user can review before resuming
                        is_paused = False
                elif key == 8:  # Backspace
                    current_annotation = current_annotation[:-1]
                elif len(current_annotation) < 2 and 32 <= key <= 126:
                    ch = chr(key)
                    if len(current_annotation) == 0 and ch.isalpha():
                        current_annotation += ch
                    elif len(current_annotation) == 1 and ch.isdigit():
                        current_annotation += ch
                continue  # block other controls while waiting for annotation

            # Paused (no annotation needed): only allow a subset of keys
            if key == 255:
                continue

            match key:
                case 113 | 27:  # q / ESC
                    break
                case 112: # p : resume
                    is_paused = False
                case 32: # Space : resume (toggle)
                    is_paused = False
                case 98: # b : back 5 s (also works while paused)
                    new_pos = max(start_frame, current_frame - int(5 * fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    current_frame = new_pos
                    ret, frame = cap.read()
                    if ret:
                        last_display = frame
                        current_frame += 1
                case 110:  # n : forward 5 s
                    new_pos = min(end_frame, current_frame + int(5 * fps))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                    current_frame = new_pos
                    ret, frame = cap.read()
                    if ret:
                        last_display = frame
                        current_frame += 1
            continue  # don't advance to next frame while paused

        # Normal playback: read next frame
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1
        last_display = frame

        display = draw_overlay(frame, timestamps, annotations,
                               current_annotation, waiting_for_annotation, is_paused)
        cv2.imshow("Playback", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 255:
            continue

        # Normal playback controls
        match key:
            case 113 | 27: # q or ESC
                break

            case 112: # p : pause
                is_paused = True

            case 98: # b : back 5 s
                new_pos = max(start_frame, current_frame - int(5 * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                current_frame = new_pos

            case 110: # n : forward 5 s
                new_pos = min(end_frame, current_frame + int(5 * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                current_frame = new_pos

            case 115 | 32:  # s or space : start / restart segment
                if is_recording:
                    timestamps.append(int(current_frame / fps))
                    timestamps.append(int(current_frame / fps))
                    waiting_for_annotation = True   # closed a segment → auto-pause
                    is_paused = True
                else:
                    is_recording = True
                    timestamps.append(int(current_frame / fps))

            case 101:  # e : end segment
                if is_recording:
                    is_recording = False
                    timestamps.append(int(current_frame / fps))
                    waiting_for_annotation = True   # auto-pause for annotation
                    is_paused = True
                else:
                    if timestamps:
                        timestamps.append(timestamps[-1])
                        timestamps.append(int(current_frame / fps))
                        waiting_for_annotation = True
                        is_paused = True

            case _:
                pass

    cap.release()
    cv2.destroyAllWindows()

    if timestamps:
        if len(timestamps) % 2 == 1:  # close last open segment
            timestamps.append(int(current_frame / fps))

        # If the last clip still has no annotation, ask in the terminal
        while len(annotations) < len(timestamps) // 2:
            raw = input(f"Annotation for clip {len(annotations) + 1} (letter + digit, e.g. A1): ").strip()
            if len(raw) == 2 and raw[0].isalpha() and raw[1].isdigit():
                annotations.append(raw.upper())
            else:
                print(" : Must be one letter followed by one digit (e.g. A1, B3).")

        result = format_timestamps(timestamps)
        print("\nTimestamps :", result)

        pl = {
            "file": os.path.basename(source_path),
            "clips": build_clips(annotations, timestamps)
        }
        with open(timestamps_json, "w") as f:
            json.dump(pl, f, indent=2)

        print(f"Saved at {timestamps_json}")
        return result

    else:
        print("\nNo timestamps registered.")
        return []

def main():
    ap = argparse.ArgumentParser(description="Outputs a list of timestamps")
    ap.add_argument("-v", "--video", default="videos/2025_08_11/20250811.mp4", help="Source video path")
    ap.add_argument("-o", "--output", default="data/timestamps.json", help="Output file path")

    args = ap.parse_args()
    input_file = args.video or input("Input file (path/to/vid.mp4) : ")
    output_file = args.output
    get_timestamps(input_file, output_file)


if __name__ == "__main__":
    main()