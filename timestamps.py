import cv2
import argparse

def format_seconds(total_seconds):
    """Convert seconds to mm:ss string."""
    m = total_seconds // 60
    s = total_seconds % 60
    return f"{m:02d}:{s:02d}"

def format_timestamps(timestamps):
    segments = []
    for i in range(0, len(timestamps) - 1, 2):
        start = format_seconds(timestamps[i])
        end   = format_seconds(timestamps[i + 1])
        segments.append(f"{start}-{end}")
    return " ".join(segments)

def draw_overlay(frame, timestamps,):
    out = frame.copy()
    if timestamps:
        seg_str = format_timestamps(timestamps)
        (tw, th), _ = cv2.getTextSize(seg_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (6, 4), (14 + tw, 14 + th), (20, 20, 20), -1)
        cv2.putText(out, seg_str, (10, 10 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 100), 1, cv2.LINE_AA)
 
    return out

def get_timestamps(source_path):
    output_file = "timestamps.txt"
    start_frame = 0
    timestamps = []

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {source_path}")

    fps       = cap.get(cv2.CAP_PROP_FPS) or 30
    end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"Playing back {source_path}")
    print("Controls: S/Space=debut  E=fin  B=reculer 5s  Q/ESC=quitter")

    current_frame = start_frame
    is_recording  = False

    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        display = draw_overlay(frame, timestamps)
        cv2.imshow("Playback", display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):                          # q ou ESC → quitter
            break

        elif key == ord("b"):                              # reculer 5 s
            new_pos = max(start_frame, current_frame - int(5 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            current_frame = new_pos

        elif key in (ord("s"), ord(" ")) and not is_recording:   # début segment
            is_recording = True
            timestamps.append(int(current_frame / fps))

        elif key == ord("e") and is_recording:             # fin segment
            is_recording = False
            timestamps.append(int(current_frame / fps))

    cap.release()
    cv2.destroyAllWindows()

    # ── Résultat final ───────────────────────────────────────────────
    if timestamps:
        if len(timestamps) % 2 == 1:
            timestamps.append(int(current_frame / fps))
        result = format_timestamps(timestamps)
        print("\nTimestamps :", result)
        with open(output_file, "w") as f:
            f.write(result)
        return result

    else:
        print("\nAucun timestamp enregistré.")
        return []

def main():
    ap = argparse.ArgumentParser(description="Outputs a list of timestamps")
    ap.add_argument("-v", "--video", default=None, help="Source video path")

    args = ap.parse_args()
    input_file = args.video or input("Input file (path/to/vid.mp4): ")
    get_timestamps(input_file)


if __name__ == "__main__":
    main()
