import ffmpeg
import argparse
from pathlib import Path
import timestamps as ts

"""
USAGE : 

    --video : specify the input path
    --timestamps : specifiy the timestamps, format xx:xx seperated by spaces.

"""


def mintosec(time):
    m, s = map(int, time.split(':'))
    return m * 60 + s

def splitts(ts):
    start_str, end_str = ts.split('-')
    start = mintosec(start_str)
    end = mintosec(end_str)

    duration = end - start
    if duration <= 0:
        raise ValueError(f"Bad timestamp (end <= start): {ts}")

    return start, duration

def extract_clips(input_file, timestamps):
    p = Path(input_file)

    for i, ts in enumerate(timestamps):
        try:
            output = str(p.with_name(f"{p.stem}_results_{i}.mp4"))
            start, duration = splitts(ts)

            (
                ffmpeg
                .input(input_file, ss=start, t=duration)
                .output(
                    output,
                    vcodec="libx264",
                    acodec="aac",
                    crf=18
                )
                .run(overwrite_output=True)
            )

            print(f"Created: {output}")

        except Exception as e:
            print(f"Skipping timestamp '{ts}' (index {i}) : {e}")

def validate_file(path_str, label):
    path = Path(path_str)

    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")

    return path

def main():
    ap = argparse.ArgumentParser(description="Cut video into clips using timestamps")

    ap.add_argument("-v", "--video", required=True, help="Source video path")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-m", "--manual", action="store_true", help="Select timestamps manually")
    group.add_argument("-ts", "--timestamps", help="Timestamps file")

    args = ap.parse_args()

    input_file = args.video or input("Input file (path/to/vid.mp4): ")

    try:
        input_file = validate_file(input_file, "Video file")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    if args.manual:
        print("Manual select")
        timestamps = ts.get_timestamps(input_file).split() # type: ignore
    else : 
        print("Using the timestamps file")
        ts_file = args.timestamps or input("Timestamps file (timestamps.txt): ")
        try : 
            ts_file = validate_file(ts_file, "Timestamps file")
        except Exception as e:
            print(f"Error: {e}")
            return
        try:
            with open(ts_file, "r") as f:
                timestamps = [t for t in f.read().split() if "-" in t]
        except Exception as e:
            print(f"Failed to read timestamps: {e}")
            return
        if not timestamps:
            print("No valid timestamps found in file")
            return

    print(f"Processing {len(timestamps)} clips...")

    extract_clips(input_file, timestamps)

if __name__ == "__main__":
    main()