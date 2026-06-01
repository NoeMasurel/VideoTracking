import ffmpeg
INPUT = 20250811

input_file = f"videos/{INPUT}.mp4"



def mintosec(time):
    m, s = map(int, time.split(':'))
    return m * 60 + s

def splitts(ts):
    start_str, end_str = ts.split('-')
    start = mintosec(start_str)
    end = mintosec(end_str)

    duration = end - start
    if duration <= 0:
        raise ValueError(f"Bad timestamp: {ts}")

    return start, duration

timestamps = ["1:13-2:35","2:35-3:52","4:17-5:45","6:50-9:05",
              "9:05-10:08","10:08-11:12","11:12-11:50",
              "11:50-13:04","13:04-14:53","15:22-18:00"]


def extract_clips(input_file, timestamps):
    for i in range(len(timestamps)):
        output = f"videos/{INPUT}_result_{i}.mp4"
        start, duration = splitts(timestamps[i])
        (
            ffmpeg
            .input(input_file, ss=start, t=duration)
            .output(output, vcodec="libx264", acodec="aac", crf=18)
            .run(overwrite_output=True)
        )

extract_clips(input_file, timestamps)