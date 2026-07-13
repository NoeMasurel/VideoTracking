import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from collections import defaultdict
import json


class ObjectTracking:
    def __init__(self, source, words, model="yolo26n.pt", start=None, end=None,
                 duration=None, confidence=None, tracker = None):
        """
        Parameters
        source      : path to the source video
        words       : list of classes to detect
        model       : YOLO .pt file
        start       : in seconds (optional)
        end         : in seconds (optional, exclusive with duration)
        duration    : in seconds (optional, exclusive with end)
        confidence  : confidence threshold (optional) (inital confidence level to pass to the tracking model)
        tracker     : tracking parameters (.yaml) (optional) 
        """

        self.model = YOLO(model)
        self.names = self.model.names
        self.indices = [{v: k for k, v in self.names.items()}[w] for w in words]
        self.confidence = confidence
        self.tracker = tracker
        self.source = source
        self.all_detections = {}

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError("Error: Cannot open video file (check path/codec)")

        self.w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames   = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / self.fps

        print(f"Video loaded: {self.w}x{self.h} @ {self.fps} FPS | "
              f"Total duration: {total_duration:.2f}s")

        if end is not None and duration is not None:
            raise ValueError("Provide either 'end' or 'duration', not both.")

        self.start_sec = start if start is not None else 0.0

        if end is not None:
            self.end_sec = end
        elif duration is not None:
            self.end_sec = self.start_sec + duration
        else:
            self.end_sec = total_duration

        self.start_sec = max(0.0, self.start_sec)
        self.end_sec   = min(total_duration, self.end_sec)

        if self.start_sec >= self.end_sec:
            raise ValueError(f"Invalid time window: [{self.start_sec}, {self.end_sec}]")

        self.start_frame = int(self.start_sec * self.fps)
        self.end_frame   = int(self.end_sec   * self.fps)

        print(f"Analyzing: {self.start_sec:.2f}s → {self.end_sec:.2f}s "
              f"(frames {self.start_frame} - {self.end_frame})")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        self.track_history = defaultdict(list)
        self.class_counts  = defaultdict(int)
        self.seen_ids      = set()

        self.rect_width         = 2
        self.font               = 1.0
        self.text_width         = 2
        self.padding            = 12
        self.margin             = 10
        self.polyline_thickness = 2

    def draw_bbox(self, im0, box, track_id, cls, conf):
        x1, y1, x2, y2 = map(int, box)
        color = colors(int(cls), True)

        cv2.rectangle(im0, (x1, y1), (x2, y2), color, self.rect_width)

        label = f"{self.names[int(cls)]}:{int(track_id)} at {conf:.2f}%"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.font, self.text_width
        )

        bg_x1, bg_y2 = x1, y1
        bg_x2 = bg_x1 + tw + 2 * self.padding
        bg_y1 = bg_y2 - (th + 2 * self.margin)

        cv2.rectangle(im0, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

        text_x = bg_x1 + ((bg_x2 - bg_x1) - tw) // 2
        text_y = bg_y1 + ((bg_y2 - bg_y1) + th) // 2 - 2

        cv2.putText(
            im0, label, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, self.font,
            (255, 255, 255), self.text_width, cv2.LINE_AA,
        )

    def update_counts(self, track_id, cls):
        key = (int(track_id), int(cls))
        if key not in self.seen_ids:
            self.seen_ids.add(key)
            self.class_counts[self.names[int(cls)]] += 1

    def draw_class_counts(self, frame):
        if not self.class_counts:
            return

        line_h  = 36
        padding = 10
        width   = 220
        height  = padding + len(self.class_counts) * line_h + padding

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + width, 10 + height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, (class_name, count) in enumerate(sorted(self.class_counts.items())):
            y = 10 + padding + i * line_h + line_h // 2 + 8
            cv2.putText(
                frame, f"{class_name}: {count}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (0, 255, 200), 2, cv2.LINE_AA
            )

    def run(self, output_json="data/detections.json"):
        current_frame = self.start_frame

        while self.cap.isOpened() and current_frame < self.end_frame:
            success, frame = self.cap.read()
            if not success:
                print("End of video or read error.")
                break

            current_frame += 1

            kwargs = {
                "persist": True,
                "verbose": False,
                "classes": self.indices,
            }
            if self.confidence is not None:
                kwargs["conf"] = self.confidence
            if self.tracker is not None:
                kwargs["tracker"] = self.tracker

            results = self.model.track(frame, **kwargs)

            if results and len(results) > 0:
                result = results[0]
                annotated_frame = results[0].plot()
                cv2.imshow("YOLO Tracking", annotated_frame)

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().tolist()  # type: ignore
                    ids   = result.boxes.id.cpu().tolist() # type: ignore
                    clss  = result.boxes.cls.tolist()
                    confs = result.boxes.conf.cpu().tolist()# type: ignore

                    frame_data = []
                    for box, track_id, cls, conf in zip(boxes, ids, clss, confs):
                        self.update_counts(track_id, cls)
                        frame_data.append({
                            "id":   int(track_id) if track_id is not None else None,
                            "cls":  int(cls),
                            "conf": conf,
                            "bbox": [float(x) for x in box]
                        })

                    self.all_detections[current_frame] = frame_data

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        # Final count 
        print("\nFinal class counts ")
        for class_name, count in sorted(self.class_counts.items()):
            print(f"  {class_name}: {count}")

       # Save JSON with metadatas 
        output = {
            "meta": {
                "source":      self.source,
                "fps":         self.fps,
                "width":       self.w,
                "height":      self.h,
                "start_frame": self.start_frame,
                "end_frame":   self.end_frame,
                "class_names": self.names,   # {int_str: class_name}
            },
            "detections": self.all_detections,
        }

        with open(output_json, "w") as f:
            json.dump(output, f)

        print(f"\nDetections saved to {output_json}")


if __name__ == "__main__":
    tracker = ObjectTracking(
        model="yolo26n.pt",
        source="videos/2025_08_11/20250811_0.mp4",
        words=["person", "bicycle"],
        tracker='deepocsort.yaml'
    )
    tracker.run()
