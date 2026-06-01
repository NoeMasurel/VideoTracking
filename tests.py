import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors
from collections import defaultdict

class ObjectTracking:
    def __init__(self, source, words, model="yolo26n.pt", start=None, end=None, duration=None):

        self.model = YOLO(model)
        self.names = self.model.names
        self.indices = [{v: k for k, v in self.names.items()}[w] for w in words]

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError("Error: Cannot open video file (check path/codec)")

        self.w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / self.fps

        print(f"Video loaded: {self.w}x{self.h} @ {self.fps} FPS | Total duration: {total_duration:.2f}s")

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
              f"(frames {self.start_frame} – {self.end_frame})")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

        output_path = source.replace(".mp4", "_tracked.mp4")
        self.writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"), # type: ignore
            self.fps,
            (self.w, self.h)
        )
        if not self.writer.isOpened():
            raise ValueError("Error: VideoWriter failed to open")

        self.track_history = defaultdict(lambda: [])

        # --- Comptage par classe ---
        # Clé : nom de classe → nombre d'IDs uniques vus
        self.class_counts = defaultdict(int)
        # Évite de compter le même (track_id, class) deux fois
        self.seen_ids = set()

        self.rect_width        = 2
        self.font              = 1.0
        self.text_width        = 2
        self.padding           = 12
        self.margin            = 10
        self.polyline_thickness = 2

        self.window_name = "YOLO Tracking"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def draw_bbox(self, im0, box, track_id, cls):
        x1, y1, x2, y2 = map(int, box)
        color = colors(int(cls), True)

        cv2.rectangle(im0, (x1, y1), (x2, y2), color, self.rect_width)

        label = f"{self.names[int(cls)]}:{int(track_id)}"
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
        """Enregistre un objet s'il n'a jamais été vu (par track_id + classe)."""
        key = (int(track_id), int(cls))
        if key not in self.seen_ids:
            self.seen_ids.add(key)
            class_name = self.names[int(cls)]
            self.class_counts[class_name] += 1

    def draw_class_counts(self, frame):
        """Affiche le tableau des comptages par classe en haut à gauche."""
        if not self.class_counts:
            return

        line_h   = 36
        padding  = 10
        width    = 220
        height   = padding + len(self.class_counts) * line_h + padding

        # Fond semi-transparent
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + width, 10 + height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        for i, (class_name, count) in enumerate(sorted(self.class_counts.items())):
            y = 10 + padding + i * line_h + line_h // 2 + 8
            label = f"{class_name}: {count}"
            cv2.putText(
                frame, label, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (0, 255, 200), 2, cv2.LINE_AA
            )

    def run(self):
        current_frame = self.start_frame

        while self.cap.isOpened() and current_frame < self.end_frame:
            success, frame = self.cap.read()
            if not success:
                print("End of video or read error.")
                break

            current_frame += 1

            results = self.model.track(frame,persist=True, verbose=False, classes=self.indices, tracker='botsort.yaml')

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu() # type: ignore
                    ids   = result.boxes.id.cpu().tolist() # type: ignore
                    clss  = result.boxes.cls.tolist()

                    for box, track_id, cls in zip(boxes, ids, clss):
                        self.draw_bbox(frame, box, track_id, cls)
                        self.update_counts(track_id, cls)

                        x1, y1, x2, y2 = box
                        cx = float((x1 + x2) / 2)
                        cy = float((y1 + y2) / 2)
                        cv2.circle(frame, (int(cx), int(cy)), 5, colors(cls, True), -1)

            self.draw_class_counts(frame)

            self.writer.write(frame)
            cv2.imshow(self.window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Affichage final dans le terminal
        print("\n=== Comptage final par classe ===")
        for class_name, count in sorted(self.class_counts.items()):
            print(f"  {class_name}: {count}")

        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tracker = ObjectTracking(
        model="yolo26n.pt",
        source="videos/2025_08_11/20250811_0.mp4",
        words=['car', 'person', 'bicycle', 'motorcycle', 'bus'],
    )
    tracker.run()   