from pathlib import Path

file = Path("src/core/app.py")
code = file.read_text(encoding="utf-8")

# --- DODAJ TRACKER ---
tracker_code = '''

class SimpleTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        new_tracks = {}

        for det in detections:
            cx, cy = det.center_xy

            best_id = None
            best_dist = 9999

            for tid, (px, py) in self.tracks.items():
                dist = ((cx - px)**2 + (cy - py)**2)**0.5
                if dist < best_dist and dist < 80:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                new_tracks[best_id] = (cx, cy)
                det.selected_track_id = best_id
            else:
                new_tracks[self.next_id] = (cx, cy)
                det.selected_track_id = self.next_id
                self.next_id += 1

        self.tracks = new_tracks
        return detections

'''

# wstaw tracker jeśli go nie ma
if "class SimpleTracker" not in code:
    insert_point = code.find("def run_app")
    code = code[:insert_point] + tracker_code + code[insert_point:]


# --- PODMIEŃ LOGIKĘ W RUN_APP ---

code = code.replace(
    "tracks = detect_bright_objects(frame, max_targets=3)",
    "detections = detect_bright_objects(frame, max_targets=3)\n        tracks = tracker.update(detections)"
)

# --- DODAJ tracker = SimpleTracker() ---
code = code.replace(
    "cap = cv2.VideoCapture(args.video_path)",
    "cap = cv2.VideoCapture(args.video_path)\n    tracker = SimpleTracker()"
)

file.write_text(code, encoding="utf-8")
print("OK: tracker dodany")
