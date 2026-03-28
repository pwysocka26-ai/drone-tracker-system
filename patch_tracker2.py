from pathlib import Path

file = Path("src/core/app.py")
code = file.read_text(encoding="utf-8")

new_tracker = '''

class SimpleTracker:
    def __init__(self):
        self.tracks = {}   # id -> (x,y,age)
        self.next_id = 1

    def update(self, detections):
        new_tracks = {}
        used_ids = set()

        for det in detections:
            cx, cy = det.center_xy

            best_id = None
            best_dist = 9999

            for tid, (px, py, age) in self.tracks.items():
                dist = ((cx - px)**2 + (cy - py)**2)**0.5

                # 🔥 ważne: ograniczamy zmianę ID
                if dist < best_dist and dist < 120 and tid not in used_ids:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                new_tracks[best_id] = (cx, cy, 0)
                det.selected_track_id = best_id
                used_ids.add(best_id)
            else:
                new_tracks[self.next_id] = (cx, cy, 0)
                det.selected_track_id = self.next_id
                used_ids.add(self.next_id)
                self.next_id += 1

        # 🔥 utrzymanie starych tracków (bezwładność)
        for tid, (px, py, age) in self.tracks.items():
            if tid not in new_tracks and age < 5:
                new_tracks[tid] = (px, py, age + 1)

        self.tracks = new_tracks
        return detections

'''

# podmień klasę
start = code.find("class SimpleTracker")
end = code.find("def run_app")

code = code[:start] + new_tracker + code[end:]

file.write_text(code, encoding="utf-8")
print("OK: tracker upgraded (stable IDs)")
