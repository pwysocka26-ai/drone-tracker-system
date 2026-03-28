from pathlib import Path

p = Path("src/core/app.py")
s = p.read_text(encoding="utf-8")

s = s.replace('zoom_w = int(w * 0.12)', 'zoom_w = int(w * 0.08)')
s = s.replace('zoom_h = int(h * 0.12)', 'zoom_h = int(h * 0.08)')
s = s.replace('a = 0.90', 'a = 0.55')
s = s.replace(
    'cv2.putText(zoom, "TRACKING DRON 2", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)',
    '''cv2.putText(zoom, "TRACKING DRON 2", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.line(zoom, (640, 0), (640, 720), (0, 255, 255), 1)
        cv2.line(zoom, (0, 360), (1280, 360), (0, 255, 255), 1)'''
)

p.write_text(s, encoding="utf-8")
print("OK: updated narrow camera movement")
