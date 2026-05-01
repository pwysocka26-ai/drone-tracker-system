"""Quick audit: dla wybranych klatek z tmp_v6_smoke wygeneruj
pelnoklatkowy JPG (1920x1080) z naniesionymi wszystkimi bboxami v6
(`dron_maly` niebieski / `dron_duzy` czerwony / `pilka` zielona).

Wczytuje JPG z tmp_v6_smoke/images/ i .txt z tmp_v6_smoke/labels/,
zapisuje audyt do tmp_v6_smoke/audit_full/.
"""
import sys
from pathlib import Path

import cv2


CLASS_NAMES = {0: "dron_maly", 1: "dron_duzy", 2: "pilka"}
CLASS_COLORS = {0: (255, 128, 0), 1: (0, 0, 255), 2: (0, 255, 0)}


def audit_frame(jpg_path: Path, txt_path: Path, out_path: Path):
    img = cv2.imread(str(jpg_path))
    if img is None:
        return
    fh, fw = img.shape[:2]
    if not txt_path.exists():
        cv2.putText(img, "NO LABEL", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        for line in txt_path.read_text().strip().split("\n"):
            parts = line.split()
            if len(parts) != 5:
                continue
            cid = int(parts[0])
            cx_n, cy_n, w_n, h_n = map(float, parts[1:])
            x = int((cx_n - w_n / 2) * fw)
            y = int((cy_n - h_n / 2) * fh)
            w = int(w_n * fw)
            h = int(h_n * fh)
            color = CLASS_COLORS.get(cid, (255, 255, 255))
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            label = f"{CLASS_NAMES.get(cid, cid)} {w}x{h}px"
            cv2.putText(img, label, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])


if __name__ == "__main__":
    img_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "tmp_v6_smoke/images")
    lbl_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "tmp_v6_smoke/labels")
    out_dir = Path(sys.argv[3] if len(sys.argv) > 3 else "tmp_v6_smoke/audit_full")
    # Audyt WSZYSTKICH klatek z labelami (czyli z detekcjami v5).
    n_done = 0
    for jpg in sorted(img_dir.glob("*.jpg")):
        txt = lbl_dir / f"{jpg.stem}.txt"
        out = out_dir / f"audit_{jpg.stem}.jpg"
        audit_frame(jpg, txt, out)
        n_done += 1
    print(f"audyt zakonczony: {n_done} klatek -> {out_dir}")
