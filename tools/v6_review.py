"""Interactive single-window review tool dla v6 multi-class datasetu.

Wszystkie operacje w jednym oknie. Operator pracuje myszka:

PRZESUWANIE WIDOKU (pan):
    - drag LMB na pustym miejscu (poza bboxami) = pan
    - drag prawym (RMB) gdziekolwiek = pan (alternatywnie)

ZOOM:
    - kolko myszy = zoom in/out wokol kursora
    - klawisze + / - = zoom in/out (centrum)
    - 0 = reset zoom

EDYCJA ISTNIEJACEGO BBOXA (mysz):
    - drag rogu (corner) = resize z tego rogu
    - drag krawedzi (edge) = przesun jedna krawedz
    - drag wewnatrz bboxa = przesun caly bbox
    Hover na bboxie pokazuje biale uchwyty (handles) - 8 punktow:
    4 rogi + 4 srodki krawedzi.

DODAWANIE NOWEGO BBOXA:
    - 'a' = wejdz w tryb DRAW
    - drag LMB na obrazie = rysuj bbox
    - po release: nacisnij 0/1/2 dla klasy
    - ESC = anuluj DRAW

PORUSZANIE PO KLATKACH I MODYFIKACJE:
    space / n   next frame
    p           previous frame
    1-9         select bbox o tym numerze (toggle)
    m           reclassify wybrany (lub wszystkie) -> dron_maly
    d           reclassify -> dron_duzy
    b           reclassify -> pilka
    x           delete wybrany (lub wszystkie) bbox
    X           delete cala klatke (.jpg + .txt)
    a           tryb add bbox
    q           quit (auto-save)
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2


CLASS_NAMES = {0: "dron_maly", 1: "dron_duzy", 2: "pilka"}
CLASS_COLORS = {0: (255, 128, 0), 1: (0, 0, 255), 2: (0, 255, 0)}
CLASS_KEYS = {ord("m"): 0, ord("d"): 1, ord("b"): 2}
CLASS_KEYS_ALL = {ord("M"): 0, ord("D"): 1, ord("B"): 2}
ADD_CLASS_KEYS = {ord("0"): 0, ord("1"): 1, ord("2"): 2}
DRAG_THRESHOLD_DISP = 3  # px display - poniżej tej odleglosci klik traktowany jako select

MAX_DISPLAY_W = 1600
ZOOM_MIN = 1.0
ZOOM_MAX = 8.0
HANDLE_DISP = 5         # px display - rysowany handle (visible)
HIT_HANDLE_RADIUS = 6   # px display - radius do hit-test handle (clickable)
HIT_EDGE_BAND = 3       # px display - szerokosc pasma edge-line do hit-test


def read_bboxes(txt_path: Path):
    if not txt_path.exists():
        return []
    out = []
    for line in txt_path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        out.append({
            "class": int(parts[0]),
            "cx": float(parts[1]),
            "cy": float(parts[2]),
            "w": float(parts[3]),
            "h": float(parts[4]),
        })
    return out


def write_bboxes(txt_path: Path, bboxes):
    if not bboxes:
        if txt_path.exists():
            txt_path.unlink()
        return
    with open(txt_path, "w", encoding="utf-8") as f:
        for b in bboxes:
            f.write(f"{b['class']} {b['cx']:.6f} {b['cy']:.6f} "
                    f"{b['w']:.6f} {b['h']:.6f}\n")


def bbox_to_xywh(b, fw, fh):
    x = int((b["cx"] - b["w"] / 2) * fw)
    y = int((b["cy"] - b["h"] / 2) * fh)
    w = int(b["w"] * fw)
    h = int(b["h"] * fh)
    return x, y, w, h


def pix_to_norm(pix, class_id, fw, fh):
    x, y, w, h = pix
    return {
        "class": class_id,
        "cx": (x + w / 2) / fw,
        "cy": (y + h / 2) / fh,
        "w": w / fw,
        "h": h / fh,
    }


def normalize_pix(pix):
    x, y, w, h = pix
    if w < 0:
        x, w = x + w, -w
    if h < 0:
        y, h = y + h, -h
    return [int(x), int(y), int(w), int(h)]


def hit_test(bbox_pix, x, y, edge_thresh):
    """bbox_pix = [x, y, w, h] w frame coords; x,y test point in frame coords.
    Zwraca: 'outside' | 'inside' | 'top'/'bottom'/'left'/'right' | 'tl'/'tr'/'bl'/'br'
    """
    bx, by, bw, bh = bbox_pix
    if bw <= 0 or bh <= 0:
        return "outside"
    L, R = bx, bx + bw
    T, B = by, by + bh
    near_L = abs(x - L) <= edge_thresh
    near_R = abs(x - R) <= edge_thresh
    near_T = abs(y - T) <= edge_thresh
    near_B = abs(y - B) <= edge_thresh
    in_x = L - edge_thresh <= x <= R + edge_thresh
    in_y = T - edge_thresh <= y <= B + edge_thresh
    if not (in_x and in_y):
        return "outside"
    if near_T and near_L:
        return "tl"
    if near_T and near_R:
        return "tr"
    if near_B and near_L:
        return "bl"
    if near_B and near_R:
        return "br"
    if near_T:
        return "top"
    if near_B:
        return "bottom"
    if near_L:
        return "left"
    if near_R:
        return "right"
    if L < x < R and T < y < B:
        return "inside"
    return "outside"


def apply_resize(kind, ob_pix, dx, dy):
    """Aplikuje delta (dx, dy) do bbox'a wedlug kind (top/right/tl/...) ."""
    x, y, w, h = ob_pix
    nx, ny, nw, nh = x, y, w, h
    if kind in ("left", "tl", "bl"):
        nx = x + dx
        nw = w - dx
    if kind in ("right", "tr", "br"):
        nw = w + dx
    if kind in ("top", "tl", "tr"):
        ny = y + dy
        nh = h - dy
    if kind in ("bottom", "bl", "br"):
        nh = h + dy
    return [nx, ny, nw, nh]


class App:
    def __init__(self, img_dir, lbl_dir, backup_dir, log_path):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.backup_dir = backup_dir
        self.log_path = log_path
        backup_dir.mkdir(exist_ok=True)

        self.frames = sorted(p.stem for p in img_dir.glob("*.jpg"))
        if not self.frames:
            print(f"no frames in {img_dir}")
            sys.exit(1)
        print(f"loaded {len(self.frames)} frames from {img_dir}")

        self.log = []
        if log_path.exists():
            try:
                self.log = json.loads(log_path.read_text())
            except Exception:
                self.log = []

        # Auto-resume: jesli log ma akcje, znajdz ostatnia tknieta klatke
        # i ustaw idx na nia (operator widzi gdzie skonczyl).
        self.idx = 0
        if self.log:
            last_frame = self.log[-1].get("frame")
            if last_frame and last_frame in self.frames:
                resume_idx = self.frames.index(last_frame)
                self.idx = resume_idx
                print(f"[resume] last touched frame: {last_frame} (idx {resume_idx+1}/{len(self.frames)})")
        self.selected = None

        # viewport
        self.zoom = 1.0
        self.cx = 0
        self.cy = 0
        self.fit_scale = 1.0
        self.disp_w = 0
        self.disp_h = 0

        # frame state
        self.img = None
        self.bboxes = []
        self.fw = 0
        self.fh = 0
        self.dirty = False
        self.stem = ""

        # interaction state
        self.mode = "navigate"
        self.drag_start_frame = None
        self.origin_norm = None       # bbox dict przed dragiem (move/resize)
        self.edit_idx = None
        self.resize_kind = None
        self.pending_pix = None       # float [x, y, w, h] dla draw mode
        self.pan_start_disp = None
        self.pan_origin_cxcy = None
        self.hover = ""

        self.win = "v6 review"
        cv2.namedWindow(self.win)
        cv2.setMouseCallback(self.win, self._on_mouse)

    # ---------- viewport ----------

    def total_scale(self):
        return self.fit_scale * self.zoom

    def disp_to_frame(self, dx, dy):
        s = self.total_scale()
        fx = (dx - self.disp_w / 2) / s + self.cx
        fy = (dy - self.disp_h / 2) / s + self.cy
        return fx, fy

    def frame_to_disp(self, fx, fy):
        s = self.total_scale()
        dx = (fx - self.cx) * s + self.disp_w / 2
        dy = (fy - self.cy) * s + self.disp_h / 2
        return dx, dy

    def clamp_center(self):
        if self.zoom <= 1.0:
            self.cx = self.fw / 2
            self.cy = self.fh / 2
            return
        s = self.total_scale()
        vw = self.disp_w / s
        vh = self.disp_h / s
        self.cx = max(vw / 2, min(self.fw - vw / 2, self.cx))
        self.cy = max(vh / 2, min(self.fh - vh / 2, self.cy))

    def zoom_at(self, dx, dy, factor):
        old_fx, old_fy = self.disp_to_frame(dx, dy)
        new_zoom = max(ZOOM_MIN, min(ZOOM_MAX, self.zoom * factor))
        if new_zoom == self.zoom:
            return
        self.zoom = new_zoom
        s = self.total_scale()
        self.cx = old_fx - (dx - self.disp_w / 2) / s
        self.cy = old_fy - (dy - self.disp_h / 2) / s
        self.clamp_center()

    def reset_view(self):
        self.zoom = 1.0
        self.cx = self.fw / 2
        self.cy = self.fh / 2

    # ---------- bbox hit test ----------

    def hit_test_bboxes_disp(self, dx, dy):
        """Test w DISPLAY coords (paint-like).

        Priority: corner handle > edge midpoint handle > edge line > inside.
        Iteruje od konca listy (ostatnio rysowany ma priorytet w nakladaniu).
        """
        for i in reversed(range(len(self.bboxes))):
            b = self.bboxes[i]
            x_f = (b["cx"] - b["w"] / 2) * self.fw
            y_f = (b["cy"] - b["h"] / 2) * self.fh
            w_f = b["w"] * self.fw
            h_f = b["h"] * self.fh
            dL, dT = self.frame_to_disp(x_f, y_f)
            dR, dB = self.frame_to_disp(x_f + w_f, y_f + h_f)
            MX = (dL + dR) / 2
            MY = (dT + dB) / 2

            if dR < -HIT_HANDLE_RADIUS or dB < -HIT_HANDLE_RADIUS:
                continue
            if dL > self.disp_w + HIT_HANDLE_RADIUS or dT > self.disp_h + HIT_HANDLE_RADIUS:
                continue

            # 1. Corner handles (priorytet)
            for kind, hx, hy in (("tl", dL, dT), ("tr", dR, dT),
                                 ("bl", dL, dB), ("br", dR, dB)):
                if (abs(dx - hx) <= HIT_HANDLE_RADIUS
                        and abs(dy - hy) <= HIT_HANDLE_RADIUS):
                    return i, kind

            # 2. Edge midpoint handles
            for kind, hx, hy in (("top", MX, dT), ("bottom", MX, dB),
                                 ("left", dL, MY), ("right", dR, MY)):
                if (abs(dx - hx) <= HIT_HANDLE_RADIUS
                        and abs(dy - hy) <= HIT_HANDLE_RADIUS):
                    return i, kind

            # 3. Edge lines (thin band wzdluz krawedzi)
            on_top = (abs(dy - dT) <= HIT_EDGE_BAND
                      and dL <= dx <= dR)
            on_bottom = (abs(dy - dB) <= HIT_EDGE_BAND
                         and dL <= dx <= dR)
            on_left = (abs(dx - dL) <= HIT_EDGE_BAND
                       and dT <= dy <= dB)
            on_right = (abs(dx - dR) <= HIT_EDGE_BAND
                        and dT <= dy <= dB)
            if on_top:
                return i, "top"
            if on_bottom:
                return i, "bottom"
            if on_left:
                return i, "left"
            if on_right:
                return i, "right"

            # 4. Inside
            if dL <= dx <= dR and dT <= dy <= dB:
                return i, "inside"

        return None, "outside"

    # ---------- frame I/O ----------

    def load_frame(self):
        # Defensywny save: jesli bylismy w pol drag-u albo cos sie nie zapisalo
        self.save_if_dirty()
        self.stem = self.frames[self.idx]
        jpg = self.img_dir / f"{self.stem}.jpg"
        txt = self.lbl_dir / f"{self.stem}.txt"
        self.img = cv2.imread(str(jpg))
        self.fh, self.fw = self.img.shape[:2]
        self.bboxes = read_bboxes(txt)
        if txt.exists():
            backup = self.backup_dir / f"{self.stem}.txt"
            if not backup.exists():
                backup.write_bytes(txt.read_bytes())
        self.fit_scale = min(MAX_DISPLAY_W / self.fw, 1.0)
        self.disp_w = int(self.fw * self.fit_scale)
        self.disp_h = int(self.fh * self.fit_scale)
        self.clamp_center()
        self.dirty = False
        self.mode = "navigate"
        self.pending_pix = None
        self.edit_idx = None
        self.hover = ""

    def save_if_dirty(self):
        if self.dirty:
            txt = self.lbl_dir / f"{self.stem}.txt"
            write_bboxes(txt, self.bboxes)
            self.dirty = False

    # ---------- render ----------

    def render(self):
        s = self.total_scale()
        vw = self.disp_w / s
        vh = self.disp_h / s
        src_x = max(0, int(self.cx - vw / 2))
        src_y = max(0, int(self.cy - vh / 2))
        src_w = min(self.fw - src_x, int(vw) + 1)
        src_h = min(self.fh - src_y, int(vh) + 1)
        crop = self.img[src_y:src_y + src_h, src_x:src_x + src_w].copy()

        # Resize crop -> display BEZ bboxow.
        # Bboxy rysujemy bezposrednio na disp w sub-pixel precision (shift=4).
        interp = cv2.INTER_NEAREST if s > 1.5 else cv2.INTER_LINEAR
        disp = cv2.resize(crop, (self.disp_w, self.disp_h), interpolation=interp)

        SHIFT = 4
        SCALE = 1 << SHIFT  # 16

        # rysuj bboxy w sub-pixel float precision na disp
        for i, b in enumerate(self.bboxes):
            x_f = (b["cx"] - b["w"] / 2) * self.fw
            y_f = (b["cy"] - b["h"] / 2) * self.fh
            w_f = b["w"] * self.fw
            h_f = b["h"] * self.fh
            dx1, dy1 = self.frame_to_disp(x_f, y_f)
            dx2, dy2 = self.frame_to_disp(x_f + w_f, y_f + h_f)
            if dx2 < 0 or dy2 < 0 or dx1 > self.disp_w or dy1 > self.disp_h:
                continue
            color = CLASS_COLORS.get(b["class"], (255, 255, 255))
            cv2.rectangle(disp,
                          (int(dx1 * SCALE), int(dy1 * SCALE)),
                          (int(dx2 * SCALE), int(dy2 * SCALE)),
                          color, 1, cv2.LINE_AA, SHIFT)
            line_h = 16
            label_y = int(dy1) - 4
            if label_y - 2 * line_h < 0:
                label_y = int(dy2) + line_h + 4
            cv2.putText(disp, f"#{i+1}",
                        (int(dx1), label_y - line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.putText(disp,
                        f"{CLASS_NAMES.get(b['class'], '?')} {int(w_f)}x{int(h_f)}",
                        (int(dx1), label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # handles na hover/selected/editing bbox
        for i, b in enumerate(self.bboxes):
            show_handles = False
            if i == self.edit_idx:
                show_handles = True
            elif self.hover.startswith(f"bbox_{i}_") and self.mode == "navigate":
                show_handles = True
            elif i == self.selected:
                show_handles = True
            if not show_handles:
                continue
            x_f = (b["cx"] - b["w"] / 2) * self.fw
            y_f = (b["cy"] - b["h"] / 2) * self.fh
            w_f = b["w"] * self.fw
            h_f = b["h"] * self.fh
            dL, dT = self.frame_to_disp(x_f, y_f)
            dR, dB = self.frame_to_disp(x_f + w_f, y_f + h_f)
            L, R, T, B = int(round(dL)), int(round(dR)), int(round(dT)), int(round(dB))
            MX = (L + R) // 2
            MY = (T + B) // 2
            HS = HANDLE_DISP
            for hx, hy in [(L, T), (R, T), (L, B), (R, B),
                           (MX, T), (MX, B), (L, MY), (R, MY)]:
                cv2.rectangle(disp, (hx - HS, hy - HS),
                              (hx + HS, hy + HS),
                              (255, 255, 255), 1)

        # pending bbox (drawing / class_pending) - sub-pixel
        if self.pending_pix is not None and self.mode in ("drawing", "class_pending"):
            x_f, y_f, w_f, h_f = self.pending_pix
            dx1, dy1 = self.frame_to_disp(x_f, y_f)
            dx2, dy2 = self.frame_to_disp(x_f + w_f, y_f + h_f)
            cv2.rectangle(disp,
                          (int(dx1 * SCALE), int(dy1 * SCALE)),
                          (int(dx2 * SCALE), int(dy2 * SCALE)),
                          (0, 255, 255), 1, cv2.LINE_AA, SHIFT)

        # status bar
        bar_h = 44
        cv2.rectangle(disp, (0, self.disp_h - bar_h),
                      (self.disp_w, self.disp_h),
                      (0, 0, 0), -1)

        sel_str = (f"sel:#{self.selected+1}"
                   if self.selected is not None else "sel:none")
        zoom_str = f"zoom={self.zoom:.1f}x"
        if self.mode == "draw_pending":
            mode_str = "DRAW MODE: drag LMB to draw bbox  [ESC=cancel]"
            mode_color = (0, 255, 255)
        elif self.mode == "drawing":
            mode_str = "DRAWING... release to confirm"
            mode_color = (0, 255, 255)
        elif self.mode == "class_pending":
            mode_str = "PRESS CLASS:  0=dron_maly  1=dron_duzy  2=pilka  [ESC=cancel]"
            mode_color = (0, 255, 0)
        elif self.hover.startswith("bbox_"):
            mode_str = f"hover: {self.hover}"
            mode_color = (200, 200, 200)
        else:
            mode_str = "navigate"
            mode_color = (200, 200, 200)

        line1 = f"{self.stem}  [{self.idx+1}/{len(self.frames)}]  {len(self.bboxes)}box  {sel_str}  {zoom_str}"
        cv2.putText(disp, line1, (8, self.disp_h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(disp, mode_str, (8, self.disp_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color, 1)

        return disp

    # ---------- helpers ----------

    def _apply_move_or_resize(self, fx, fy):
        """Aplikuje aktualny ruch myszy do bbox'a w trybie moving/resizing."""
        sx, sy = self.drag_start_frame
        dx_n = (fx - sx) / self.fw
        dy_n = (fy - sy) / self.fh
        on = self.origin_norm
        if self.mode == "moving":
            self.bboxes[self.edit_idx] = {
                "class": on["class"],
                "cx": on["cx"] + dx_n,
                "cy": on["cy"] + dy_n,
                "w": on["w"],
                "h": on["h"],
            }
        else:  # resizing
            kind = self.resize_kind
            new_cx = on["cx"]
            new_cy = on["cy"]
            new_w = on["w"]
            new_h = on["h"]
            if kind in ("left", "tl", "bl"):
                new_cx = on["cx"] + dx_n / 2
                new_w = on["w"] - dx_n
            if kind in ("right", "tr", "br"):
                new_cx = on["cx"] + dx_n / 2
                new_w = on["w"] + dx_n
            if kind in ("top", "tl", "tr"):
                new_cy = on["cy"] + dy_n / 2
                new_h = on["h"] - dy_n
            if kind in ("bottom", "bl", "br"):
                new_cy = on["cy"] + dy_n / 2
                new_h = on["h"] + dy_n
            min_w = 2 / self.fw
            min_h = 2 / self.fh
            if new_w < min_w:
                new_w = min_w
            if new_h < min_h:
                new_h = min_h
            self.bboxes[self.edit_idx] = {
                "class": on["class"],
                "cx": new_cx, "cy": new_cy,
                "w": new_w, "h": new_h,
            }
        self.dirty = True

    # ---------- mouse callback ----------

    def _on_mouse(self, event, x, y, flags, param):
        fx, fy = self.disp_to_frame(x, y)

        # wheel zoom dziala zawsze
        if event == cv2.EVENT_MOUSEWHEEL:
            delta_raw = (flags >> 16) & 0xFFFF
            if delta_raw > 32767:
                delta_raw -= 65536
            factor = 1.25 if delta_raw > 0 else 1 / 1.25
            self.zoom_at(x, y, factor)
            return

        if self.mode == "draw_pending":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.mode = "drawing"
                self.drag_start_frame = (fx, fy)
                self.pending_pix = [int(fx), int(fy), 0, 0]
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.mode = "navigate"
                self.pending_pix = None
            return

        if self.mode == "drawing":
            if event == cv2.EVENT_MOUSEMOVE:
                sx, sy = self.drag_start_frame
                self.pending_pix = normalize_pix(
                    [sx, sy, fx - sx, fy - sy])
            elif event == cv2.EVENT_LBUTTONUP:
                if (self.pending_pix is None
                        or self.pending_pix[2] < 2 or self.pending_pix[3] < 2):
                    self.mode = "draw_pending"
                    self.pending_pix = None
                else:
                    self.mode = "class_pending"
            return

        if self.mode == "class_pending":
            return  # tylko klawisz 0/1/2 albo ESC

        # NAVIGATE
        if self.mode == "navigate":
            if event == cv2.EVENT_MOUSEMOVE:
                idx, kind = self.hit_test_bboxes_disp(x, y)
                self.hover = f"bbox_{idx}_{kind}" if idx is not None else ""
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                idx, kind = self.hit_test_bboxes_disp(x, y)
                if idx is not None and kind != "outside":
                    # NIE od razu moving/resizing - moze to byl klik (select)
                    self.mode = "click_or_drag"
                    self.edit_idx = idx
                    self.resize_kind = kind   # "inside" / "tl" / "top" itd
                    self.drag_start_frame = (fx, fy)
                    self.origin_norm = dict(self.bboxes[idx])
                else:
                    # LMB na pustym = pan
                    self.mode = "panning"
                    self.pan_start_disp = (x, y)
                    self.pan_origin_cxcy = (self.cx, self.cy)
                return
            if event == cv2.EVENT_RBUTTONDOWN:
                self.mode = "panning"
                self.pan_start_disp = (x, y)
                self.pan_origin_cxcy = (self.cx, self.cy)
                return

        if self.mode == "click_or_drag":
            if event == cv2.EVENT_MOUSEMOVE:
                sx, sy = self.drag_start_frame
                s = self.total_scale()
                # delta w pikselach DISPLAY zeby threshold byl niezalezny od zoom
                delta_disp = max(abs(fx - sx), abs(fy - sy)) * s
                if delta_disp > DRAG_THRESHOLD_DISP:
                    # eskalacja do drag
                    if self.resize_kind == "inside":
                        self.mode = "moving"
                    else:
                        self.mode = "resizing"
                    self._apply_move_or_resize(fx, fy)
            elif event == cv2.EVENT_LBUTTONUP:
                # Brak ruchu = klik = SELECT (toggle)
                if self.selected == self.edit_idx:
                    self.selected = None
                else:
                    self.selected = self.edit_idx
                self.mode = "navigate"
                self.edit_idx = None
                self.resize_kind = None
                self.origin_norm = None
            return

        if self.mode == "panning":
            if event == cv2.EVENT_MOUSEMOVE:
                sx, sy = self.pan_start_disp
                ocx, ocy = self.pan_origin_cxcy
                s = self.total_scale()
                self.cx = ocx - (x - sx) / s
                self.cy = ocy - (y - sy) / s
                self.clamp_center()
            elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
                self.mode = "navigate"
            return

        if self.mode in ("moving", "resizing"):
            if event == cv2.EVENT_MOUSEMOVE:
                self._apply_move_or_resize(fx, fy)
            elif event == cv2.EVENT_LBUTTONUP:
                self.dirty = True
                self.log.append({
                    "ts": datetime.utcnow().isoformat(),
                    "frame": self.stem,
                    "action": f"{self.mode}_bbox",
                    "idx": self.edit_idx,
                })
                self.mode = "navigate"
                self.edit_idx = None
                self.resize_kind = None
            return

    # ---------- key handling ----------

    def handle_key(self, k):
        """Zwraca: 'next' / 'prev' / 'quit' / 'delete_frame' / False."""
        # mode-specific
        if self.mode == "draw_pending":
            if k == 27:  # ESC
                self.mode = "navigate"
                self.pending_pix = None
            return False
        if self.mode == "drawing":
            if k == 27:
                self.mode = "navigate"
                self.pending_pix = None
            return False
        if self.mode == "class_pending":
            if k == 27:
                self.mode = "navigate"
                self.pending_pix = None
            elif k in ADD_CLASS_KEYS:
                cls = ADD_CLASS_KEYS[k]
                bb = pix_to_norm(self.pending_pix, cls, self.fw, self.fh)
                self.bboxes.append(bb)
                self.dirty = True
                self.log.append({
                    "ts": datetime.utcnow().isoformat(),
                    "frame": self.stem,
                    "action": "add_bbox",
                    "class": cls,
                    "pix": self.pending_pix,
                })
                self.mode = "navigate"
                self.pending_pix = None
            return False

        # navigate
        if k == ord("q"):
            return "quit"
        if k == ord("n") or k == ord(" "):
            return "next"
        if k == ord("p"):
            return "prev"
        if k in (ord("+"), ord("=")):
            self.zoom_at(self.disp_w / 2, self.disp_h / 2, 1.25)
            return False
        if k in (ord("-"), ord("_")):
            self.zoom_at(self.disp_w / 2, self.disp_h / 2, 1 / 1.25)
            return False
        if k == ord("0"):
            self.reset_view()
            return False
        if ord("1") <= k <= ord("9"):
            n = k - ord("0")
            if n <= len(self.bboxes):
                self.selected = (n - 1) if self.selected != (n - 1) else None
            return False
        if k in CLASS_KEYS:
            new_class = CLASS_KEYS[k]
            if self.selected is None or self.selected >= len(self.bboxes):
                # brak selekcji - no-op (nie zmieniamy WSZYSTKICH bez intencji)
                return False
            old = self.bboxes[self.selected]["class"]
            self.bboxes[self.selected]["class"] = new_class
            self.log.append({
                "ts": datetime.utcnow().isoformat(),
                "frame": self.stem,
                "action": "reclassify",
                "idx": self.selected, "old": old, "new": new_class,
            })
            self.dirty = True
            return False
        if k in CLASS_KEYS_ALL:
            new_class = CLASS_KEYS_ALL[k]
            if not self.bboxes:
                return False
            old = [b["class"] for b in self.bboxes]
            for b in self.bboxes:
                b["class"] = new_class
            self.log.append({
                "ts": datetime.utcnow().isoformat(),
                "frame": self.stem,
                "action": "reclassify_all",
                "old": old, "new": new_class,
            })
            self.dirty = True
            return False
        if k == ord("x"):
            if self.selected is not None and self.selected < len(self.bboxes):
                removed = self.bboxes.pop(self.selected)
                self.log.append({
                    "ts": datetime.utcnow().isoformat(),
                    "frame": self.stem,
                    "action": "delete_bbox",
                    "class": removed["class"],
                })
                self.selected = None
            elif self.bboxes:
                self.log.append({
                    "ts": datetime.utcnow().isoformat(),
                    "frame": self.stem,
                    "action": "delete_all_bboxes",
                    "count": len(self.bboxes),
                })
                self.bboxes.clear()
            self.dirty = True
            return False
        if k == ord("X"):
            return "delete_frame"
        if k == ord("a"):
            self.mode = "draw_pending"
            self.pending_pix = None
            return False
        return False

    # ---------- main loop ----------

    def run(self):
        while True:
            if self.idx >= len(self.frames):
                print("end of frames")
                break
            if self.idx < 0:
                self.idx = 0
            self.load_frame()
            advance = self._frame_loop()
            if advance == "quit":
                break
            if advance == "next":
                self.idx += 1
                self.selected = None
            elif advance == "prev":
                self.idx -= 1
                self.selected = None
            elif advance == "delete_frame":
                self._delete_current_frame()
                if not self.frames:
                    print("all frames deleted")
                    break
                if self.idx >= len(self.frames):
                    self.idx = len(self.frames) - 1
        self.shutdown()

    def _frame_loop(self):
        while True:
            disp = self.render()
            cv2.imshow(self.win, disp)
            k = cv2.waitKey(20) & 0xFF
            self.save_if_dirty()
            if k == 255:
                continue
            ret = self.handle_key(k)
            if ret in ("quit", "next", "prev", "delete_frame"):
                self.save_if_dirty()
                return ret

    def _delete_current_frame(self):
        jpg = self.img_dir / f"{self.stem}.jpg"
        txt = self.lbl_dir / f"{self.stem}.txt"
        if jpg.exists():
            jpg.unlink()
        if txt.exists():
            txt.unlink()
        self.log.append({
            "ts": datetime.utcnow().isoformat(),
            "frame": self.stem,
            "action": "delete_frame",
        })
        print(f"deleted: {self.stem}")
        self.frames.pop(self.idx)
        self.selected = None

    def shutdown(self):
        self.log_path.write_text(json.dumps(self.log, indent=2))
        print(f"log: {self.log_path}")
        cv2.destroyAllWindows()


def main():
    img_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "tmp_v6_smoke/images")
    lbl_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "tmp_v6_smoke/labels")
    backup_dir = lbl_dir.parent / "labels_backup_review"
    log_path = lbl_dir.parent / "review_log.json"
    App(img_dir, lbl_dir, backup_dir, log_path).run()


if __name__ == "__main__":
    main()
