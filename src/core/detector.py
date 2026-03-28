import cv2
import numpy as np
from typing import List

from core.models import Detection


def detect_dark_objects(frame, max_targets: int = 16) -> List[Detection]:
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, th1 = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY_INV)
    _, th2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(th1, th2)

    sky_limit = int(h * 0.78)
    sky_mask = np.zeros_like(mask)
    sky_mask[:sky_limit, :] = 255
    mask = cv2.bitwise_and(mask, sky_mask)

    small = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small)
    mask = cv2.dilate(mask, small, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4 or area > 900:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 2 or bh < 2:
            continue

        ratio = bw / float(max(1, bh))
        if ratio < 0.18 or ratio > 9.0:
            continue

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        conf = min(0.99, max(0.35, area / 80.0))
        candidates.append((conf, x, y, bw, bh, cx, cy))

    candidates.sort(reverse=True, key=lambda x: x[0])

    out: List[Detection] = []
    kept = []
    min_dist = 22

    for conf, x, y, bw, bh, cx, cy in candidates:
        if any(np.hypot(cx - px, cy - py) < min_dist for px, py in kept):
            continue

        kept.append((cx, cy))
        out.append(
            Detection(
                bbox_xyxy=(x, y, x + bw, y + bh),
                center_xy=(cx, cy),
                confidence=float(conf),
            )
        )
        if len(out) >= max_targets:
            break

    return out
