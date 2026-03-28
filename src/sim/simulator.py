import math
import random
import time
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from core.utils import DetectionTrack


@dataclass
class Drone:
    drone_id: int
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    radius: int = 9


class FormationSimulator:
    def __init__(self, width: int, height: int, drones: int, fps: int):
        self.width = width
        self.height = height
        self.num_drones = min(3, drones)
        self.fps = fps
        self.dt = 1.0 / max(1, fps)
        self.t = 0.0
        self.drones = self._init_drones()

    def _init_drones(self) -> List[Drone]:
        colors = [(80, 220, 255), (180, 255, 120), (255, 160, 120)]
        starts = [
            (self.width * 0.18, self.height * 0.32, 170.0, 70.0),
            (self.width * 0.78, self.height * 0.24, -140.0, 110.0),
            (self.width * 0.42, self.height * 0.78, 90.0, -150.0),
        ]
        out: List[Drone] = []
        for i in range(self.num_drones):
            x, y, vx, vy = starts[i]
            out.append(Drone(i + 1, x, y, vx, vy, colors[i]))
        return out

    def _bounce(self, d: Drone):
        m = 35
        if d.x < m:
            d.x = m
            d.vx = abs(d.vx)
        if d.x > self.width - m:
            d.x = self.width - m
            d.vx = -abs(d.vx)
        if d.y < m:
            d.y = m
            d.vy = abs(d.vy)
        if d.y > self.height - m:
            d.y = self.height - m
            d.vy = -abs(d.vy)

    def step(self):
        self.t += self.dt
        for i, d in enumerate(self.drones):
            if i == 0:
                ax = math.sin(self.t * 1.7) * 18.0
                ay = math.cos(self.t * 1.1) * 22.0
            elif i == 1:
                ax = math.cos(self.t * 1.3 + 1.1) * 22.0
                ay = math.sin(self.t * 1.8 + 0.7) * 16.0
            else:
                ax = math.sin(self.t * 2.0 + 2.0) * 20.0
                ay = math.cos(self.t * 1.5 + 0.3) * 20.0

            d.vx += ax * self.dt
            d.vy += ay * self.dt

            speed = math.sqrt(d.vx * d.vx + d.vy * d.vy)
            max_speed = 240.0
            if speed > max_speed:
                scale = max_speed / speed
                d.vx *= scale
                d.vy *= scale

            d.x += d.vx * self.dt
            d.y += d.vy * self.dt

            wobble_x = math.sin(self.t * (1.4 + i * 0.3) + i) * (5 + i * 2)
            wobble_y = math.cos(self.t * (1.7 + i * 0.2) + i * 0.5) * (5 + i * 2)
            d.x += wobble_x
            d.y += wobble_y

            self._bounce(d)

        return self.drones, 0, self.t


def synthesize_tracks(drones: Sequence[Drone], miss_rate: float, noise_px: float, box_size: Tuple[int, int]):
    now = time.time()
    bw, bh = box_size
    tracks = []
    for d in drones:
        if random.random() < miss_rate:
            continue
        cx = d.x + random.uniform(-noise_px, noise_px)
        cy = d.y + random.uniform(-noise_px, noise_px)
        conf = max(0.35, min(0.99, 0.90 - abs(random.gauss(0, 0.08))))
        tracks.append(
            DetectionTrack(
                track_id=d.drone_id,
                cls_id=0,
                conf=float(conf),
                bbox_xyxy=(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
                center_xy=(cx, cy),
                timestamp=now,
            )
        )
    return tracks


def draw_sim_frame(width: int, height: int, drones: Sequence[Drone], phase: int, sim_t: float):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = (10, 10, 18)

    for y in range(0, height, 80):
        cv2.line(frame, (0, y), (width, y), (25, 25, 35), 1)
    for x in range(0, width, 80):
        cv2.line(frame, (x, 0), (x, height), (25, 25, 35), 1)

    for d in drones:
        cv2.circle(frame, (int(d.x), int(d.y)), d.radius * 3, d.color, 1)
        cv2.circle(frame, (int(d.x), int(d.y)), d.radius, d.color, -1)
        cv2.putText(frame, f"DRON {d.drone_id}", (int(d.x) + 14, int(d.y) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, d.color, 2)

    cv2.putText(frame, f"t={sim_t:05.1f}", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
    return frame
