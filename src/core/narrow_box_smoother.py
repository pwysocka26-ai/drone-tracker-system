class NarrowBoxSmoother:
    def __init__(self, alpha=0.22):
        self.alpha = float(alpha)
        self.track_id = None
        self.rect = None

    def reset(self):
        self.track_id = None
        self.rect = None

    def update(self, track_id, rect):
        if rect is None:
            self.reset()
            return None

        x1, y1, x2, y2 = rect

        if self.track_id != track_id or self.rect is None:
            self.track_id = track_id
            self.rect = (int(x1), int(y1), int(x2), int(y2))
            return self.rect

        px1, py1, px2, py2 = self.rect
        a = self.alpha

        sx1 = int(px1 * (1.0 - a) + x1 * a)
        sy1 = int(py1 * (1.0 - a) + y1 * a)
        sx2 = int(px2 * (1.0 - a) + x2 * a)
        sy2 = int(py2 * (1.0 - a) + y2 * a)

        self.rect = (sx1, sy1, sx2, sy2)
        return self.rect
