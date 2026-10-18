import cv2
import numpy as np


PRESETS = {
    (1280, 720): {
        "src": [(0.453, 0.635), (0.547, 0.635), (0.828, 0.944), (0.172, 0.944)],
        "dst_x": (0.25, 0.75),
    },
    (960, 540): {
        "src": [(0.450, 0.630), (0.552, 0.630), (0.880, 0.960), (0.140, 0.960)],
        "dst_x": (0.25, 0.75),
    },
}


def preset_for(width, height):
    return PRESETS.get((width, height), PRESETS[(1280, 720)])


def src_from_vanishing_point(width, height, vp):
    (tlx, ty), (trx, _), (brx, by), (blx, _) = preset_for(width, height)["src"]
    preset_vy = ty + (by - ty) * (trx - tlx) / ((trx - tlx) - (brx - blx))
    k = (ty - preset_vy) / (by - preset_vy)
    vx, vy = vp
    half_bottom = (brx - blx) / 2
    top_y = vy + k * (by - vy)
    half_top = half_bottom * k
    return [(vx - half_top, top_y), (vx + half_top, top_y),
            (vx + half_bottom, by), (vx - half_bottom, by)]


class PerspectiveTransform:
    def __init__(self, width, height, src=None, dst_x=None):
        preset = preset_for(width, height)
        src_rel = src if src is not None else preset["src"]
        dst_rel = dst_x if dst_x is not None else preset["dst_x"]

        self.size = (width, height)
        self.src = np.float32([(x * width, y * height) for x, y in src_rel])
        left = dst_rel[0] * width
        right = dst_rel[1] * width
        self.dst = np.float32([(left, 0), (right, 0), (right, height), (left, height)])

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.Minv = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp(self, img, flags=cv2.INTER_LINEAR):
        return cv2.warpPerspective(img, self.M, self.size, flags=flags)

    def unwarp(self, img, flags=cv2.INTER_LINEAR):
        return cv2.warpPerspective(img, self.Minv, self.size, flags=flags)

    def lane_width_px(self):
        return self.dst[1][0] - self.dst[0][0]

    def draw_src(self, img, color=(0, 0, 255)):
        out = img.copy()
        cv2.polylines(out, [self.src.astype(np.int32)], True, color, 3)
        return out

    def draw_dst(self, img, color=(0, 0, 255)):
        out = img.copy()
        cv2.polylines(out, [self.dst.astype(np.int32)], True, color, 3)
        return out


def estimate_vanishing_point(binary):
    h, w = binary.shape
    roi = np.zeros_like(binary)
    roi[int(h * 0.6):int(h * 0.92), :] = binary[int(h * 0.6):int(h * 0.92), :]
    lines = cv2.HoughLinesP(roi * 255, 1, np.pi / 180, threshold=40,
                            minLineLength=int(h * 0.06), maxLineGap=int(h * 0.03))
    if lines is None:
        return None

    left, right = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if y1 == y2:
            continue
        slope = (x2 - x1) / (y2 - y1)
        length = np.hypot(x2 - x1, y2 - y1)
        cx = (x1 + x2) / 2
        if -2.5 < slope < -0.4 and cx < w * 0.55:
            left.append((x1, y1, x2, y2, length))
        elif 0.4 < slope < 2.5 and cx > w * 0.45:
            right.append((x1, y1, x2, y2, length))
    if not left or not right:
        return None

    def fit_line(segments):
        xs, ys, ws = [], [], []
        for x1, y1, x2, y2, length in segments:
            xs += [x1, x2]
            ys += [y1, y2]
            ws += [length, length]
        return np.polyfit(ys, xs, 1, w=ws)

    a1, b1 = fit_line(left)
    a2, b2 = fit_line(right)
    if abs(a1 - a2) < 1e-6:
        return None
    vy = (b2 - b1) / (a1 - a2)
    vx = a1 * vy + b1
    if not (w * 0.3 < vx < w * 0.7 and h * 0.45 < vy < h * 0.7):
        return None
    return vx / w, vy / h
