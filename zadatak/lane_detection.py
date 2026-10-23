from collections import deque

import cv2
import numpy as np


LANE_WIDTH_M = 3.7
LANE_LENGTH_M = 30.0


def find_lane_pixels_sliding_window(binary_warped, nwindows=9, margin=None, minpix=None):
    h, w = binary_warped.shape
    margin = margin or int(w * 0.08)
    minpix = minpix or int(w * 0.04)

    histogram = np.sum(binary_warped[h // 2:, :], axis=0)
    midpoint = w // 2
    leftx_base = np.argmax(histogram[int(w * 0.1):midpoint]) + int(w * 0.1)
    rightx_base = np.argmax(histogram[midpoint:int(w * 0.9)]) + midpoint

    window_height = h // nwindows
    nonzeroy, nonzerox = binary_warped.nonzero()

    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []
    windows = []

    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        windows.append(((win_xleft_low, win_y_low), (win_xleft_high, win_y_high)))
        windows.append(((win_xright_low, win_y_low), (win_xright_high, win_y_high)))

        in_y = (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
        good_left = (in_y & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right = (in_y & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left)
        right_lane_inds.append(good_right)

        if len(good_left) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left]))
        if len(good_right) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    return (
        nonzerox[left_lane_inds], nonzeroy[left_lane_inds],
        nonzerox[right_lane_inds], nonzeroy[right_lane_inds],
        windows,
    )


def find_lane_pixels_around_poly(binary_warped, left_fit, right_fit, margin=None):
    h, w = binary_warped.shape
    margin = margin or int(w * 0.06)
    nonzeroy, nonzerox = binary_warped.nonzero()

    left_center = np.polyval(left_fit, nonzeroy)
    right_center = np.polyval(right_fit, nonzeroy)
    left_inds = np.abs(nonzerox - left_center) < margin
    right_inds = np.abs(nonzerox - right_center) < margin

    return (
        nonzerox[left_inds], nonzeroy[left_inds],
        nonzerox[right_inds], nonzeroy[right_inds],
    )


def fit_polynomial(x, y, min_points=200):
    if len(x) < min_points:
        return None
    return np.polyfit(y, x, 2)


class LaneGeometry:
    def __init__(self, width, height, lane_width_px):
        self.width = width
        self.height = height
        self.ym_per_pix = LANE_LENGTH_M / height
        self.xm_per_pix = LANE_WIDTH_M / lane_width_px

    def curvature(self, fit):
        ploty = np.linspace(0, self.height - 1, self.height)
        plotx = np.polyval(fit, ploty)
        fit_m = np.polyfit(ploty * self.ym_per_pix, plotx * self.xm_per_pix, 2)
        y_eval = (self.height - 1) * self.ym_per_pix
        return ((1 + (2 * fit_m[0] * y_eval + fit_m[1]) ** 2) ** 1.5) / max(abs(2 * fit_m[0]), 1e-9)

    def vehicle_offset(self, left_fit, right_fit):
        y = self.height - 1
        lane_center = (np.polyval(left_fit, y) + np.polyval(right_fit, y)) / 2
        return (self.width / 2 - lane_center) * self.xm_per_pix

    def lane_width_m(self, left_fit, right_fit):
        ys = np.linspace(0, self.height - 1, 10)
        widths = (np.polyval(right_fit, ys) - np.polyval(left_fit, ys)) * self.xm_per_pix
        return widths.min(), widths.max()


class Line:
    def __init__(self, n=8):
        self.recent_fits = deque(maxlen=n)
        self.best_fit = None
        self.detected = False
        self.misses = 0

    def update(self, fit):
        if fit is None:
            self.detected = False
            self.misses += 1
            if self.misses > 10:
                self.recent_fits.clear()
                self.best_fit = None
            return
        self.detected = True
        self.misses = 0
        self.recent_fits.append(fit)
        weights = np.arange(1, len(self.recent_fits) + 1)
        self.best_fit = np.average(np.array(self.recent_fits), axis=0, weights=weights)

    def reset(self):
        self.recent_fits.clear()
        self.best_fit = None
        self.detected = False
        self.misses = 0


class LaneDetector:
    def __init__(self, geometry, smooth=True):
        self.geometry = geometry
        self.smooth = smooth
        self.left = Line()
        self.right = Line()
        self.last_debug = None

    def _sane(self, left_fit, right_fit):
        if left_fit is None or right_fit is None:
            return False
        min_w, max_w = self.geometry.lane_width_m(left_fit, right_fit)
        if min_w < 2.4 or max_w > 5.0 or (max_w - min_w) > 1.2:
            return False
        if self.left.best_fit is not None and self.right.best_fit is not None:
            y = self.geometry.height - 1
            dl = abs(np.polyval(left_fit, y) - np.polyval(self.left.best_fit, y))
            dr = abs(np.polyval(right_fit, y) - np.polyval(self.right.best_fit, y))
            limit = self.geometry.width * 0.08
            if dl > limit or dr > limit:
                return False
        return True

    def _search(self, binary_warped, use_prior):
        windows = []
        if use_prior:
            lx, ly, rx, ry = find_lane_pixels_around_poly(
                binary_warped, self.left.best_fit, self.right.best_fit)
        else:
            lx, ly, rx, ry, windows = find_lane_pixels_sliding_window(binary_warped)
        return lx, ly, rx, ry, windows

    def detect(self, binary_warped):
        use_prior = (self.smooth and self.left.best_fit is not None
                     and self.right.best_fit is not None
                     and self.left.detected and self.right.detected)

        lx, ly, rx, ry, windows = self._search(binary_warped, use_prior)
        left_fit = fit_polynomial(lx, ly)
        right_fit = fit_polynomial(rx, ry)
        ok = self._sane(left_fit, right_fit)

        if not ok and use_prior:
            lx, ly, rx, ry, windows = self._search(binary_warped, False)
            left_fit = fit_polynomial(lx, ly)
            right_fit = fit_polynomial(rx, ry)
            ok = self._sane(left_fit, right_fit)

        if not self.smooth:
            self.left.reset()
            self.right.reset()
            if left_fit is not None and right_fit is not None:
                ok = True

        if ok:
            self.left.update(left_fit)
            self.right.update(right_fit)
        else:
            self.left.update(None)
            self.right.update(None)

        self.last_debug = (lx, ly, rx, ry, windows)
        return self.left.best_fit, self.right.best_fit

    def debug_image(self, binary_warped):
        out = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        if self.last_debug is None:
            return out
        lx, ly, rx, ry, windows = self.last_debug
        for (p1, p2) in windows:
            cv2.rectangle(out, p1, p2, (0, 255, 0), 2)
        out[ly, lx] = (0, 0, 255)
        out[ry, rx] = (255, 0, 0)
        h = binary_warped.shape[0]
        ploty = np.linspace(0, h - 1, h)
        for fit in (self.left.best_fit, self.right.best_fit):
            if fit is None:
                continue
            pts = np.int32(np.column_stack((np.polyval(fit, ploty), ploty)))
            cv2.polylines(out, [pts], False, (0, 255, 255), 4)
        return out
