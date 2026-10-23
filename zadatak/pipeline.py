import os

import cv2
import numpy as np

from calibration import Undistorter, load_or_calibrate
from lane_detection import LaneDetector, LaneGeometry
from perspective import PerspectiveTransform, estimate_vanishing_point, src_from_vanishing_point
from thresholding import binary_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_CAL_DIR = os.path.join(BASE_DIR, "..", "camera_cal")
CALIBRATION_FILE = os.path.join(BASE_DIR, "calibration.npz")


class LanePipeline:
    def __init__(self, smooth=True, debug_view=False):
        mtx, dist, img_size = load_or_calibrate(CAMERA_CAL_DIR, CALIBRATION_FILE)
        self.undistort = Undistorter(mtx, dist, img_size)
        self.smooth = smooth
        self.debug_view = debug_view
        self.size = None
        self.vp_samples = []
        self.vp_frames = 0
        self.vp_done = False

    def _setup(self, width, height, src=None):
        self.size = (width, height)
        self.transform = PerspectiveTransform(width, height, src=src)
        self.geometry = LaneGeometry(width, height, self.transform.lane_width_px())
        self.detector = LaneDetector(self.geometry, smooth=self.smooth)

    def _update_vanishing_point(self, binary, max_frames=20, min_samples=6):
        if self.vp_done:
            return
        vp = estimate_vanishing_point(binary)
        if vp is not None:
            self.vp_samples.append(vp)
        self.vp_frames += 1
        if self.vp_frames < max_frames and (self.smooth or vp is None):
            return
        self.vp_done = True
        samples = np.array(self.vp_samples)
        needed = min_samples if self.smooth else 1
        if len(samples) < needed or samples.std(axis=0).max() > 0.02:
            return
        vp = tuple(np.median(samples, axis=0))
        self._setup(self.size[0], self.size[1], src=src_from_vanishing_point(*self.size, vp))

    def stages(self, img):
        h, w = img.shape[:2]
        if self.size != (w, h):
            self._setup(w, h)
        undist = self.undistort(img)
        binary = binary_image(undist)
        self._update_vanishing_point(binary)
        warped = self.transform.warp(binary, flags=cv2.INTER_NEAREST)
        left_fit, right_fit = self.detector.detect(warped)
        return undist, binary, warped, left_fit, right_fit

    def draw_lane(self, undist, left_fit, right_fit):
        h, w = undist.shape[:2]
        if left_fit is None or right_fit is None:
            return undist.copy()
        ploty = np.linspace(0, h - 1, h)
        left_x = np.polyval(left_fit, ploty)
        right_x = np.polyval(right_fit, ploty)

        overlay = np.zeros_like(undist)
        pts_left = np.column_stack((left_x, ploty))
        pts_right = np.flipud(np.column_stack((right_x, ploty)))
        pts = np.int32(np.vstack((pts_left, pts_right)))
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.polylines(overlay, [np.int32(pts_left)], False, (0, 0, 255), 20)
        cv2.polylines(overlay, [np.int32(pts_right)], False, (255, 0, 0), 20)

        unwarped = self.transform.unwarp(overlay)
        return cv2.addWeighted(undist, 1.0, unwarped, 0.35, 0)

    def annotate(self, img, left_fit, right_fit):
        if left_fit is None or right_fit is None:
            cv2.putText(img, "Traka nije detektovana", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            return img
        curvature = (self.geometry.curvature(left_fit) + self.geometry.curvature(right_fit)) / 2
        offset = self.geometry.vehicle_offset(left_fit, right_fit)
        side = "desno" if offset > 0 else "levo"
        curv_text = "Radijus krivine: {:.0f} m".format(curvature) if curvature < 5000 \
            else "Radijus krivine: pravac (>5000 m)"
        offset_text = "Pozicija vozila: {:.2f} m {} od centra".format(abs(offset), side)
        scale = img.shape[1] / 1280
        for i, text in enumerate((curv_text, offset_text)):
            pos = (int(30 * scale), int((50 + 45 * i) * scale))
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale,
                        (0, 0, 0), int(5 * scale) + 1, cv2.LINE_AA)
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.1 * scale,
                        (255, 255, 255), int(2 * scale) + 1, cv2.LINE_AA)
        return img

    def _with_debug(self, result, binary, warped):
        h, w = result.shape[:2]
        tw, th = w // 4, h // 4
        binary_rgb = np.dstack((binary, binary, binary)) * 255
        lane_dbg = self.detector.debug_image(warped)
        thumbs = [cv2.resize(binary_rgb, (tw, th)), cv2.resize(lane_dbg, (tw, th))]
        for i, t in enumerate(thumbs):
            x = w - (len(thumbs) - i) * (tw + 10)
            result[10:10 + th, x:x + tw] = t
        return result

    def process(self, img):
        undist, binary, warped, left_fit, right_fit = self.stages(img)
        result = self.draw_lane(undist, left_fit, right_fit)
        result = self.annotate(result, left_fit, right_fit)
        if self.debug_view:
            result = self._with_debug(result, binary, warped)
        return result
