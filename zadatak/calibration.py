import glob
import os

import cv2
import numpy as np


def calibrate_camera(image_dir, nx=9, ny=6):
    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    img_size = None

    for fname in sorted(glob.glob(os.path.join(image_dir, "*.jpg"))):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if not found:
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)

    _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
    return mtx, dist, img_size, len(objpoints)


def load_or_calibrate(image_dir, cache_file):
    if os.path.exists(cache_file):
        data = np.load(cache_file)
        return data["mtx"], data["dist"], tuple(data["img_size"])
    mtx, dist, img_size, _ = calibrate_camera(image_dir)
    np.savez(cache_file, mtx=mtx, dist=dist, img_size=np.array(img_size))
    return mtx, dist, img_size


class Undistorter:
    def __init__(self, mtx, dist, img_size):
        self.mtx = mtx
        self.dist = dist
        self.img_size = img_size
        self._cache = {}

    def _matrix_for(self, width, height):
        key = (width, height)
        if key not in self._cache:
            sx = width / self.img_size[0]
            sy = height / self.img_size[1]
            mtx = self.mtx.copy()
            mtx[0, 0] *= sx
            mtx[0, 2] *= sx
            mtx[1, 1] *= sy
            mtx[1, 2] *= sy
            self._cache[key] = mtx
        return self._cache[key]

    def __call__(self, img):
        h, w = img.shape[:2]
        mtx = self._matrix_for(w, h)
        return cv2.undistort(img, mtx, self.dist, None, mtx)
