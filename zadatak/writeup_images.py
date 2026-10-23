import os

import cv2
import numpy as np

from calibration import calibrate_camera
from pipeline import CAMERA_CAL_DIR, LanePipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")
OUT_DIR = os.path.join(BASE_DIR, "output_images", "writeup")


def side_by_side(left, right, left_title, right_title):
    if left.ndim == 2:
        left = cv2.cvtColor(left * 255, cv2.COLOR_GRAY2BGR)
    if right.ndim == 2:
        right = cv2.cvtColor(right * 255, cv2.COLOR_GRAY2BGR)
    pair = np.hstack((left, np.full((left.shape[0], 20, 3), 255, np.uint8), right))
    header = np.full((60, pair.shape[1], 3), 255, np.uint8)
    cv2.putText(header, left_title, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(header, right_title, (left.shape[1] + 40, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack((header, pair))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mtx, dist, _, used = calibrate_camera(CAMERA_CAL_DIR)
    print("Kalibracija: iskorisceno {} slika".format(used))

    chess = cv2.imread(os.path.join(CAMERA_CAL_DIR, "calibration1.jpg"))
    cv2.imwrite(os.path.join(OUT_DIR, "01_undistort_chessboard.jpg"),
                side_by_side(chess, cv2.undistort(chess, mtx, dist, None, mtx), "Original", "Undistorted"))

    img = cv2.imread(os.path.join(ROOT_DIR, "test_images", "test2.jpg"))
    pipeline = LanePipeline(smooth=False)
    undist, binary, warped, left_fit, right_fit = pipeline.stages(img)

    cv2.imwrite(os.path.join(OUT_DIR, "02_undistort_road.jpg"),
                side_by_side(img, undist, "Original", "Undistorted"))
    cv2.imwrite(os.path.join(OUT_DIR, "03_binary.jpg"),
                side_by_side(undist, binary, "Undistorted", "Binary"))

    straight = pipeline.undistort(cv2.imread(os.path.join(ROOT_DIR, "test_images", "straight_lines1.jpg")))
    t = pipeline.transform
    cv2.imwrite(os.path.join(OUT_DIR, "04_warp.jpg"),
                side_by_side(t.draw_src(straight), t.draw_dst(t.warp(straight)), "Source points", "Warped"))

    cv2.imwrite(os.path.join(OUT_DIR, "05_fit.jpg"),
                side_by_side(warped, pipeline.detector.debug_image(warped), "Warped binary", "Sliding windows + fit"))

    cv2.imwrite(os.path.join(OUT_DIR, "06_result.jpg"), pipeline.process(img))
    print("Slike sacuvane u", OUT_DIR)


if __name__ == "__main__":
    main()
