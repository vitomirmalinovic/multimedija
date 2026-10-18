import glob
import os

import cv2
import numpy as np
import pytest

from calibration import Undistorter, calibrate_camera, load_or_calibrate


CAMERA_CAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "camera_cal")


@pytest.fixture(scope="module")
def calibration():
    return calibrate_camera(CAMERA_CAL_DIR)


def test_uses_images_from_folder(calibration):
    _, _, img_size, used = calibration
    total = len(glob.glob(os.path.join(CAMERA_CAL_DIR, "*.jpg")))
    assert total == 20
    assert 15 <= used <= total
    assert img_size == (1280, 720)


def test_camera_matrix_shape_and_values(calibration):
    mtx, dist, _, _ = calibration
    assert mtx.shape == (3, 3)
    assert dist.shape == (1, 5)
    assert mtx[0, 0] > 0 and mtx[1, 1] > 0
    assert 0 < mtx[0, 2] < 1280
    assert 0 < mtx[1, 2] < 720
    assert mtx[2, 2] == pytest.approx(1.0)


def test_empty_folder_raises(tmp_path):
    with pytest.raises(cv2.error):
        calibrate_camera(str(tmp_path))


def test_folder_without_chessboard_raises(tmp_path):
    cv2.imwrite(str(tmp_path / "blank.jpg"), np.zeros((100, 100, 3), np.uint8))
    with pytest.raises(cv2.error):
        calibrate_camera(str(tmp_path))


def test_wrong_pattern_size_raises():
    with pytest.raises(cv2.error):
        calibrate_camera(CAMERA_CAL_DIR, nx=20, ny=20)


def test_load_or_calibrate_creates_cache(tmp_path, calibration):
    mtx, dist, img_size, _ = calibration
    cache = tmp_path / "cal.npz"
    np.savez(str(cache), mtx=mtx, dist=dist, img_size=np.array(img_size))
    loaded_mtx, loaded_dist, loaded_size = load_or_calibrate("nepostojeci_folder", str(cache))
    assert np.allclose(loaded_mtx, mtx)
    assert np.allclose(loaded_dist, dist)
    assert loaded_size == img_size


def test_undistorter_keeps_size(calibration):
    mtx, dist, img_size, _ = calibration
    undistort = Undistorter(mtx, dist, img_size)
    for w, h in [(1280, 720), (960, 540)]:
        img = np.full((h, w, 3), 128, np.uint8)
        assert undistort(img).shape == (h, w, 3)


def test_undistorter_scales_matrix(calibration):
    mtx, dist, img_size, _ = calibration
    undistort = Undistorter(mtx, dist, img_size)
    scaled = undistort._matrix_for(640, 360)
    assert scaled[0, 0] == pytest.approx(mtx[0, 0] / 2)
    assert scaled[1, 2] == pytest.approx(mtx[1, 2] / 2)
