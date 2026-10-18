import cv2
import numpy as np
import pytest

from perspective import (PRESETS, PerspectiveTransform, estimate_vanishing_point,
                         src_from_vanishing_point)


def test_warp_keeps_size():
    t = PerspectiveTransform(1280, 720)
    img = np.zeros((720, 1280, 3), np.uint8)
    assert t.warp(img).shape == img.shape
    assert t.unwarp(img).shape == img.shape


def test_src_points_map_to_dst():
    t = PerspectiveTransform(1280, 720)
    mapped = cv2.perspectiveTransform(t.src.reshape(-1, 1, 2), t.M).reshape(-1, 2)
    assert np.allclose(mapped, t.dst, atol=1e-3)


def test_inverse_matrix():
    t = PerspectiveTransform(960, 540)
    assert np.allclose(t.M @ t.Minv / (t.M @ t.Minv)[2, 2], np.eye(3), atol=1e-3)


def test_lane_width_px():
    assert PerspectiveTransform(1280, 720).lane_width_px() == pytest.approx(640)
    assert PerspectiveTransform(960, 540).lane_width_px() == pytest.approx(480)


def test_unknown_resolution_uses_default_preset():
    t = PerspectiveTransform(640, 360)
    expected = np.float32([(x * 640, y * 360) for x, y in PRESETS[(1280, 720)]["src"]])
    assert np.allclose(t.src, expected)


def test_wrong_number_of_points_raises():
    with pytest.raises(cv2.error):
        PerspectiveTransform(1280, 720, src=[(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)])


def test_collinear_points_give_degenerate_matrix():
    t = PerspectiveTransform(1280, 720, src=[(0.1, 0.5), (0.2, 0.5), (0.3, 0.5), (0.4, 0.5)])
    assert not np.isfinite(t.M).all() or abs(np.linalg.det(t.M)) < 1e-6


def test_vanishing_point_from_preset_gives_preset_src():
    w, h = 1280, 720
    (tlx, ty), (trx, _), (brx, by), (blx, _) = PRESETS[(w, h)]["src"]
    vy = ty + (by - ty) * (trx - tlx) / ((trx - tlx) - (brx - blx))
    src = src_from_vanishing_point(w, h, (0.5, vy))
    assert np.allclose(src, PRESETS[(w, h)]["src"], atol=1e-3)


def test_estimate_vanishing_point_on_synthetic_lines():
    w, h = 1280, 720
    vp = (640, 420)
    binary = np.zeros((h, w), np.uint8)
    cv2.line(binary, vp, (200, h), 1, 6)
    cv2.line(binary, vp, (1080, h), 1, 6)
    result = estimate_vanishing_point(binary)
    assert result is not None
    assert result[0] == pytest.approx(vp[0] / w, abs=0.02)
    assert result[1] == pytest.approx(vp[1] / h, abs=0.02)


def test_estimate_vanishing_point_empty_image():
    assert estimate_vanishing_point(np.zeros((720, 1280), np.uint8)) is None


def test_estimate_vanishing_point_single_line():
    binary = np.zeros((720, 1280), np.uint8)
    cv2.line(binary, (640, 420), (200, 720), 1, 6)
    assert estimate_vanishing_point(binary) is None
