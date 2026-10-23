import numpy as np
import pytest

from lane_detection import (LaneDetector, LaneGeometry, Line, find_lane_pixels_sliding_window,
                            fit_polynomial)


W, H = 1280, 720


def lane_binary(left=320, right=960):
    binary = np.zeros((H, W), np.uint8)
    binary[:, left - 5:left + 5] = 1
    binary[:, right - 5:right + 5] = 1
    return binary


@pytest.fixture
def geometry():
    return LaneGeometry(W, H, 640)


def test_sliding_window_finds_both_lines():
    lx, ly, rx, ry, windows = find_lane_pixels_sliding_window(lane_binary())
    assert len(windows) == 18
    assert abs(np.mean(lx) - 320) < 2
    assert abs(np.mean(rx) - 960) < 2


def test_fit_vertical_line():
    y = np.arange(H)
    fit = fit_polynomial(np.full(H, 300.0), y)
    assert fit == pytest.approx([0, 0, 300], abs=1e-6)


def test_fit_too_few_points_returns_none():
    assert fit_polynomial(np.array([1, 2, 3]), np.array([1, 2, 3])) is None


def test_straight_lane_has_large_curvature(geometry):
    assert geometry.curvature(np.array([0.0, 0.0, 320.0])) > 1e6


def test_curved_lane_radius(geometry):
    fit = np.array([2e-4, 0.0, 320.0])
    assert 100 < geometry.curvature(fit) < 5000


def test_vehicle_offset(geometry):
    centered = geometry.vehicle_offset(np.array([0, 0, 320.0]), np.array([0, 0, 960.0]))
    assert centered == pytest.approx(0)
    shifted = geometry.vehicle_offset(np.array([0, 0, 220.0]), np.array([0, 0, 860.0]))
    assert shifted == pytest.approx(100 * 3.7 / 640)


def test_lane_width(geometry):
    min_w, max_w = geometry.lane_width_m(np.array([0, 0, 320.0]), np.array([0, 0, 960.0]))
    assert min_w == pytest.approx(3.7)
    assert max_w == pytest.approx(3.7)


def test_line_averages_and_resets():
    line = Line()
    line.update(np.array([0, 0, 300.0]))
    line.update(np.array([0, 0, 310.0]))
    assert 300 < line.best_fit[2] < 310
    for _ in range(11):
        line.update(None)
    assert line.best_fit is None


def test_detector_finds_lane(geometry):
    left, right = LaneDetector(geometry).detect(lane_binary())
    assert left[2] == pytest.approx(320, abs=2)
    assert right[2] == pytest.approx(960, abs=2)


def test_detector_rejects_too_narrow_lane(geometry):
    left, right = LaneDetector(geometry).detect(lane_binary(560, 720))
    assert left is None and right is None


def test_detector_empty_image(geometry):
    left, right = LaneDetector(geometry).detect(np.zeros((H, W), np.uint8))
    assert left is None and right is None
