import cv2
import numpy as np

from thresholding import binary_image, ridge_mask, white_mask, yellow_mask


def road_image(line_color):
    img = np.full((720, 1280, 3), 70, np.uint8)
    cv2.line(img, (400, 720), (600, 450), line_color, 15)
    return img


def test_binary_image_shape_and_values():
    result = binary_image(road_image((255, 255, 255)))
    assert result.shape == (720, 1280)
    assert result.dtype == np.uint8
    assert set(np.unique(result)) <= {0, 1}


def test_white_line_detected():
    mask = white_mask(road_image((255, 255, 255)))
    assert mask[650, 440:470].any()
    assert not mask[650, 900:1000].any()


def test_yellow_line_detected():
    mask = yellow_mask(road_image((0, 200, 255)))
    assert mask[650, 440:470].any()
    assert not mask[650, 900:1000].any()


def test_empty_road_gives_empty_binary():
    img = np.full((720, 1280, 3), 70, np.uint8)
    assert binary_image(img).sum() == 0


def test_ridge_ignores_step_edge():
    channel = np.full((100, 400), 60, np.uint8)
    channel[:, 200:] = 200
    assert ridge_mask(channel).any(axis=0).sum() <= 1


def test_ridge_finds_thin_line():
    channel = np.full((100, 400), 60, np.uint8)
    channel[:, 200:210] = 200
    assert ridge_mask(channel)[:, 200:210].all()
