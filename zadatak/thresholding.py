import cv2
import numpy as np


def abs_sobel_thresh(channel, orient="x", ksize=3, thresh=(20, 100)):
    if orient == "x":
        sobel = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=ksize)
    else:
        sobel = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobel = np.absolute(sobel)
    scaled = np.uint8(255 * abs_sobel / max(np.max(abs_sobel), 1e-6))
    return (scaled >= thresh[0]) & (scaled <= thresh[1])


def dir_threshold(channel, ksize=15, thresh=(0.7, 1.3)):
    sx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=ksize)
    sy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=ksize)
    direction = np.arctan2(np.absolute(sy), np.absolute(sx))
    return (direction >= thresh[0]) & (direction <= thresh[1])


def ridge_mask(channel, thresh=25):
    h, w = channel.shape
    kw = max(int(w * 0.03), 9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
    tophat = cv2.morphologyEx(channel, cv2.MORPH_TOPHAT, kernel)
    return tophat >= thresh


def yellow_mask(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    b = lab[:, :, 2].astype(np.float32)
    b_norm = b - np.median(b)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, (15, 70, 100), (35, 255, 255)) > 0
    return (b_norm > 18) | hsv_mask


def white_mask(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l = hls[:, :, 1]
    s = hls[:, :, 2]
    h, w = l.shape
    road = l[int(h * 0.6):, int(w * 0.2):int(w * 0.8)]
    adaptive = max(np.percentile(road, 97), 190)
    return (l >= adaptive) & (s < 120)


def binary_image(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l = hls[:, :, 1]
    s = hls[:, :, 2]

    grad_x = abs_sobel_thresh(l, "x", ksize=3, thresh=(20, 150))
    ridge = ridge_mask(l)
    s_binary = (s >= 170) & (l >= 80) & ridge

    color = yellow_mask(img) | white_mask(img)
    combined = color | (grad_x & ridge) | s_binary

    return combined.astype(np.uint8)


def color_binary_stack(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l = hls[:, :, 1]
    grad = abs_sobel_thresh(l, "x", ksize=3, thresh=(20, 150)) & ridge_mask(l)
    color = yellow_mask(img) | white_mask(img)
    stack = np.dstack((np.zeros_like(l), grad, color)).astype(np.uint8) * 255
    return stack
