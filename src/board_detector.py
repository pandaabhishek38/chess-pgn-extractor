import cv2
import numpy as np


BOARD_SIZE = 800


def order_points(pts):
    """Order corner points as top-left, top-right, bottom-right, bottom-left."""

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def warp_to_board(frame, pts):
    """Warp perspective of frame to a square board image."""

    rect = order_points(pts)

    dst = np.array([
        [0, 0],
        [BOARD_SIZE - 1, 0],
        [BOARD_SIZE - 1, BOARD_SIZE - 1],
        [0, BOARD_SIZE - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(frame, M, (BOARD_SIZE, BOARD_SIZE))

    return warped
