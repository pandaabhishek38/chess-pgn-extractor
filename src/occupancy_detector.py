import cv2
import numpy as np


def compare_squares(squares_before, squares_after):
    """Compare two sets of squares and return pixel diff scores."""

    diff_scores = {}

    for square_name in squares_before:

        img_a = squares_before[square_name]
        img_b = squares_after[square_name]

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        h, w = gray_a.shape
        margin_h = int(h * 0.15)
        margin_w = int(w * 0.15)

        gray_a = gray_a[margin_h:h - margin_h, margin_w:w - margin_w]
        gray_b = gray_b[margin_h:h - margin_h, margin_w:w - margin_w]

        diff = cv2.absdiff(gray_a, gray_b)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        diff_scores[square_name] = float(np.sum(diff))

    return diff_scores


def get_changed_squares(diff_scores, change_threshold=50000):
    """Return list of squares that exceeded the change threshold."""

    return [
        square for square, score in diff_scores.items()
        if score > change_threshold
    ]
