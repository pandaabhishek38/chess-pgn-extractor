import cv2
import numpy as np


def is_square_occupied(square_image, edge_threshold=70000):

    gray = cv2.cvtColor(square_image, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape

    margin_h = int(h * 0.15)
    margin_w = int(w * 0.15)

    gray = gray[
        margin_h:h - margin_h,
        margin_w:w - margin_w
    ]

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    edge_score = np.sum(edges)

    variance_score = np.var(gray)

    occupied = (
        edge_score > edge_threshold
        or variance_score > 1200
    )

    return occupied, edge_score, variance_score
