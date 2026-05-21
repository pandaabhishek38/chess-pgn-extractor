import cv2
import numpy as np


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def perspective_transform(image, pts):
    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)

    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # crop a little inward to remove board border
    crop_x = int(maxWidth * 0.02)
    crop_y = int(maxHeight * 0.02)

    warped = warped[
        crop_y:maxHeight - crop_y,
        crop_x:maxWidth - crop_x
    ]

    return warped


def detect_board(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 30, 100)

    cv2.imwrite("debug/edges.jpg", edges)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    debug_image = frame.copy()

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < 50000:
            continue

        # rotated rectangle around contour
        x, y, w, h = cv2.boundingRect(contour)

        aspect_ratio = w / float(h)

        if aspect_ratio < 0.7 or aspect_ratio > 1.5:
            continue

        cv2.rectangle(
            debug_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            5
        )

        cv2.imwrite("debug/contours.jpg", debug_image)

        padding = 40

        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)

        x2 = min(x + w + padding, frame.shape[1])
        y2 = min(y + h + padding, frame.shape[0])

        board = frame[y1:y2, x1:x2]

        return board, (x, y, w, h)

    return None, None
