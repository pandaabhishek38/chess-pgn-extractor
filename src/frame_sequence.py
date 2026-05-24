import cv2
import numpy as np


def extract_move_transition_frames(
    video_path,
    movement_threshold=250000,
    cooldown_frames=30
):
    """Extract frames immediately after movement stops — one per move transition."""

    cap = cv2.VideoCapture(video_path)

    success, prev_frame = cap.read()

    if not success:
        return []

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    stable_frames = []
    movement_active = False
    cooldown = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        movement_score = np.sum(cv2.absdiff(prev_gray, gray))

        if movement_score > movement_threshold:
            movement_active = True

        elif movement_active and cooldown == 0:
            stable_frames.append(frame)
            movement_active = False
            cooldown = cooldown_frames

        if cooldown > 0:
            cooldown -= 1

        prev_gray = gray

    cap.release()

    return stable_frames
