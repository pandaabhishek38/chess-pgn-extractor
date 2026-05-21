import cv2
import numpy as np


def find_stable_frame(video_path, output_name, stable_threshold=250000, stable_frames_needed=8):

    cap = cv2.VideoCapture(video_path)

    success, prev_frame = cap.read()

    if not success:
        return None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    stable_count = 0

    stable_frame = None

    while True:

        success, frame = cap.read()

        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_gray, gray)

        movement_score = np.sum(diff)

        #print("Movement:", movement_score)

        if movement_score < stable_threshold:

            stable_count += 1

        else:

            stable_count = 0

        if stable_count >= stable_frames_needed:

            print("Stable frame selected")

            cv2.imwrite(
                f"debug/{output_name}_stable.jpg",
                frame
            )

            stable_frame = frame
            break

        prev_gray = gray

    cap.release()

    return stable_frame
