import cv2
import os

from src.board_detector import detect_board
from src.square_mapper import (
    split_board_into_squares,
    save_squares
)

VIDEO_FOLDER = "videos"

OUTPUT_FOLDER = "debug"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

video_files = sorted(os.listdir(VIDEO_FOLDER))

for video_name in video_files:

    # skip hidden/system files
    if not video_name.endswith(".mp4"):
        continue

    video_path = os.path.join(VIDEO_FOLDER, video_name)

    print(f"\nProcessing: {video_name}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open video")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    middle_frame = total_frames // 2

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)

    success, frame = cap.read()

    if not success:
        print("Could not read frame")
        continue

    height = frame.shape[0]

    cropped_frame = frame[:int(height * 0.42), :]

    board, pts = detect_board(cropped_frame)

    base_name = os.path.splitext(video_name)[0]

    cv2.imwrite(
        f"{OUTPUT_FOLDER}/{base_name}_frame.jpg",
        frame
    )

    cv2.imwrite(
        f"{OUTPUT_FOLDER}/{base_name}_cropped.jpg",
        cropped_frame
    )

    if board is not None:

        cv2.imwrite(
            f"{OUTPUT_FOLDER}/{base_name}_board.jpg",
            board
        )

        squares = split_board_into_squares(board)

        square_output_dir = (
            f"{OUTPUT_FOLDER}/{base_name}_squares"
        )

        save_squares(
            squares,
            output_dir=square_output_dir
        )

        print("Board detected successfully!")
        print("Board split into 64 squares!")

    else:
        print("Board detection failed.")

    cap.release()
