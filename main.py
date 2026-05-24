import cv2
import os
import chess
import numpy as np
import json
import copy

from src.frame_sequence import (
    extract_move_transition_frames
)

from src.board_detector import (
    warp_to_board
)

from src.square_mapper import (
    split_board_into_squares
)

from src.occupancy_detector import (
    compare_squares,
    get_changed_squares
)

from src.inference import infer_move


VIDEO_FOLDER = "videos"

video_files = sorted(os.listdir(VIDEO_FOLDER))


for video_name in video_files:

    if not video_name.endswith(".mp4"):
        continue

    if video_name != "game_5.mp4":
        continue

    video_path = os.path.join(
        VIDEO_FOLDER,
        video_name
    )

    with open(
        "config.json",
        "r"
    ) as f:

        calibration_data = json.load(f)

    reference_pts = np.array(
        calibration_data[video_path],
        dtype=np.float32
    )

    print(f"\nProcessing: {video_name}")

    game_board = chess.Board()

    stable_frames = extract_move_transition_frames(
        video_path
    )

    # summary counters
    board_detection_failures = 0
    noisy_frames = 0
    legal_move_failures = 0
    confirmed_moves = 0

    print(
        f"Stable Frames Found: "
        f"{len(stable_frames)}"
    )

    previous_squares = None

    for index, frame in enumerate(stable_frames):

        board = warp_to_board(
            frame,
            reference_pts
        )

        board = cv2.rotate(board, cv2.ROTATE_90_CLOCKWISE)

        if index <= 5:

            print(
                f"Frame {index} calibration corners: "
                f"{reference_pts}"
            )

        # debug warped boards
        if index <= 5:

            cv2.imwrite(
                f"debug/board_{index}.jpg",
                board
            )

        squares = split_board_into_squares(
            board
        )

        # debug sample squares
        if index <= 5:

            cv2.imwrite(
                f"debug/e4_{index}.jpg",
                squares["e4"]
            )

            cv2.imwrite(
                f"debug/a1_{index}.jpg",
                squares["a1"]
            )

        # first frame initializes
        if previous_squares is None:

            previous_squares = copy.deepcopy(squares)

            continue

        # compare current squares
        # against previous frame squares
        diff_scores = compare_squares(
            previous_squares,
            squares
        )

        changed_squares = get_changed_squares(
            diff_scores
        )

        print(
            f"Frame {index}: "
            f"{len(changed_squares)} changed: "
            f"{sorted(changed_squares)}"
        )

        top_diffs = sorted(
            diff_scores.items(),
            key=lambda x: -x[1]
        )[:5]

        print(
            f"Top diffs: "
            f"{top_diffs}"
        )

        sorted_diffs = sorted(
            diff_scores.items(), key=lambda x: -x[1]
        )

        top_vals = [v for _, v in sorted_diffs[:5]]

        top_n = [
            sq for sq, _ in sorted(
                diff_scores.items(), key=lambda x: -x[1]
            )[:6]
        ]

        if len(changed_squares) == 2:
            status, move = infer_move(changed_squares, game_board)
        else:
            status, move = infer_move(top_n, game_board)

        if status == "noisy":

            noisy_frames += 1

        elif status == "legal_failure":

            legal_move_failures += 1
            attempted = changed_squares if len(changed_squares) == 2 else top_n
            print(f"  Legal failure: tried {attempted}")
            print(f"  Turn: {'White' if game_board.turn else 'Black'}")
            print(f"  Legal moves: {[m.uci() for m in list(game_board.legal_moves)[:8]]}")

        elif status == "legal_move":

            confirmed_moves += 1

            print(f"Detected Move: {move}")

            previous_squares = copy.deepcopy(squares)

    print("\n===== VIDEO SUMMARY =====")

    print(
        f"Stable Frames: "
        f"{len(stable_frames)}"
    )

    print(
        f"Confirmed Moves: "
        f"{confirmed_moves}"
    )

    print(
        f"Noisy Frames: "
        f"{noisy_frames}"
    )

    print(
        f"Legal Move Failures: "
        f"{legal_move_failures}"
    )

    print(
        f"Board Detection Failures: "
        f"{board_detection_failures}"
    )
