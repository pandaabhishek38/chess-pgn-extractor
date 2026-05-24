import cv2
import os
import chess
import numpy as np
import json
import copy
import shutil
import chess.pgn

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

from src.inference import (
    infer_move
)

from src.calibrate import calibrate_video


VIDEO_FOLDER = "videos"

OUTPUT_FOLDER = "outputs"

DEBUG_FOLDER = "debug"

CONFIG_FILE = "config.json"

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)

os.makedirs(
    DEBUG_FOLDER,
    exist_ok=True
)


def process_video(video_path):

    video_name = os.path.basename(
        video_path
    )

    print(
        f"\n=== Processing: {video_name} ==="
    )

    with open(
        "config.json",
        "r"
    ) as f:

        calibration_data = json.load(f)

    if video_path not in calibration_data:

        print(
            "\nNo calibration found "
            "for this video."
        )

        print(
            "Launching calibration..."
        )

        success = calibrate_video(
            video_path
        )

        if not success:

            print(
                "\nCalibration failed."
            )

            return

        with open(
            CONFIG_FILE,
            "r"
        ) as f:

            calibration_data = json.load(f)

    reference_pts = np.array(
        calibration_data[video_path],
        dtype=np.float32
    )

    game_board = chess.Board()

    pgn_game = chess.pgn.Game()

    pgn_game.headers["Event"] = video_name
    pgn_game.headers["White"] = "Detected"
    pgn_game.headers["Black"] = "Detected"

    pgn_node = pgn_game

    stable_frames = extract_move_transition_frames(
        video_path
    )

    # summary counters
    board_detection_failures = 0
    noisy_frames = 0
    legal_move_failures = 0
    confirmed_moves = 0

    print(
        f"\nStable Frames Found: "
        f"{len(stable_frames)}"
    )

    previous_squares = None

    for index, frame in enumerate(stable_frames):

        board = warp_to_board(
            frame,
            reference_pts
        )

        if board is None:

            board_detection_failures += 1

            continue

        # rotate because white pieces
        # appear on left side in videos
        board = cv2.rotate(
            board,
            cv2.ROTATE_90_CLOCKWISE
        )

        # debug warped boards
        if index <= 3:

            print(
                f"\nFrame {index} calibration corners:"
            )

            print(reference_pts)

            cv2.imwrite(
                os.path.join(
                    DEBUG_FOLDER,
                    f"{video_name}_board_{index}.jpg"
                ),
                board
            )

        squares = split_board_into_squares(
            board
        )

        # debug sample squares
        if index <= 3:

            cv2.imwrite(
                os.path.join(
                    DEBUG_FOLDER,
                    f"{video_name}_e4_{index}.jpg"
                ),
                squares["e4"]
            )

            cv2.imwrite(
                os.path.join(
                    DEBUG_FOLDER,
                    f"{video_name}_a1_{index}.jpg"
                ),
                squares["a1"]
            )

        # initialize baseline
        if previous_squares is None:

            previous_squares = copy.deepcopy(
                squares
            )

            continue

        # compare current board
        # against previous accepted board
        diff_scores = compare_squares(
            previous_squares,
            squares
        )

        changed_squares = get_changed_squares(
            diff_scores
        )

        print(
            f"\nFrame {index}: "
            f"{len(changed_squares)} changed"
        )

        print(
            f"Changed Squares: "
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

        # fallback candidate set
        top_n = [
            sq for sq, _ in sorted(
                diff_scores.items(),
                key=lambda x: -x[1]
            )[:6]
        ]

        # exact move candidates
        if len(changed_squares) == 2:

            status, move = infer_move(
                changed_squares,
                game_board
            )

        # noisy candidate fallback
        else:

            status, move = infer_move(
                top_n,
                game_board
            )

        if status == "noisy":

            noisy_frames += 1

        elif status == "legal_failure":

            legal_move_failures += 1

            attempted = (
                changed_squares
                if len(changed_squares) == 2
                else top_n
            )

            print(
                f"Legal failure: "
                f"tried {attempted}"
            )

            print(
                f"Turn: "
                f"{'White' if game_board.turn else 'Black'}"
            )

        elif status == "legal_move":

            confirmed_moves += 1

            print(
                f"Detected Move: {move}"
            )

            # update PGN
            pgn_node = pgn_node.add_variation(
                move
            )

            # update accepted baseline
            previous_squares = copy.deepcopy(
                squares
            )

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

    pgn_path = os.path.join(
        OUTPUT_FOLDER,
        video_name.replace(".mp4", ".pgn")
    )

    with open(
        pgn_path,
        "w"
    ) as pgn_file:

        exporter = chess.pgn.FileExporter(
            pgn_file
        )

        pgn_game.accept(exporter)

    print(
        f"\nPGN saved to: "
        f"{pgn_path}"
    )


print("\n=== Chess Video to PGN Pipeline ===")

print("\n1. Use existing video")

print("2. Upload new video")

choice = input(
    "\nChoose option: "
)

if choice == "1":

    video_files = sorted([
        f for f in os.listdir(
            VIDEO_FOLDER
        )
        if f.endswith(".mp4")
    ])

    print("\nAvailable videos:\n")

    for index, video in enumerate(
        video_files
    ):

        print(
            f"{index + 1}. {video}"
        )

    selected = int(
        input(
            "\nSelect video number: "
        )
    ) - 1

    selected_video = video_files[
        selected
    ]

    video_path = os.path.join(
        VIDEO_FOLDER,
        selected_video
    )

    process_video(video_path)

elif choice == "2":

    source_path = input(
        "\nEnter full path to video: "
    )

    file_name = os.path.basename(
        source_path
    )

    destination_path = os.path.join(
        VIDEO_FOLDER,
        file_name
    )

    shutil.copy(
        source_path,
        destination_path
    )

    print(
        f"\nUploaded video to:"
    )

    print(destination_path)

    process_video(destination_path)

else:

    print(
        "\nInvalid option selected."
    )
