import cv2
import os
import chess
import numpy as np
import json
import copy
import shutil
import chess.pgn

from src.frame_sequence import extract_move_transition_frames
from src.board_detector import warp_to_board
from src.square_mapper import split_board_into_squares
from src.occupancy_detector import compare_squares, get_changed_squares
from src.inference import infer_move
from src.calibrate import calibrate_video


VIDEO_FOLDER = "videos"
OUTPUT_FOLDER = "outputs"
DEBUG_FOLDER = "debug"
CONFIG_FILE = "config.json"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)


def process_video(video_path):

    video_name = os.path.basename(video_path)

    print(f"\n=== Processing: {video_name} ===")

    with open(CONFIG_FILE, "r") as f:
        calibration_data = json.load(f)

    if video_path not in calibration_data:

        print("\nNo calibration found. Launching calibration tool...")

        success = calibrate_video(video_path)

        if not success:
            print("\nCalibration failed.")
            return

        with open(CONFIG_FILE, "r") as f:
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

    stable_frames = extract_move_transition_frames(video_path)

    print(f"Stable Frames Found: {len(stable_frames)}")

    board_detection_failures = 0
    noisy_frames = 0
    legal_move_failures = 0
    confirmed_moves = 0

    previous_squares = None

    for index, frame in enumerate(stable_frames):

        board = warp_to_board(frame, reference_pts)

        if board is None:
            board_detection_failures += 1
            continue

        # rotate because white pieces appear on left side in videos
        board = cv2.rotate(board, cv2.ROTATE_90_CLOCKWISE)

        # save debug board images for first few frames
        if index <= 3:
            cv2.imwrite(
                os.path.join(DEBUG_FOLDER, f"{video_name}_board_{index}.jpg"),
                board
            )

        squares = split_board_into_squares(board)

        # initialize baseline on first frame
        if previous_squares is None:
            previous_squares = copy.deepcopy(squares)
            continue

        diff_scores = compare_squares(previous_squares, squares)
        changed_squares = get_changed_squares(diff_scores)

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

        elif status == "legal_move":
            confirmed_moves += 1
            pgn_node = pgn_node.add_variation(move)
            previous_squares = copy.deepcopy(squares)

    print("\n===== VIDEO SUMMARY =====")
    print(f"Stable Frames:          {len(stable_frames)}")
    print(f"Confirmed Moves:        {confirmed_moves}")
    print(f"Noisy Frames:           {noisy_frames}")
    print(f"Legal Move Failures:    {legal_move_failures}")
    print(f"Board Detection Failures: {board_detection_failures}")

    pgn_path = os.path.join(
        OUTPUT_FOLDER,
        video_name.replace(".mp4", ".pgn")
    )

    with open(pgn_path, "w") as pgn_file:
        exporter = chess.pgn.FileExporter(pgn_file)
        pgn_game.accept(exporter)

    print(f"\nPGN saved to: {pgn_path}")

    # copy best board image to screenshots
    src_img = os.path.join(DEBUG_FOLDER, f"{video_name}_board_1.jpg")
    dst_img = os.path.join("screenshots", f"{video_name}_board.jpg")
    os.makedirs("screenshots", exist_ok=True)
    if os.path.exists(src_img):
        shutil.copy(src_img, dst_img)
        print(f"Screenshot saved to: {dst_img}")


# ─── Entry Point ───────────────────────────────────────────────────────────────

print("\n=== Chess Video to PGN Pipeline ===")
print("\n1. Process existing video")
print("2. Upload new video")

choice = input("\nChoose option: ")

if choice == "1":

    video_files = sorted([
        f for f in os.listdir(VIDEO_FOLDER)
        if f.endswith(".mp4")
    ])

    print("\nAvailable videos:\n")
    for index, video in enumerate(video_files):
        print(f"  {index + 1}. {video}")

    selected = int(input("\nSelect video number: ")) - 1
    video_path = os.path.join(VIDEO_FOLDER, video_files[selected])
    process_video(video_path)

elif choice == "2":

    source_path = input("\nEnter full path to video: ")
    file_name = os.path.basename(source_path)
    destination_path = os.path.join(VIDEO_FOLDER, file_name)
    shutil.copy(source_path, destination_path)
    print(f"\nVideo copied to: {destination_path}")
    process_video(destination_path)

else:
    print("\nInvalid option.")
