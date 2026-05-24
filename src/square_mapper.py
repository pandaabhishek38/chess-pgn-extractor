import cv2
import os


FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


def split_board_into_squares(board_image):
    """Split warped board image into 64 named squares."""

    height, width = board_image.shape[:2]

    square_h = height // 8
    square_w = width // 8

    squares = {}

    for row in range(8):
        for col in range(8):
            y1 = row * square_h
            y2 = (row + 1) * square_h
            x1 = col * square_w
            x2 = (col + 1) * square_w

            square = board_image[y1:y2, x1:x2]
            square_name = f"{FILES[col]}{8 - row}"
            squares[square_name] = square

    return squares


def save_squares(squares, output_dir="debug/squares"):
    """Save all 64 squares as images for debugging."""

    os.makedirs(output_dir, exist_ok=True)

    for square_name, image in squares.items():
        cv2.imwrite(f"{output_dir}/{square_name}.jpg", image)
