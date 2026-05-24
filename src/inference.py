import chess


def infer_move(
    changed_squares,
    board
):

    # ignore noisy detections
    if len(changed_squares) < 2:

        return "noisy", None

    sq_set = set(changed_squares)

    for move in board.legal_moves:

        legal_from = chess.square_name(
            move.from_square
        )

        legal_to = chess.square_name(
            move.to_square
        )

        if (
            legal_from in sq_set
            and legal_to in sq_set
        ):

            board.push(move)

            return "legal_move", move

    return "legal_failure", None
