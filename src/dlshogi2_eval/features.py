from __future__ import annotations

"""Feature encoding and move-label encoding for python-dlshogi2.

This module intentionally mirrors the reference logic in upstream
`pydlshogi2/features.py`, with minor reshaping for standalone use.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from ._compat import require_cshogi

# Fallback numeric values so the module can be imported without cshogi.
# These match the current python-dlshogi2 reference formulas:
#   FEATURES_NUM    = len(PIECE_TYPES) * 2 + sum(MAX_PIECES_IN_HAND) * 2 = 104
#   MOVE_PLANES_NUM = 20 + len(HAND_PIECES) = 27
#   MOVE_LABELS_NUM = MOVE_PLANES_NUM * 81 = 2187
FEATURES_NUM = 104
MOVE_PLANES_NUM = 27
MOVE_LABELS_NUM = 2187

# Move direction constants, preserved as ordinal labels.
(
    UP,
    UP_LEFT,
    UP_RIGHT,
    LEFT,
    RIGHT,
    DOWN,
    DOWN_LEFT,
    DOWN_RIGHT,
    UP2_LEFT,
    UP2_RIGHT,
    UP_PROMOTE,
    UP_LEFT_PROMOTE,
    UP_RIGHT_PROMOTE,
    LEFT_PROMOTE,
    RIGHT_PROMOTE,
    DOWN_PROMOTE,
    DOWN_LEFT_PROMOTE,
    DOWN_RIGHT_PROMOTE,
    UP2_LEFT_PROMOTE,
    UP2_RIGHT_PROMOTE,
) = range(20)


@dataclass(frozen=True)
class EncodedBoard:
    """Feature tensor for a single board snapshot."""

    features: np.ndarray  # [FEATURES_NUM, 9, 9], float32

    def as_batch(self) -> np.ndarray:
        return self.features[np.newaxis, ...]



def _refresh_shape_constants_from_cshogi() -> tuple[int, int, int]:
    cshogi = require_cshogi()
    features_num = len(cshogi.PIECE_TYPES) * 2 + sum(cshogi.MAX_PIECES_IN_HAND) * 2
    move_planes_num = 20 + len(cshogi.HAND_PIECES)
    move_labels_num = move_planes_num * 81
    return features_num, move_planes_num, move_labels_num


try:  # pragma: no cover - depends on optional dependency presence
    FEATURES_NUM, MOVE_PLANES_NUM, MOVE_LABELS_NUM = _refresh_shape_constants_from_cshogi()
except Exception:
    pass



def empty_feature_array(dtype: np.dtype = np.float32) -> np.ndarray:
    return np.zeros((FEATURES_NUM, 9, 9), dtype=dtype)



def make_input_features(board: Any, out: np.ndarray | None = None) -> np.ndarray:
    """Encode a `cshogi.Board` into python-dlshogi2 input features.

    Parameters
    ----------
    board:
        `cshogi.Board` instance.
    out:
        Optional preallocated array with shape `[FEATURES_NUM, 9, 9]`.
    """

    cshogi = require_cshogi()
    if out is None:
        out = empty_feature_array()
    if out.shape != (FEATURES_NUM, 9, 9):
        raise ValueError(f"expected out shape {(FEATURES_NUM, 9, 9)}, got {out.shape}")

    out.fill(0)

    # Board piece planes.
    if board.turn == cshogi.BLACK:
        board.piece_planes(out)
        pieces_in_hand = board.pieces_in_hand
    else:
        board.piece_planes_rotate(out)
        pieces_in_hand = reversed(board.pieces_in_hand)

    # Hand-piece planes.
    i = 28
    for hands in pieces_in_hand:
        for num, max_num in zip(hands, cshogi.MAX_PIECES_IN_HAND):
            out[i : i + num].fill(1)
            i += max_num

    return out



def encode_board(board: Any) -> EncodedBoard:
    return EncodedBoard(make_input_features(board))



def make_move_label(move: int, color: int) -> int:
    """Map a legal move to the dense policy index used by python-dlshogi2.

    This mirrors upstream `make_move_label` and keeps the dense policy layout
    stable for comparisons against reference checkpoints.
    """

    cshogi = require_cshogi()

    if not cshogi.move_is_drop(move):
        to_sq = cshogi.move_to(move)
        from_sq = cshogi.move_from(move)

        if color == cshogi.WHITE:
            to_sq = 80 - to_sq
            from_sq = 80 - from_sq

        to_x, to_y = divmod(to_sq, 9)
        from_x, from_y = divmod(from_sq, 9)
        dir_x = to_x - from_x
        dir_y = to_y - from_y

        if dir_y < 0:
            if dir_x == 0:
                move_direction = UP
            elif dir_y == -2 and dir_x == -1:
                move_direction = UP2_RIGHT
            elif dir_y == -2 and dir_x == 1:
                move_direction = UP2_LEFT
            elif dir_x < 0:
                move_direction = UP_RIGHT
            else:
                move_direction = UP_LEFT
        elif dir_y == 0:
            if dir_x < 0:
                move_direction = RIGHT
            else:
                move_direction = LEFT
        else:
            if dir_x == 0:
                move_direction = DOWN
            elif dir_x < 0:
                move_direction = DOWN_RIGHT
            else:
                move_direction = DOWN_LEFT

        if cshogi.move_is_promotion(move):
            move_direction += 10
    else:
        to_sq = cshogi.move_to(move)
        if color == cshogi.WHITE:
            to_sq = 80 - to_sq
        move_direction = 20 + cshogi.move_drop_hand_piece(move)

    return move_direction * 81 + to_sq



def legal_move_labels(board: Any) -> np.ndarray:
    moves = list(board.legal_moves)
    labels = np.fromiter((make_move_label(move, board.turn) for move in moves), dtype=np.int64, count=len(moves))
    return labels



def legal_moves_and_labels(board: Any) -> tuple[list[int], np.ndarray]:
    moves = list(board.legal_moves)
    labels = np.fromiter((make_move_label(move, board.turn) for move in moves), dtype=np.int64, count=len(moves))
    return moves, labels
