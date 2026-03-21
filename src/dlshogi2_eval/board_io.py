from __future__ import annotations

from typing import Any, Iterable

from ._compat import require_cshogi



def load_board(*, position: str | None = None, sfen: str | None = None) -> Any:
    """Create and initialize a `cshogi.Board`.

    Parameters
    ----------
    position:
        Either `startpos moves ...`, `sfen ...`, or the full USI command
        `position ...`.
    sfen:
        Bare SFEN string or `sfen ...`.
    """

    if (position is None) == (sfen is None):
        raise ValueError("exactly one of `position` or `sfen` must be provided")

    cshogi = require_cshogi()
    board = cshogi.Board()

    if position is not None:
        pos = position.strip()
        if pos.startswith("position "):
            pos = pos[len("position ") :]
        board.set_position(pos)
        return board

    assert sfen is not None
    sfen_text = sfen.strip()
    if sfen_text.startswith("sfen "):
        sfen_text = sfen_text[len("sfen ") :]
    board.set_sfen(sfen_text)
    return board



def legal_moves_usi(board: Any) -> list[str]:
    cshogi = require_cshogi()
    return [cshogi.move_to_usi(move) for move in board.legal_moves]
