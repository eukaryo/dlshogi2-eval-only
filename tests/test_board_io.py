from __future__ import annotations

import pytest

from dlshogi2_eval._compat import CSHOGI_AVAILABLE
from dlshogi2_eval.board_io import load_board

pytestmark = pytest.mark.skipif(not CSHOGI_AVAILABLE, reason="cshogi is not installed")



def test_load_position_startpos() -> None:
    board = load_board(position="position startpos moves 7g7f 3c3d")
    assert board.move_number == 3
