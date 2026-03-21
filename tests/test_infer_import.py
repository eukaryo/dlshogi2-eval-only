from __future__ import annotations

from dlshogi2_eval import __version__



def test_package_import() -> None:
    assert isinstance(__version__, str)
