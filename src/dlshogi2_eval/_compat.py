from __future__ import annotations

from typing import Any

try:
    import cshogi as _cshogi  # type: ignore
except ImportError:  # pragma: no cover - exercised only when dependency missing
    _cshogi = None

CSHOGI_AVAILABLE = _cshogi is not None


def require_cshogi() -> Any:
    if _cshogi is None:
        raise ImportError(
            "cshogi is required for board parsing and feature generation. "
            "Install it with `pip install cshogi`."
        )
    return _cshogi
