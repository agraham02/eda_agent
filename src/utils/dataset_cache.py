"""Lightweight per-process dataset access cache.

The existing `get_dataset` in `data_store` already returns in-memory
DataFrames. This helper adds an optional shallow caching layer for
frequently accessed dataset_ids within hot paths (e.g., multiple tools
in one orchestration step) and exposes cache stats + clear.

Use `get_dataset_cached(dataset_id)` instead of `get_dataset` when
repeated access in same turn; avoids dict lookup overhead & enables
future hooks (e.g., read/write tracking, instrumentation).
"""

from typing import Any, Dict

import pandas as pd

from .data_store import get_dataset

_CACHE: Dict[str, pd.DataFrame] = {}
_HITS: int = 0
_MISSES: int = 0


def get_dataset_cached(dataset_id: str) -> pd.DataFrame:
    global _HITS, _MISSES
    if dataset_id in _CACHE:
        _HITS += 1
        return _CACHE[dataset_id]
    _MISSES += 1
    df = get_dataset(dataset_id)
    _CACHE[dataset_id] = df
    return df


def cache_stats() -> Dict[str, Any]:
    return {
        "entries": len(_CACHE),
        "hits": _HITS,
        "misses": _MISSES,
        "hit_ratio": (_HITS / (_HITS + _MISSES)) if (_HITS + _MISSES) else None,
    }


def clear_cache() -> None:
    _CACHE.clear()
    global _HITS, _MISSES
    _HITS = 0
    _MISSES = 0
