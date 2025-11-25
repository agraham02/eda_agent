"""Shared lightweight parsing utilities."""

from typing import List, Optional


def parse_columns_csv(columns_csv: str) -> Optional[List[str]]:
    """Parse a comma-separated column list; return None if empty.

    Normalizes whitespace and ignores empty tokens.
    """
    cols = [c.strip() for c in columns_csv.split(",") if c.strip()]
    return cols if cols else None
