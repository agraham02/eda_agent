"""Unified error/response helpers for tools.

All tools should prefer returning a consistent envelope:
Success: {"ok": true, ...original payload...}
Error:   {"ok": false, "error": {"type": str, "message": str, "hint": Optional[str], "context": Optional[dict]}}

Tools keep their original top-level fields so existing agent instructions
continue to function (backward compatibility). Agents may optionally
inspect the "ok" flag.
"""

import traceback
from typing import Any, Dict, Optional


def make_error(
    error_type: str,
    message: str,
    hint: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "ok": False,
        "error": {
            "type": error_type,
            "message": message,
            "hint": hint,
            "context": context or {},
        },
    }


def wrap_success(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "ok" not in payload:  # preserve explicit ok if caller set it
        payload["ok"] = True
    return payload


def exception_to_error(
    error_type: str, exc: Exception, hint: Optional[str] = None
) -> Dict[str, Any]:
    return make_error(
        error_type=error_type,
        message=str(exc),
        hint=hint,
        context={"trace": traceback.format_exc(limit=3)},
    )
