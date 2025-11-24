from typing import Dict

from ..utils.errors import wrap_success


def announce_step(agent_name: str, reason: str) -> Dict[str, str]:
    """Emit a concise message about an upcoming delegation.

    Returns a normalized success envelope.
    """
    msg = f"Delegating to {agent_name} — {reason}"
    return wrap_success({"message": msg, "agent": agent_name})
