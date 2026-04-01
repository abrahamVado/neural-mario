"""Action labels for Mario policies and viewers."""
from __future__ import annotations

ACTION_NAMES: tuple[str, ...] = (
    "noop",
    "right",
    "right_jump",
    "right_run",
    "right_run_jump",
    "jump",
    "left",
)


def action_name(action: int) -> str:
    """Return a stable label for an action index."""
    if 0 <= action < len(ACTION_NAMES):
        return ACTION_NAMES[action]
    return f"unknown_{action}"
