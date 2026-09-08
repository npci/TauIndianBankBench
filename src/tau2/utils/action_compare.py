"""Soft argument comparison for ACTION evaluation.

Ignores ephemeral / free-text fields so wrong *semantic* tool args still fail
while paraphrases, timestamps, and generated ids do not.
"""

from __future__ import annotations

from typing import Any, Optional

# Keys that must not gate ACTION matching (ephemeral / free-text only).
# Do NOT include product targets like deposit_id, card_id, mandate_id.
VOLATILE_ARG_KEYS = frozenset(
    {
        "updated_at",
        "cancelled_at",
        "blocked_at",
        "booked_at",
        "closed_at",
        "logged_at",
        "requested_at",
        "created_at",
        "timestamp",
        "receipt_ref",
        "request_id",
        "reference_id",
        "reference",
        "download_url",
        "expiry",
        "eta",
        "description",
        "note",
    }
)

# If either side omits/None, do not require equality.
OPTIONAL_ARG_KEYS = frozenset({"account_id"})

_MISSING = object()


def args_match(
    gold_args: dict[str, Any],
    pred_args: dict[str, Any],
    compare_args: Optional[list[str]] = None,
) -> bool:
    """Return True if predicted args match gold under soft rules.

    - ``compare_args is None``: compare gold's keys (minus volatile).
    - ``compare_args == []``: always True (name-only match at caller).
    - else: compare listed keys (minus volatile).
    """
    if compare_args is not None and len(compare_args) == 0:
        return True

    if compare_args is None:
        keys = [k for k in gold_args.keys() if k not in VOLATILE_ARG_KEYS]
    else:
        keys = [k for k in compare_args if k not in VOLATILE_ARG_KEYS]

    for key in keys:
        gold_v = gold_args.get(key, _MISSING)
        pred_v = pred_args.get(key, _MISSING)
        if key in OPTIONAL_ARG_KEYS:
            if gold_v in (None, _MISSING) or pred_v in (None, _MISSING):
                continue
        if gold_v is _MISSING:
            # Gold did not specify this compare key — only require pred if
            # compare_args explicitly listed it; treat missing gold as no constraint.
            continue
        if pred_v is _MISSING or gold_v != pred_v:
            return False
    return True
