"""Normalize environment state for equality checks (response replay and DB hash).

Tool responses and persisted records carry randomly generated references that
carry no meaning for correctness. Strip those fields, and bag-normalize maps
keyed only by generated ids, before comparing.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Field names that must not gate equality.
#
# Determinism is enforced at the source: the domain runs on a frozen simulation
# clock (``BANK_SIM_CLOCK``), so every timestamp is identical between the agent
# run and the gold replay, and ``create_fd`` / ``create_rd`` derive their
# persisted deposit id from a content hash (``BANK_SIM_SEED``). Timestamps and
# deposit ids therefore need no stripping.
#
# What is listed here is the irreducible minimum — values that still differ
# between two correct runs and carry no meaning for database correctness:
VOLATILE_STATE_KEYS = frozenset(
    {
        # Random server refs still generated per-call (not content-hashed).
        # These sit inside bag-normalized dicts or list records where the id is
        # not the semantic content (the mutation itself is checked elsewhere).
        "request_id",  # raise_request SR-…, cheque/statement/address CHQ…/STMT…/ADR…
        "reference_id",  # transaction rows appended by debits (TXN…)
        "reference",  # stop_cheque STP… refs
        # Free-text agent wording on raise_request / tickets — gold vs agent
        # descriptions never match byte-for-byte even when the SR is correct.
        "description",
        "note",
        # Response-only fields (not persisted to the DB, so no false-pass risk);
        # kept so tool-response replay equality ignores them.
        "receipt_ref",
        "download_url",
        "expiry",
    }
)

# Optional fields on service-request records that agents often fill differently
# from gold (e.g. account_id present vs omitted). Semantic match is category +
# related_transaction_id + status/priority.
_REQUEST_OPTIONAL_KEYS = frozenset({"account_id"})

# Replacement / general SRs: agents freely pick other vs general_query.
_CATEGORY_ALIASES = {
    "other": "general_query",
    "general_query": "general_query",
}

# Dict keys that are only server-assigned refs (e.g. requests["SR-12345678"]).
# When every key in a map matches, compare as an order-independent bag of values.
_EPHEMERAL_DICT_KEY = re.compile(
    r"^(?:SR-|TXN|ADR|CHQ|STMT|RCPT|STP|FD|RD)\d+$",
    re.IGNORECASE,
)


def _stable_sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _looks_like_request_record(d: dict) -> bool:
    """Heuristic: raise_request / ticket-shaped dicts."""
    return "category" in d and (
        "request_id" in d or "related_transaction_id" in d or "priority" in d
    )


def normalize_state(value: Any) -> Any:
    """Return a copy of ``value`` with ephemeral ids/timestamps removed."""
    if isinstance(value, dict):
        if value and all(
            isinstance(k, str) and _EPHEMERAL_DICT_KEY.match(k) for k in value
        ):
            items = [normalize_state(v) for v in value.values()]
            return sorted(items, key=_stable_sort_key)
        drop = VOLATILE_STATE_KEYS
        if _looks_like_request_record(value):
            drop = VOLATILE_STATE_KEYS | _REQUEST_OPTIONAL_KEYS
        out = {
            k: normalize_state(v)
            for k, v in value.items()
            if k not in drop
        }
        if _looks_like_request_record(value) and "category" in out:
            cat = out["category"]
            if isinstance(cat, str) and cat in _CATEGORY_ALIASES:
                out["category"] = _CATEGORY_ALIASES[cat]
                # Priority follows category; keep soft SRs comparable.
                if out.get("priority") in ("NORMAL", "URGENT", None) and cat in (
                    "other",
                    "general_query",
                ):
                    out["priority"] = "NORMAL"
        return out
    if isinstance(value, list):
        return [normalize_state(item) for item in value]
    return value
