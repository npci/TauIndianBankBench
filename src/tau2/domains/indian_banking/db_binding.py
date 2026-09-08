"""Bind the per-simulation database to the tool implementations in `service`.

The service module reads its customer database through `_load_db`. Each tau2
simulation owns its own database object, so `_load_db` is redirected to a
context variable that the toolkit sets for the duration of a call, and
persistence is disabled: state lives in the simulation, never on disk.
"""

from __future__ import annotations

import contextvars
from types import ModuleType
from typing import Any, Optional

from tau2.domains.indian_banking import service

_BOUND_DB: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "indian_banking_bound_db", default=None
)
_BOUND_CUSTOMER: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "indian_banking_bound_customer", default=None
)

_PATCHED = False


def _patch_module(mod: ModuleType) -> None:
    _orig_load_db = mod._load_db
    _orig_active_customer_id = mod._active_customer_id

    def _load_db() -> dict[str, Any]:
        bound = _BOUND_DB.get()
        if bound is not None:
            return bound
        return _orig_load_db()

    def _active_customer_id() -> Optional[str]:
        bound_cid = _BOUND_CUSTOMER.get()
        if bound_cid:
            db = _load_db()
            if bound_cid in db.get("customers", {}):
                return bound_cid
        return _orig_active_customer_id()

    def _persist_db() -> None:
        return None

    mod._load_db = _load_db  # type: ignore[attr-defined]
    mod._active_customer_id = _active_customer_id  # type: ignore[attr-defined]
    mod._persist_db = _persist_db  # type: ignore[attr-defined]


def get_service_module() -> ModuleType:
    """Return the tool implementation module, bound to the simulation database."""
    global _PATCHED
    if not _PATCHED:
        _patch_module(service)
        _PATCHED = True
    return service


def bind_db(raw_db: dict[str, Any], active_customer: Optional[str]) -> tuple[Any, Any]:
    token_db = _BOUND_DB.set(raw_db)
    token_cid = _BOUND_CUSTOMER.set(active_customer)
    return token_db, token_cid


def reset_db(token_db: Any, token_cid: Any) -> None:
    _BOUND_DB.reset(token_db)
    _BOUND_CUSTOMER.reset(token_cid)
