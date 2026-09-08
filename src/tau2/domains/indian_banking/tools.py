"""tau2 ToolKit over the banking service implementation (33 tools + transfer)."""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Optional, get_type_hints

from tau2.domains.indian_banking.data_model import IndianBankingDB
from tau2.domains.indian_banking.db_binding import (
    bind_db,
    get_service_module,
    reset_db,
)
from tau2.domains.indian_banking.utils import (
    INDIAN_BANKING_DB_PATH,
    INDIAN_BANKING_KB_DIR,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

_WRITE_TOOLS = {
    "create_fd",
    "create_rd",
    "close_deposit",
    "update_deposit_renewal",
    "toggle_card_freeze",
    "block_card",
    "set_card_controls",
    "cancel_mandate",
    "stop_cheque_payment",
    "update_address",
    "raise_request",
}

_SOFT_WRITE_TOOLS = {
    "request_cheque_book",
    "request_duplicate_statement",
}

TOOL_NAMES = [
    "search_knowledge_base",
    "get_account_balance",
    "get_account_details",
    "get_transaction_history",
    "get_fd_details",
    "get_rd_details",
    "get_deposit_loan_rates",
    "calculate_fd_maturity",
    "calculate_rd_maturity",
    "create_fd",
    "create_rd",
    "get_deposit_closure_quote",
    "close_deposit",
    "update_deposit_renewal",
    "get_loan_details",
    "get_loan_foreclosure_quote",
    "calculate_emi",
    "get_gold_rate",
    "calculate_gold_loan_ltv",
    "get_card_details",
    "toggle_card_freeze",
    "block_card",
    "set_card_controls",
    "show_mandates",
    "cancel_mandate",
    "stop_cheque_payment",
    "request_cheque_book",
    "request_duplicate_statement",
    "update_address",
    "raise_request",
    "get_request_status",
    "get_insurance_details",
    "get_products_and_offers",
]


def _tool_type(name: str) -> ToolType:
    if name in _WRITE_TOOLS or name in _SOFT_WRITE_TOOLS:
        return ToolType.WRITE
    return ToolType.READ


def _mutates_state(name: str) -> bool:
    return name in _WRITE_TOOLS


def _configure_service_env() -> None:
    """Point the service module at the packaged data and disable persistence."""
    os.environ.setdefault("BANK_CUSTOMER_DB", str(INDIAN_BANKING_DB_PATH))
    os.environ.setdefault("BANK_KB_DIR", str(INDIAN_BANKING_KB_DIR))
    os.environ["BANK_DB_PERSIST"] = "0"
    mod = get_service_module()
    mod.KB_DIR = os.environ["BANK_KB_DIR"]
    mod._DB_CACHE = None
    mod._KB_CACHE = None


def _resolved_params(func: Callable[..., str]) -> list[inspect.Parameter]:
    try:
        hints = get_type_hints(func, globalns=func.__globals__)
    except Exception:
        hints = {}
    sig = inspect.signature(func)
    params: list[inspect.Parameter] = []
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            continue
        anno = hints.get(p.name, Any)
        if anno is inspect.Parameter.empty:
            anno = Any
        params.append(
            inspect.Parameter(
                p.name,
                inspect.Parameter.KEYWORD_ONLY,
                default=p.default,
                annotation=anno,
            )
        )
    return params


def _make_tool_method(name: str, func: Callable[..., str]):
    params = _resolved_params(func)

    def method(self: "IndianBankingTools", *args: Any, **kwargs: Any) -> str:
        return self._call_service(name, func, *args, **kwargs)

    method.__name__ = name
    method.__doc__ = func.__doc__ or f"Banking tool: {name}"
    method.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=[
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            *params,
        ],
        return_annotation=str,
    )
    method.__annotations__ = {
        **{p.name: p.annotation for p in params},
        "return": str,
    }
    return is_tool(_tool_type(name), mutates_state=_mutates_state(name))(method)


def _transfer_to_human_agents(self, summary: str) -> str:
    """
    Transfer the user to a human agent, with a summary of the user's issue.
    Only transfer if the user explicitly asks for a human agent, or if given
    the policy and available tools you cannot solve the user's issue.

    Args:
        summary: A summary of the user's issue.

    Returns:
        A message indicating the user has been transferred to a human agent.
    """
    return "Transfer successful"


def _build_tools_class() -> type:
    _configure_service_env()
    service = get_service_module()
    tool_funcs = {name: getattr(service, name) for name in TOOL_NAMES}

    def __init__(
        self, db: IndianBankingDB, active_customer: Optional[str] = None
    ) -> None:
        ToolKitBase.__init__(self, db)
        if active_customer:
            self.db.active_customer = active_customer
        elif self.db.active_customer is None and len(self.db.customers) == 1:
            self.db.active_customer = next(iter(self.db.customers))
        os.environ["BANK_ACTIVE_CUSTOMER"] = self.db.active_customer or ""

    def _call_service(
        self, name: str, func: Callable[..., str], *args: Any, **kwargs: Any
    ) -> str:
        token_db, token_cid = bind_db(self.db.as_raw(), self.db.active_customer)
        try:
            return func(*args, **kwargs)
        finally:
            reset_db(token_db, token_cid)

    def update_db(self, update_data: Optional[dict[str, Any]] = None) -> None:
        ToolKitBase.update_db(self, update_data)
        os.environ["BANK_ACTIVE_CUSTOMER"] = self.db.active_customer or ""

    def close(self) -> None:
        return None

    attrs: dict[str, Any] = {
        "__doc__": "All the tools for the Indian banking domain.",
        "db": IndianBankingDB,
        "__init__": __init__,
        "_call_service": _call_service,
        "update_db": update_db,
        "close": close,
        "transfer_to_human_agents": is_tool(ToolType.GENERIC)(
            _transfer_to_human_agents
        ),
    }
    for name, func in tool_funcs.items():
        attrs[name] = _make_tool_method(name, func)

    return type("IndianBankingTools", (ToolKitBase,), attrs)


IndianBankingTools = _build_tools_class()
