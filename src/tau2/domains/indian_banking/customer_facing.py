"""Detect internal tool/API leaks in the agent's customer-facing messages."""

from __future__ import annotations

import re
from typing import Iterable

from tau2.data_model.message import AssistantMessage, Message
from tau2.data_model.simulation import CommunicateCheck
from tau2.data_model.tasks import StructuredUserInstructions, Task
from tau2.domains.indian_banking.tools import TOOL_NAMES

# Service-request categories and other backend enums customers should not see.
_INTERNAL_CATEGORY_CODES = frozenset(
    {
        "failed_transaction",
        "unauthorized_debit",
        "service_quality",
        "app_issue",
        "charges_dispute",
        "general_query",
    }
)

_INTERNAL_API_FIELD_NAMES = frozenset(
    {
        "related_transaction_id",
        "compare_args",
    }
)

# Tool names as registered; the transfer tool's exposed alias is not included.
_BANKING_TOOL_NAMES = frozenset(TOOL_NAMES) | {"transfer_to_human_agents"}


def is_banking_task(task: Task) -> bool:
    if task.user_scenario is None:
        return False
    instructions = task.user_scenario.instructions
    return (
        isinstance(instructions, StructuredUserInstructions)
        and instructions.domain == "banking"
    )


def _assistant_text_messages(
    full_trajectory: Iterable[Message],
) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for message in full_trajectory:
        if not isinstance(message, AssistantMessage):
            continue
        if not message.has_text_content():
            continue
        content = message.content or ""
        if content.strip():
            out.append((message.turn_idx, content))
    return out


def find_internal_leaks(text: str) -> list[str]:
    """Return sorted unique internal identifiers found in *text*."""
    lower = text.lower()
    found: set[str] = set()

    for tool in _BANKING_TOOL_NAMES:
        if re.search(rf"\b{re.escape(tool)}\b", lower):
            found.add(tool)

    for code in _INTERNAL_CATEGORY_CODES:
        if re.search(rf"\b{re.escape(code)}\b", lower):
            found.add(code)

    for field in _INTERNAL_API_FIELD_NAMES:
        if re.search(rf"\b{re.escape(field)}\b", lower):
            found.add(field)

    return sorted(found)


def evaluate_banking_customer_facing(
    full_trajectory: list[Message],
) -> CommunicateCheck:
    """Fail if any agent message exposes internal tool/API identifiers."""
    violations: list[str] = []
    for turn_idx, content in _assistant_text_messages(full_trajectory):
        leaks = find_internal_leaks(content)
        if leaks:
            violations.append(f"turn {turn_idx}: {', '.join(leaks)}")

    if not violations:
        return CommunicateCheck(
            info="no_internal_tool_or_api_leaks",
            met=True,
            justification="Agent messages did not expose tool names, category codes, or API field names.",
        )

    return CommunicateCheck(
        info="no_internal_tool_or_api_leaks",
        met=False,
        justification=(
            "Agent exposed internal tool/API identifiers: "
            + "; ".join(violations)
        ),
    )
