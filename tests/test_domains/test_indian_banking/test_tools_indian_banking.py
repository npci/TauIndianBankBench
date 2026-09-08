"""Smoke tests for the indian_banking domain tools."""

from tau2.data_model.message import ToolCall
from tau2.domains.indian_banking.environment import (
    get_environment,
    get_tasks,
    get_tasks_split,
)

CUSTOMER = "CUST_35359B3E"
ACCOUNT = "SB1785318658"
CARD = "CARD77485"


def _environment():
    return get_environment(active_customer=CUSTOMER)


def test_get_environment_and_tools():
    env = _environment()
    assert env.domain_name == "indian_banking"
    tool_names = {t.name for t in env.get_tools()}
    assert "get_account_balance" in tool_names
    assert "create_fd" in tool_names
    assert "search_knowledge_base" in tool_names
    assert "_transfer_to_human_agents" in tool_names
    assert len(tool_names) == 34


def test_get_account_balance():
    env = _environment()
    response = env.get_response(
        ToolCall(
            id="1",
            name="get_account_balance",
            arguments={"account_ids": [ACCOUNT]},
        )
    )
    assert ACCOUNT in response.content
    assert "2500000" in response.content


def test_unlinked_account_is_refused():
    env = _environment()
    response = env.get_response(
        ToolCall(
            id="2",
            name="get_account_balance",
            arguments={"account_ids": ["SB0000000000"]},
        )
    )
    assert "not linked" in response.content.lower()


def test_toggle_card_freeze_mutates_db():
    env = _environment()
    before = env.get_db_hash()
    response = env.get_response(
        ToolCall(
            id="3",
            name="toggle_card_freeze",
            arguments={"card_id": CARD, "state": "freeze"},
        )
    )
    assert "freeze" in response.content.lower() or "frozen" in response.content.lower()
    assert before != env.get_db_hash()
    card = env.tools.db.customers[CUSTOMER]["cards"][CARD]
    assert str(card.get("status", "")).lower() == "frozen"


def test_read_tool_does_not_mutate_db():
    env = _environment()
    before = env.get_db_hash()
    env.get_response(
        ToolCall(id="4", name="get_card_details", arguments={"card_ids": [CARD]})
    )
    assert before == env.get_db_hash()


def test_tools_are_deterministic_across_environments():
    """Two environments running the same write must agree on the database."""
    call = ToolCall(
        id="5",
        name="raise_request",
        arguments={"category": "service_quality", "description": "Branch queue"},
    )
    hashes = []
    for env in (_environment(), _environment()):
        env.get_response(call)
        hashes.append(env.get_db_hash())
    assert hashes[0] == hashes[1]


def test_tasks_load():
    tasks = get_tasks("base")
    assert len(tasks) == 1000
    assert all(task.user_scenario is not None for task in tasks)
    assert all(
        task.user_scenario.instructions.domain == "banking" for task in tasks
    )


def test_task_splits():
    splits = get_tasks_split()
    assert set(splits) == {"base", "lite", "train", "test"}
    base = set(splits["base"])
    assert len(base) == 1000
    for name in ("lite", "train", "test"):
        assert set(splits[name]) <= base
        assert len(get_tasks(name)) == len(splits[name])
