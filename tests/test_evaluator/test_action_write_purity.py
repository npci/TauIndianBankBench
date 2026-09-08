"""Tests for soft ACTION arg compare and write-purity checks."""

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import Action
from tau2.environment.toolkit import ToolType
from tau2.evaluator.evaluator_action import _check_actions
from tau2.utils.action_compare import args_match


def test_args_match_ignores_description_and_optional_account():
    gold = {
        "category": "unauthorized_debit",
        "related_transaction_id": "TXN4739654514",
        "account_id": "SB7118875992",
        "description": "gold wording",
    }
    pred = {
        "category": "unauthorized_debit",
        "related_transaction_id": "TXN4739654514",
        "description": "different agent wording",
    }
    assert args_match(gold, pred, compare_args=["category", "account_id", "related_transaction_id"])
    assert args_match(gold, pred, compare_args=None)


def test_args_match_fails_on_wrong_related_transaction():
    gold = {
        "category": "unauthorized_debit",
        "related_transaction_id": "TXN4739654514",
    }
    pred = {
        "category": "unauthorized_debit",
        "related_transaction_id": "TXN2407174670",
    }
    assert not args_match(gold, pred, compare_args=["category", "related_transaction_id"])


def test_write_purity_fails_wrong_then_correct_raise_request():
    gold = [
        Action(
            action_id="sr",
            name="raise_request",
            arguments={
                "category": "unauthorized_debit",
                "related_transaction_id": "TXN4739654514",
                "account_id": "SB7118875992",
                "description": "Customer disputes suspicious debit TXN4739654514",
            },
            compare_args=["category", "account_id", "related_transaction_id"],
        ),
        Action(
            action_id="fz",
            name="toggle_card_freeze",
            arguments={"card_id": "CARD67950", "state": "freeze"},
        ),
    ]
    predicted = [
        ToolCall(
            id="1",
            name="raise_request",
            arguments={
                "category": "unauthorized_debit",
                "related_transaction_id": "TXN2407174670",
                "description": "wrong txn first",
            },
        ),
        ToolCall(
            id="2",
            name="raise_request",
            arguments={
                "category": "unauthorized_debit",
                "related_transaction_id": "TXN4739654514",
                "description": "corrected",
            },
        ),
        ToolCall(
            id="3",
            name="toggle_card_freeze",
            arguments={"card_id": "CARD67950", "state": "freeze"},
        ),
    ]
    tool_types = {
        "raise_request": ToolType.WRITE,
        "toggle_card_freeze": ToolType.WRITE,
        "get_transaction_history": ToolType.READ,
    }
    checks = _check_actions(predicted, gold, tool_types)
    assert any(c.action_match for c in checks if c.action.action_id == "sr")
    assert any(c.action_match for c in checks if c.action.action_id == "fz")
    unexpected = [c for c in checks if c.action.action_id.startswith("unexpected_write_")]
    assert unexpected and all(not c.action_match for c in unexpected)
    assert not all(c.action_match for c in checks)


def test_write_purity_passes_clean_path_with_paraphrase():
    gold = [
        Action(
            action_id="sr",
            name="raise_request",
            arguments={
                "category": "unauthorized_debit",
                "related_transaction_id": "TXN4739654514",
                "description": "gold",
            },
            compare_args=["category", "related_transaction_id"],
        ),
    ]
    predicted = [
        ToolCall(
            id="1",
            name="raise_request",
            arguments={
                "category": "unauthorized_debit",
                "related_transaction_id": "TXN4739654514",
                "description": "paraphrase ok",
            },
        ),
    ]
    tool_types = {"raise_request": ToolType.WRITE}
    checks = _check_actions(predicted, gold, tool_types)
    assert all(c.action_match for c in checks)


def test_abort_task_any_write_fails_purity():
    """Gold has only reads; any WRITE in trajectory fails."""
    gold = [
        Action(
            action_id="q",
            name="get_deposit_closure_quote",
            arguments={"deposit_ids": ["FD468014"]},
        ),
    ]
    predicted = [
        ToolCall(
            id="1",
            name="get_deposit_closure_quote",
            arguments={"deposit_ids": ["FD468014"]},
        ),
        ToolCall(
            id="2",
            name="close_deposit",
            arguments={"deposit_id": "FD468014", "payout_account": "SB7118875992"},
        ),
    ]
    tool_types = {
        "get_deposit_closure_quote": ToolType.READ,
        "close_deposit": ToolType.WRITE,
    }
    checks = _check_actions(predicted, gold, tool_types)
    assert any(c.action.action_id.startswith("unexpected_write_") for c in checks)
    assert not all(c.action_match for c in checks)
