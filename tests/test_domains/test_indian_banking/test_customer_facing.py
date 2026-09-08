from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import CommunicateCheck
from tau2.data_model.tasks import (
    EvaluationCriteria,
    StructuredUserInstructions,
    Task,
    UserScenario,
)
from tau2.domains.indian_banking.customer_facing import (
    evaluate_banking_customer_facing,
    find_internal_leaks,
    is_banking_task,
)
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator


def _domain_task() -> Task:
    return Task(
        id="test_leak",
        description={"purpose": "test"},
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="banking",
                reason_for_call="test",
                task_instructions="test",
            )
        ),
    )


def test_find_internal_leaks_tool_and_category():
    text = (
        "To proceed, I need information for the `raise_request` tool. "
        "Category: **unauthorized_debit**."
    )
    leaks = find_internal_leaks(text)
    assert "raise_request" in leaks
    assert "unauthorized_debit" in leaks


def test_find_internal_leaks_allows_customer_facing_ids():
    text = "Your request ID is SR-27213418 and card CARD40024 is frozen."
    assert find_internal_leaks(text) == []


def test_evaluate_banking_customer_facing_fails_trial1_style_leak():
    trajectory = [
        AssistantMessage(
            role="assistant",
            content=(
                "I need a little more information for the raise_request tool. "
                "Category: unauthorized_debit."
            ),
            turn_idx=4,
        ),
    ]
    check = evaluate_banking_customer_facing(trajectory)
    assert isinstance(check, CommunicateCheck)
    assert check.met is False
    assert check.info == "no_internal_tool_or_api_leaks"


def test_communicate_evaluator_applies_leak_check():
    task = _domain_task()
    task.evaluation_criteria = EvaluationCriteria(
        communicate_info=["SR-"],
        actions=[],
        reward_basis=["COMMUNICATE"],
    )
    trajectory = [
        AssistantMessage(
            role="assistant",
            content="Use raise_request with category unauthorized_debit.",
            turn_idx=1,
        ),
    ]
    reward = CommunicateEvaluator.calculate_reward(task, trajectory)
    assert reward.reward == 0.0
    assert any(
        c.info == "no_internal_tool_or_api_leaks" and not c.met
        for c in (reward.communicate_checks or [])
    )


def test_communicate_evaluator_skips_leak_check_for_other_domains():
    task = Task(
        id="other",
        description={"purpose": "test"},
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="other_domain",
                reason_for_call="test",
                task_instructions="test",
            )
        ),
        evaluation_criteria=EvaluationCriteria(communicate_info=[], actions=[]),
    )
    trajectory = [
        AssistantMessage(
            role="assistant",
            content="calling raise_request with unauthorized_debit",
            turn_idx=1,
        ),
    ]
    reward = CommunicateEvaluator.calculate_reward(task, trajectory)
    assert reward.reward == 1.0
    assert reward.communicate_checks is None or all(
        c.info != "no_internal_tool_or_api_leaks" for c in reward.communicate_checks
    )


def test_is_domain_task():
    assert is_banking_task(_domain_task()) is True
