"""Shared fixtures and builders for the test suite."""

import pytest

from tau2.data_model.simulation import (
    Info,
    Results,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo
from tau2.registry import registry
from tau2.run import get_tasks

DOMAIN = "indian_banking"


@pytest.fixture
def domain_name() -> str:
    return DOMAIN


@pytest.fixture
def environment_constructor():
    return registry.get_env_constructor(DOMAIN)


@pytest.fixture
def base_task() -> Task:
    """An arbitrary, stable task from the domain."""
    return get_tasks(DOMAIN)[0]


# ---- Builders for synthetic Results, shared by the storage/checkpoint tests ----


def make_info() -> Info:
    return Info(
        git_commit="abc123",
        num_trials=1,
        max_steps=100,
        max_errors=10,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name=DOMAIN, policy="test policy"),
    )


def make_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(instructions="test instruction"),
        evaluation_criteria=EvaluationCriteria(),
    )


def make_sim(
    task_id: str,
    trial: int = 0,
    seed: int = 42,
    termination_reason: TerminationReason = TerminationReason.USER_STOP,
) -> SimulationRun:
    return SimulationRun(
        id=f"sim-{task_id}-t{trial}-s{seed}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=termination_reason,
        messages=[],
        trial=trial,
        seed=seed,
    )


@pytest.fixture
def sample_results() -> Results:
    return Results(
        info=make_info(),
        tasks=[make_task("t0"), make_task("t1")],
        simulations=[
            make_sim("t0", trial=0, seed=42),
            make_sim("t1", trial=0, seed=42),
        ],
    )
