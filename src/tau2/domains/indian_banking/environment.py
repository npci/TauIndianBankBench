import os
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.indian_banking.data_model import IndianBankingDB
from tau2.domains.indian_banking.tools import IndianBankingTools
from tau2.domains.indian_banking.utils import (
    INDIAN_BANKING_DB_PATH,
    INDIAN_BANKING_POLICY_PATH,
    INDIAN_BANKING_TASKS_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file

DOMAIN_NAME = "indian_banking"


def get_environment(
    db: Optional[IndianBankingDB] = None,
    solo_mode: bool = False,
    active_customer: Optional[str] = None,
) -> Environment:
    if solo_mode:
        raise ValueError("The indian_banking domain does not support solo mode")
    # Reproducibility: the database check builds a gold database by replaying the
    # expected actions in a separate process, so wall-clock timestamps and random
    # ids would differ from the agent run even for identical behaviour. Freeze the
    # simulation clock (the date the policy states) and seed id generation, so the
    # environment is a deterministic function of (initial state, actions).
    os.environ.setdefault("BANK_SIM_CLOCK", "2026-07-16")
    os.environ.setdefault("BANK_SIM_SEED", "tau2-banking")
    if db is None:
        db = IndianBankingDB.load(str(INDIAN_BANKING_DB_PATH))
    tools = IndianBankingTools(db, active_customer=active_customer)
    with open(INDIAN_BANKING_POLICY_PATH, "r", encoding="utf-8") as fp:
        policy = fp.read()
    return Environment(
        domain_name=DOMAIN_NAME,
        policy=policy,
        tools=tools,
    )


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """Load the task suite, optionally restricted to one split."""
    tasks = [Task.model_validate(task) for task in load_file(INDIAN_BANKING_TASKS_PATH)]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. "
            f"Valid splits are: {list(task_splits.keys())}"
        )
    split_ids = set(task_splits[task_split_name])
    return [task for task in tasks if task.id in split_ids]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        INDIAN_BANKING_TASKS_PATH.parent / f"split_{INDIAN_BANKING_TASKS_PATH.stem}.json"
    )
    return load_file(split_file)
