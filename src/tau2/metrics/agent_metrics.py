import math
import re

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from tau2.data_model.simulation import Results, TerminationReason


def is_successful(reward: float) -> bool:
    """
    Check if the reward is successful.
    """
    return (1 - 1e-6) <= reward <= (1 + 1e-6)


def sum_message_tokens(sim) -> dict[str, int]:
    """Sum prompt/completion tokens from message.usage, split by agent vs user."""
    agent_prompt = agent_completion = 0
    user_prompt = user_completion = 0
    for msg in sim.messages or []:
        usage = getattr(msg, "usage", None) or {}
        if not usage:
            continue
        prompt = int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or 0
        )
        completion = int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        )
        role = getattr(msg, "role", None)
        if role == "assistant":
            agent_prompt += prompt
            agent_completion += completion
        elif role == "user":
            user_prompt += prompt
            user_completion += completion
    return {
        "agent_prompt_tokens": agent_prompt,
        "agent_completion_tokens": agent_completion,
        "agent_total_tokens": agent_prompt + agent_completion,
        "user_prompt_tokens": user_prompt,
        "user_completion_tokens": user_completion,
        "user_total_tokens": user_prompt + user_completion,
        "total_tokens": agent_prompt
        + agent_completion
        + user_prompt
        + user_completion,
    }


def count_turns(sim) -> dict[str, int]:
    """Count conversation turns from simulation messages.

    - agent_turns: assistant messages (text or tool-call turns)
    - user_turns: user messages
    - num_messages: all messages including tool responses (orchestrator steps)
    """
    msgs = []
    get_messages = getattr(sim, "get_messages", None)
    if callable(get_messages):
        msgs = get_messages() or []
    else:
        msgs = getattr(sim, "messages", None) or []
    agent_turns = user_turns = 0
    for msg in msgs:
        role = getattr(msg, "role", None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role")
        if role == "assistant":
            agent_turns += 1
        elif role == "user":
            user_turns += 1
    return {
        "agent_turns": agent_turns,
        "user_turns": user_turns,
        "num_messages": len(msgs),
    }


class AgentMetrics(BaseModel):
    # Core metrics
    avg_reward: float
    pass_hat_ks: dict[int, float]
    avg_agent_cost: float

    # Efficiency metrics (per conversation / simulation)
    avg_duration: float = 0.0
    avg_agent_turns: float = 0.0
    avg_user_turns: float = 0.0
    avg_num_messages: float = 0.0
    avg_agent_prompt_tokens: float = 0.0
    avg_agent_completion_tokens: float = 0.0
    avg_agent_total_tokens: float = 0.0
    avg_total_tokens: float = 0.0

    # Simulation counts
    total_simulations: int = 0
    total_tasks: int = 0
    infra_error_count: int = 0

    # Action metrics
    total_read_actions: int = 0
    correct_read_actions: int = 0
    total_write_actions: int = 0
    correct_write_actions: int = 0

    # DB match metrics
    db_match_count: int = 0
    db_mismatch_count: int = 0
    db_not_checked: int = 0

    # Termination reason counts
    termination_user_stop: int = 0
    termination_agent_stop: int = 0
    termination_max_steps: int = 0
    termination_error: int = 0
    termination_infrastructure_error: int = 0

    def as_dict(self) -> dict:
        data = {
            "avg_reward": self.avg_reward,
            "avg_agent_cost": self.avg_agent_cost,
            "avg_duration": self.avg_duration,
            "avg_agent_turns": self.avg_agent_turns,
            "avg_user_turns": self.avg_user_turns,
            "avg_num_messages": self.avg_num_messages,
            "avg_agent_prompt_tokens": self.avg_agent_prompt_tokens,
            "avg_agent_completion_tokens": self.avg_agent_completion_tokens,
            "avg_agent_total_tokens": self.avg_agent_total_tokens,
            "avg_total_tokens": self.avg_total_tokens,
            "total_simulations": self.total_simulations,
            "total_tasks": self.total_tasks,
            "infra_error_count": self.infra_error_count,
        }
        for k, v in self.pass_hat_ks.items():
            data[f"pass_hat_{k}"] = v
        return data


def pass_hat_k(num_trials: int, success_count: int, k: int) -> float:
    """
    Compute the pass^k metric for the given number of trials, success count, and k.
    from https://arxiv.org/pdf/2406.12045
    Args:
        num_trials: The number of trials.
        success_count: The number of successful trials.
        k: The number of trials to consider.
    Returns:
        The pass^k metric.
    """
    if num_trials < k:
        raise ValueError(f"Number of trials {num_trials} is less than k {k}.")
    return math.comb(success_count, k) / math.comb(num_trials, k)


def get_metrics_df(results: Results) -> tuple[pd.DataFrame, int]:
    """
    Convert the results to a dataframe and add a column for success.
    Filters out infrastructure errors (simulations that never ran).
    Checks that all simulations have the same number of trials.
    Returns the maximum number of trials that can be used for pass^k metrics.
    """
    df = results.to_df()

    infra_count = (
        df.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    ).sum()
    if infra_count > 0:
        logger.warning(
            f"Excluding {infra_count} infrastructure error simulation(s) from metrics."
        )
        df = df[df.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR]

    if df.empty:
        df["success"] = pd.Series(dtype=bool)
        return df, 0

    df["success"] = df.reward.apply(is_successful)
    if len(df.info_num_trials.unique()) > 1:
        logger.warning(
            f"All simulations must have the same number of trials. Found {df.info_num_trials.unique()}"
        )
    max_k = df.info_num_trials.max()

    task_ids_counts = [(tid, count) for tid, count in df.task_id.value_counts().items()]
    task_ids_counts.sort(key=lambda x: x[1])
    min_k = task_ids_counts[0][1]
    if min_k < max_k:
        logger.warning(
            f"The minimum number of trials for a task is {min_k}, which is less than the expected number of trials {max_k}. Setting max k to {min_k}."
        )
        max_k = min_k
    return df, max_k


def get_tasks_pass_hat_k(results: Results) -> pd.DataFrame:
    """
    Compute the pass^k for each k from 1 to the maximum number of trials.
    """
    df, max_k = get_metrics_df(results)
    if df.empty or max_k == 0:
        return pd.DataFrame()
    dfs = []
    for k in range(1, max_k + 1):
        res = df.groupby("task_id")["success"].apply(
            lambda df: pass_hat_k(len(df), df.sum(), k)
        )
        res.name = f"pass^{k}"
        dfs.append(res)
    df_pass_hat_k = pd.concat(dfs, axis=1)
    task_columns = [
        "task_num_agent_actions",
        "task_num_user_actions",
        "task_num_actions",
    ]
    df_task_infos = df.groupby("task_id").first()[task_columns]
    df_pass_hat_k = df_task_infos.join(df_pass_hat_k)
    return df_pass_hat_k


def prepare_dfs(results: Results) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, max_k = get_metrics_df(results)
    df_pass_hat_k = get_tasks_pass_hat_k(results)
    df_pass_hat_k["num_actions"] = df.groupby("task_id").first()["task_num_actions"]
    df_pass_hat_k = df_pass_hat_k.sort_values(by="num_actions")
    return df, df_pass_hat_k


def compute_metrics(results: Results) -> AgentMetrics:
    """
    Compute comprehensive metrics for the agent including:
    - average reward and pass^k
    - action metrics (read/write)
    - DB match and termination stats
    """
    if not results.simulations:
        return AgentMetrics(
            avg_reward=0.0,
            pass_hat_ks={},
            avg_agent_cost=0.0,
        )

    infra_error_count = sum(
        1
        for sim in results.simulations
        if sim.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR
    )
    evaluated_sims = [
        sim
        for sim in results.simulations
        if sim.termination_reason != TerminationReason.INFRASTRUCTURE_ERROR
    ]

    if not evaluated_sims:
        return AgentMetrics(
            avg_reward=0.0,
            pass_hat_ks={},
            avg_agent_cost=0.0,
            total_simulations=0,
            total_tasks=0,
            infra_error_count=infra_error_count,
        )

    df, df_pass_hat_k = prepare_dfs(results)
    avg_reward = df.reward.mean()
    pass_hat_ks = {}
    for column in df_pass_hat_k.columns:
        if match := re.match(r"pass\^(\d+)", column):
            k = int(match.group(1))
            pass_hat_ks[k] = df_pass_hat_k[column].mean()
    avg_agent_cost = df.agent_cost.mean()

    # Counts exclude infrastructure errors
    total_simulations = len(evaluated_sims)
    total_tasks = len(set(sim.task_id for sim in evaluated_sims))

    durations = [sim.duration for sim in evaluated_sims if sim.duration is not None]
    avg_duration = sum(durations) / len(durations) if durations else 0.0

    turn_sums = [count_turns(sim) for sim in evaluated_sims]
    n_turn = len(turn_sums) or 1
    avg_agent_turns = sum(t["agent_turns"] for t in turn_sums) / n_turn
    avg_user_turns = sum(t["user_turns"] for t in turn_sums) / n_turn
    avg_num_messages = sum(t["num_messages"] for t in turn_sums) / n_turn

    token_sums = [sum_message_tokens(sim) for sim in evaluated_sims]
    n_tok = len(token_sums) or 1
    avg_agent_prompt_tokens = (
        sum(t["agent_prompt_tokens"] for t in token_sums) / n_tok
    )
    avg_agent_completion_tokens = (
        sum(t["agent_completion_tokens"] for t in token_sums) / n_tok
    )
    avg_agent_total_tokens = (
        sum(t["agent_total_tokens"] for t in token_sums) / n_tok
    )
    avg_total_tokens = sum(t["total_tokens"] for t in token_sums) / n_tok

    # Action metrics
    total_read_actions = 0
    correct_read_actions = 0
    total_write_actions = 0
    correct_write_actions = 0

    # DB match
    db_match_count = 0
    db_mismatch_count = 0
    db_not_checked = 0

    # Termination
    termination_user_stop = 0
    termination_agent_stop = 0
    termination_max_steps = 0
    termination_error = 0

    for sim in evaluated_sims:
        # Action metrics
        if sim.reward_info and sim.reward_info.action_checks:
            partial = sim.reward_info.partial_action_reward
            if partial:
                if partial.get("read"):
                    total_read_actions += partial["read"]["count"]
                    correct_read_actions += partial["read"]["correct"]
                if partial.get("write"):
                    total_write_actions += partial["write"]["count"]
                    correct_write_actions += partial["write"]["correct"]

        # DB match
        if sim.reward_info and sim.reward_info.db_check:
            if sim.reward_info.db_check.db_match:
                db_match_count += 1
            else:
                db_mismatch_count += 1
        else:
            db_not_checked += 1

        # Termination reason
        if sim.termination_reason == TerminationReason.USER_STOP:
            termination_user_stop += 1
        elif sim.termination_reason == TerminationReason.AGENT_STOP:
            termination_agent_stop += 1
        elif sim.termination_reason == TerminationReason.MAX_STEPS:
            termination_max_steps += 1
        elif sim.termination_reason in (
            TerminationReason.TOO_MANY_ERRORS,
            TerminationReason.AGENT_ERROR,
            TerminationReason.USER_ERROR,
        ):
            termination_error += 1

    return AgentMetrics(
        avg_reward=avg_reward,
        pass_hat_ks=pass_hat_ks,
        avg_agent_cost=avg_agent_cost,
        avg_duration=avg_duration,
        avg_agent_turns=avg_agent_turns,
        avg_user_turns=avg_user_turns,
        avg_num_messages=avg_num_messages,
        avg_agent_prompt_tokens=avg_agent_prompt_tokens,
        avg_agent_completion_tokens=avg_agent_completion_tokens,
        avg_agent_total_tokens=avg_agent_total_tokens,
        avg_total_tokens=avg_total_tokens,
        total_simulations=total_simulations,
        total_tasks=total_tasks,
        infra_error_count=infra_error_count,
        total_read_actions=total_read_actions,
        correct_read_actions=correct_read_actions,
        total_write_actions=total_write_actions,
        correct_write_actions=correct_write_actions,
        db_match_count=db_match_count,
        db_mismatch_count=db_mismatch_count,
        db_not_checked=db_not_checked,
        termination_user_stop=termination_user_stop,
        termination_agent_stop=termination_agent_stop,
        termination_max_steps=termination_max_steps,
        termination_error=termination_error,
        termination_infrastructure_error=infra_error_count,
    )


def display_metrics(metrics: AgentMetrics) -> None:
    print(f"🏆 Average reward: {metrics.avg_reward}")
    print("📈 Pass^k")
    for k, pass_hat_k in metrics.pass_hat_ks.items():
        print(f"  k={k}: {pass_hat_k}")
    print(f"💰 Average agent cost: {metrics.avg_agent_cost}")
    print(f"⏱️  Average duration: {metrics.avg_duration:.2f}s")
    print(
        f"🔁 Average turns: agent {metrics.avg_agent_turns:.1f} / "
        f"user {metrics.avg_user_turns:.1f} / "
        f"messages {metrics.avg_num_messages:.1f}"
    )
    print(
        f"🔢 Average agent tokens: {metrics.avg_agent_total_tokens:.0f} "
        f"(prompt {metrics.avg_agent_prompt_tokens:.0f} / "
        f"completion {metrics.avg_agent_completion_tokens:.0f})"
    )
    print(f"🔢 Average total tokens (agent+user): {metrics.avg_total_tokens:.0f}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    args = parser.parse_args()
    results = Results.load(Path(args.results))
    metrics = compute_metrics(results)
    display_metrics(metrics)
