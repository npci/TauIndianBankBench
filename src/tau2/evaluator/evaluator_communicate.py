from tau2.data_model.message import AssistantMessage, Message
from tau2.data_model.simulation import CommunicateCheck, RewardInfo
from tau2.data_model.tasks import RewardType, Task
from tau2.evaluator.evaluator_base import EvaluatorBase


class CommunicateEvaluator(EvaluatorBase[Message]):
    """
    Evaluates whether or not the agent communicated the required information.
    """

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
    ) -> RewardInfo:
        """
        Calculate the reward based on whether the agent communicated the required information.
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                info={"notes": "No evaluation criteria"},
                reward_breakdown={RewardType.COMMUNICATE: 1.0},
            )
        communicate_info = task.evaluation_criteria.communicate_info
        communicate_info_checks = (
            cls.evaluate_communicate_info(full_trajectory, communicate_info)
            if communicate_info
            else []
        )
        communicate_info_checks.extend(
            cls._customer_facing_checks(task, full_trajectory)
        )

        if not communicate_info_checks:
            return RewardInfo(
                reward=1.0,
                info={"note": "No communicate_info to evaluate"},
                reward_breakdown={RewardType.COMMUNICATE: 1.0},
            )

        return cls._reward_from_checks(communicate_info_checks)

    @classmethod
    def _customer_facing_checks(
        cls,
        task: Task,
        full_trajectory: list[Message],
    ) -> list[CommunicateCheck]:
        from tau2.domains.indian_banking.customer_facing import (
            evaluate_banking_customer_facing,
            is_banking_task,
        )

        if not is_banking_task(task):
            return []
        return [evaluate_banking_customer_facing(full_trajectory)]

    @classmethod
    def _reward_from_checks(
        cls, communicate_info_checks: list[CommunicateCheck]
    ) -> RewardInfo:
        all_expectations_met = all(result.met for result in communicate_info_checks)
        reward = 1.0 if all_expectations_met else 0.0
        return RewardInfo(
            reward=reward,
            communicate_checks=communicate_info_checks,
            reward_breakdown={RewardType.COMMUNICATE: reward},
        )

    @classmethod
    def evaluate_communicate_info(
        cls,
        full_trajectory: list[Message],
        communicate_info: list[str],
    ) -> list[CommunicateCheck]:
        """
        Evaluate whether the agent communicates the information correctly.
        """
        if len(communicate_info) == 0:
            return []

        outputs = []
        for info_str in communicate_info:
            found = False
            for message in full_trajectory:
                if not isinstance(message, AssistantMessage):
                    continue
                if not message.has_text_content():
                    continue
                if info_str.lower() in message.content.lower().replace(",", ""):
                    found = True
                    break
            if found:
                met = True
                justification = f"Information '{info_str}' communicated in the message:\n '{message.content}'"
            else:
                met = False
                justification = f"Information '{info_str}' not communicated."
            outputs.append(
                CommunicateCheck(
                    info=info_str,
                    met=met,
                    justification=justification,
                )
            )
        return outputs
