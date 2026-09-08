"""
Tau2: Conversational Agent Benchmark Framework

Main exports for easy access to key components.
"""

# Runner package: clean API for simulation execution
# - Layer 1: run_simulation (execute pre-built orchestrator)
# - Layer 2: build_* functions (construct instances from config/names)
# - Layer 3: run_domain, run_tasks, run_single_task (batch execution)
import tau2.runner as runner
from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import HalfDuplexAgent
from tau2.agent.llm_agent import LLMAgent, LLMSoloAgent
from tau2.data_model.simulation import (
    BaseRunConfig,
    RunConfig,
    SimulationRun,
    TextRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.modes import CommunicationMode
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import Registry, registry
from tau2.run import run_domain
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import HalfDuplexUser
from tau2.utils.display import ConsoleDisplay, MarkdownDisplay

__all__ = [
    # Core
    "Orchestrator",
    "LLMAgent",
    "LLMSoloAgent",
    "LLMConfigMixin",
    "UserSimulator",
    "HalfDuplexAgent",
    "HalfDuplexUser",
    "Environment",
    "Registry",
    "registry",
    "SimulationRun",
    "Task",
    "evaluate_simulation",
    "EvaluationType",
    "BaseRunConfig",
    "TextRunConfig",
    "RunConfig",
    "run_domain",
    "runner",
    "CommunicationMode",
    # Utils
    "ConsoleDisplay",
    "MarkdownDisplay",
]
