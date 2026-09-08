"""
Base agent components.

This module exports the foundational building blocks for creating
conversation participants.
"""

# LLM configuration mixin
from tau2.agent.base.llm_config import LLMConfigMixin

# Protocol base classes
from tau2.agent.base.participant import HalfDuplexParticipant

__all__ = [
    "HalfDuplexParticipant",
    "LLMConfigMixin",
]
