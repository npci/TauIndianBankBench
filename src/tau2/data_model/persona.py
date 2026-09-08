"""
Persona configuration for user simulator behavior.

IMPORTANT: User behavior/persona is controlled in THREE places:
1. Global simulation guidelines (data/tau2/user_simulator/simulation_guidelines.md) - Base behavior for all users
2. Task-specific persona (UserScenario.persona field) - Baked into task JSON at creation time
3. Runtime persona config (PersonaConfig, this file) - Configurable at simulation time

This allows for:
- Global defaults via guidelines
- Task-specific personas (e.g., "tech-savvy" vs "elderly confused user")
- Runtime variation (e.g., terseness level, interrupt tendency, quirks)
"""

import random
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Verbosity(str, Enum):
    """How verbose the user is in their responses."""

    STANDARD = "standard"  # Normal conversational responses
    MINIMAL = "minimal"  # 1-2 word responses when sufficient


class PersonaConfig(BaseModel):
    """
    Runtime configuration for user simulator persona attributes.
    These settings control behavioral aspects that can be varied at simulation time.

    Default behavior (no persona config): Standard verbosity with normal conversational flow.
    """

    verbosity: Verbosity = Field(
        default=Verbosity.STANDARD,
        description="How verbose the user's responses are. Default: STANDARD",
    )

    def to_guidelines_text(self) -> Optional[str]:
        """
        Convert persona config to additional guidelines text to append to system prompt.
        Returns None if no modifications needed (all defaults).
        """
        guidelines = []

        if self.verbosity == Verbosity.MINIMAL:
            guidelines.append(
                """
## MINIMAL VERBOSITY
You are terse in your responses.

- When a 1-2 word response is sufficient, respond with only those 1-2 words.
  Example: Agent: "Is this your savings account?" → You: "Yes" and NOT "Yes, it is my savings account."

- When a short phrase is sufficient, respond with the phrase instead of the full sentence.
  Example: Agent: "Which account and what period?" → You: "Savings, June 2026" and NOT "I would like a statement for my savings account for June 2026."

- Avoid filler words, pleasantries, or elaboration unless specifically needed.
  Example: Agent: "You're all set. Please let me know if you need anything else." → You: "Bye." and NOT "Thank you. That's all I needed."
""".strip()
            )

        return "\n\n".join(guidelines) if guidelines else None

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "PersonaConfig":
        """Create a PersonaConfig from a dictionary with support for weighted random values.

        This method allows flexible specification of persona attributes:
        - Explicit values: {"verbosity": "minimal"}
        - Weighted random: {"verbosity": {"minimal": 0.8, "standard": 0.2}}

        Args:
            config: Dictionary mapping attribute names to values or randomization specs.

        Returns:
            PersonaConfig with values either specified or randomly selected.

        Examples:
            # Explicit value
            PersonaConfig.from_dict({"verbosity": "minimal"})

            # Weighted random (80% minimal, 20% standard)
            PersonaConfig.from_dict({"verbosity": {"minimal": 0.8, "standard": 0.2}})

            # Weighted random across several options
            PersonaConfig.from_dict({
                "verbosity": {"minimal": 0.5, "standard": 0.5}
            })
        """
        resolved_config = {}

        # Generic processing for all fields
        for field_name, value in config.items():
            # Handle explicit value (string or other non-dict types)
            if not isinstance(value, dict):
                resolved_config[field_name] = value
            # Handle weighted random: {"option1": 0.8, "option2": 0.2}
            else:
                # Normalize probabilities to ensure they sum to 1.0
                total = sum(value.values())
                normalized = {k: v / total for k, v in value.items()}

                # Random selection based on weights
                rand_val = random.random()
                cumulative = 0.0
                for option, probability in normalized.items():
                    cumulative += probability
                    if rand_val < cumulative:
                        resolved_config[field_name] = option
                        break

        return cls(**resolved_config)
