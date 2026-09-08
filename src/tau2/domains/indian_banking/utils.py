"""Data paths for the Indian banking domain."""

from __future__ import annotations

from tau2.utils.utils import DATA_DIR

INDIAN_BANKING_DATA_DIR = DATA_DIR / "tau2" / "domains" / "indian_banking"
INDIAN_BANKING_DB_PATH = INDIAN_BANKING_DATA_DIR / "db.json"
INDIAN_BANKING_POLICY_PATH = INDIAN_BANKING_DATA_DIR / "policy.md"
INDIAN_BANKING_TASKS_PATH = INDIAN_BANKING_DATA_DIR / "tasks.json"
INDIAN_BANKING_AGENT_INSTRUCTION_PATH = (
    INDIAN_BANKING_DATA_DIR / "agent_instruction.txt"
)
# Markdown corpus backing `search_knowledge_base`. One file per topic, split
# into `##` sections at load time.
INDIAN_BANKING_KB_DIR = INDIAN_BANKING_DATA_DIR / "kb"
