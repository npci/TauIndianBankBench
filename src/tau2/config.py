import os as _os

# =============================================================================
# SIMULATION DEFAULTS (overridable via CLI)
# =============================================================================
DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_ERRORS = 10
DEFAULT_SEED = 300
DEFAULT_MAX_CONCURRENCY = 3
DEFAULT_NUM_TRIALS = 1
DEFAULT_SAVE_TO = None
DEFAULT_LOG_LEVEL = "ERROR"

# =============================================================================
# LLM DEFAULTS (overridable via CLI)
# =============================================================================
DEFAULT_AGENT_IMPLEMENTATION = "llm_agent"
DEFAULT_USER_IMPLEMENTATION = "user_simulator"
DEFAULT_LLM_AGENT = "gpt-4.1-2025-04-14"
DEFAULT_LLM_USER = "gpt-4.1-2025-04-14"
DEFAULT_LLM_TEMPERATURE_AGENT = 0.0
DEFAULT_LLM_TEMPERATURE_USER = 0.0
DEFAULT_LLM_ARGS_AGENT = {"temperature": DEFAULT_LLM_TEMPERATURE_AGENT}
DEFAULT_LLM_ARGS_USER = {"temperature": DEFAULT_LLM_TEMPERATURE_USER}

# Judge for natural-language assertions. The indian_banking task suite scores on
# database state and required actions and never invokes this judge, so there is
# no default endpoint; set these to point at a judge of your own.
DEFAULT_LLM_NL_ASSERTIONS = _os.environ.get("TAU2_NL_JUDGE_MODEL", "")
DEFAULT_LLM_NL_ASSERTIONS_ARGS = {
    "temperature": 0.0,
    "max_tokens": 8192,
}
for _name, _value in (
    ("api_base", _os.environ.get("TAU2_NL_JUDGE_API_BASE")),
    ("api_key", _os.environ.get("TAU2_NL_JUDGE_API_KEY")),
):
    if _value:
        DEFAULT_LLM_NL_ASSERTIONS_ARGS[_name] = _value

DEFAULT_LLM_ENV_INTERFACE = "gpt-4.1-2025-04-14"
DEFAULT_LLM_ENV_INTERFACE_ARGS = {"temperature": 0.0}

# LLM debug logging
DEFAULT_LLM_LOG_MODE = "latest"  # Options: "all", "latest"

# =============================================================================
# LLM INFRASTRUCTURE (fixed operational constants)
# =============================================================================
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_MIN_WAIT = 1.0  # seconds

# LiteLLM cache
LLM_CACHE_ENABLED = False
DEFAULT_LLM_CACHE_TYPE = "redis"

# Redis, used only when LLM_CACHE_ENABLED is turned on above.
REDIS_HOST = _os.environ.get("TAU2_REDIS_HOST", "localhost")
REDIS_PORT = int(_os.environ.get("TAU2_REDIS_PORT", "6379"))
REDIS_PASSWORD = _os.environ.get("TAU2_REDIS_PASSWORD", "")
REDIS_PREFIX = _os.environ.get("TAU2_REDIS_PREFIX", "tau2")
REDIS_CACHE_VERSION = "v1"
REDIS_CACHE_TTL = 60 * 60 * 24 * 30

# Langfuse
USE_LANGFUSE = False

# =============================================================================
# DISPLAY
# =============================================================================
TERM_DARK_MODE = True
