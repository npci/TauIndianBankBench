# Agents

An agent is the party under evaluation: it receives the customer's messages and
tool results, and replies with text or tool calls, one turn at a time.

## Classes

| Class | File | Role |
| --- | --- | --- |
| `HalfDuplexParticipant` | `base/participant.py` | Protocol every turn-based participant implements: `get_init_state()`, `generate_next_message()`, `is_stop()`. |
| `HalfDuplexAgent` | `base_agent.py` | Agent base class over that protocol, plus message-format validation. |
| `LLMConfigMixin` | `base/llm_config.py` | Holds the LLM name and call arguments. |
| `LLMAgent` | `llm_agent.py` | The default agent: system prompt = the domain's instruction block and policy, tools = the domain's tools. |
| `LLMGTAgent` | `llm_agent.py` | Same, but also shown the task's gold actions; for debugging tasks, not for scoring. |
| `LLMSoloAgent` | `llm_agent.py` | Agent that runs without a user simulator; the `indian_banking` domain does not support this mode. |

The agent's system prompt is built from `data/tau2/domains/indian_banking/agent_instruction.txt`
(overridable with `TAU2_AGENT_INSTRUCTION_FILE`) followed by the domain policy.

## Writing an agent

1. Subclass `HalfDuplexAgent` (and `LLMConfigMixin` if it calls an LLM) and
   define a pydantic state class holding whatever the agent needs between turns.
2. Implement `get_init_state(message_history)`, `generate_next_message(message, state)`
   returning `(AssistantMessage, state)`, and `is_stop(message)`.
3. Add a factory `create_my_agent(tools, domain_policy, **kwargs)` and register it
   in `registry.py` with `registry.register_agent_factory(create_my_agent, "my_agent")`.
4. Run it with `tau2 run --domain indian_banking --agent my_agent ...`.

`AssistantMessage` must carry either text content or tool calls; the orchestrator
rejects anything else and counts it as an agent error.
