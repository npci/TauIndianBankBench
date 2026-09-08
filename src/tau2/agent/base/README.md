# Base agent components

| File | Contents |
| --- | --- |
| `participant.py` | `HalfDuplexParticipant`, the generic protocol for a turn-based participant: `get_init_state()`, `generate_next_message()`, `is_stop()`, `set_seed()`. Both agents and user simulators implement it. |
| `llm_config.py` | `LLMConfigMixin`: stores the LLM name and call arguments (`temperature`, `max_tokens`, `api_base`, ...) used when the participant calls a model. |

`HalfDuplexAgent` in `../base_agent.py` and `HalfDuplexUser` in
`../../user/user_simulator_base.py` are the two concrete bases built on the
protocol. Combine `LLMConfigMixin` with one of them for an LLM-backed participant:

```python
class MyAgent(LLMConfigMixin, HalfDuplexAgent[MyState]):
    ...
```

The mixin must come first in the inheritance order so its `__init__` runs
before the participant base's.
