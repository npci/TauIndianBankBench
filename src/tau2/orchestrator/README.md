# Orchestrator Module

This module drives simulations between agents, users, and environments: it routes messages between the participants, executes tool calls against the environment, and produces the resulting trajectory.

## Module Layout

| File | Contents |
|------|----------|
| `orchestrator.py` | `BaseOrchestrator` (shared lifecycle) and `Orchestrator` (half-duplex simulation) |
| `modes.py` | `CommunicationMode` enum (`HALF_DUPLEX`) |

---

## Orchestrator

`Orchestrator` runs turn-based simulations: agent and user alternate, each sending one complete message per turn.

### Communication Pattern

```
Agent ──message──> User ──message──> Agent ──tool_call──> Environment ──result──> Agent
```

Each participant takes turns sending complete messages. Only one party "speaks" at a time. A message must contain either text content or tool calls — never both, and never neither.

### Participant Interface

```python
class MyAgent(HalfDuplexAgent):
    def generate_next_message(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        """Generate a complete response to the incoming message."""
        ...
```

### Tool Execution

- **Synchronous**: Tool calls block until complete
- **Immediate**: Results returned in the same step

### Trajectory Structure

```python
trajectory: list[Message]  # Flat list of messages
```

### Compatible Classes

| Role | Classes |
|------|---------|
| Agent | `LLMAgent`, `LLMGTAgent`, `LLMSoloAgent` |
| User | `UserSimulator`, `DummyUser` |

### Usage

```python
from tau2.orchestrator.orchestrator import Orchestrator

orchestrator = Orchestrator(
    domain="indian_banking",
    agent=LLMAgent(tools=tools, domain_policy=policy, llm="gpt-4"),
    user=UserSimulator(llm="gpt-4", instructions=instructions, tools=user_tools),
    environment=environment,
    task=task,
    max_steps=100,
    seed=42,
)
result = orchestrator.run()
```

### Key Options

| Argument | Default | Effect |
|----------|---------|--------|
| `max_steps` | `100` | Maximum number of simulation steps before termination |
| `max_errors` | `10` | Maximum number of tool execution errors before termination |
| `seed` | `None` | Random seed passed to the agent and user for reproducibility |
| `solo_mode` | `False` | Agent runs without user interaction (see below) |
| `validate_communication` | `False` | Enforce the protocol rules on every message |
| `timeout` | `None` | Maximum wallclock time in seconds |

### Solo Mode

In solo mode the user is replaced by a `DummyUser` and the agent operates autonomously against the environment: it may only send tool calls, apart from the stop signal (`###STOP###`) that ends the simulation. It requires a solo-capable agent such as `LLMSoloAgent`.

### Termination

A simulation ends when the agent sends the stop signal, the user sends a stop signal, `max_steps` or `max_errors` is reached, the timeout expires, or — when `validate_communication=True` — a protocol violation is detected.
