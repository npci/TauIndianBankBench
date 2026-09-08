# tau2.runner

Simulation execution framework with a layered architecture for running evaluations at different levels of control.

## Module Structure

```
runner/
├── __init__.py        # Package exports
├── simulation.py      # Layer 1: run_simulation()
├── build.py           # Layer 2: build_* functions
├── batch.py           # Layer 3: run_domain(), run_tasks(), run_single_task()
├── helpers.py         # Task loading, run metadata, utilities
├── checkpoint.py      # Save/resume logic for batch runs
├── progress.py        # Retry logic and status monitoring
└── README.md          # This file
```

## Layers

### Layer 1: `simulation.py` -- Execute

Pure simulation execution and evaluation. Takes a fully constructed orchestrator, runs it, evaluates the result, and returns a `SimulationRun` with `reward_info` attached.

- **No registry dependency**: Everything is encapsulated in the orchestrator.
- **No config parsing**: No `RunConfig` needed.
- **No side effects**: No logging setup, no file saving.

```python
from tau2.runner import run_simulation

result = run_simulation(orchestrator)
```

### Layer 2: `build.py` -- Build

Turns names and configuration into live instances using the registry for name resolution.

**Low-level builders** (take individual parameters):
- `build_environment(domain)` -- Resolve domain name to environment instance.
- `build_agent(agent_name, environment)` -- Resolve agent name to agent instance.
- `build_user(user_name, environment, task)` -- Resolve user name to user instance.

**High-level builders** (take `RunConfig`):
- `build_text_orchestrator(config, task)` -- Build an `Orchestrator` from `TextRunConfig`.
- `build_orchestrator(config, task)` -- Build an orchestrator from any `RunConfig`.

```python
from tau2.runner import build_text_orchestrator, run_simulation
from tau2 import TextRunConfig

config = TextRunConfig(
    domain="indian_banking", agent="llm_agent", llm_agent="openai/gpt-4.1"
)
orchestrator = build_text_orchestrator(config, task, seed=42)
result = run_simulation(orchestrator)
```

### Layer 3: `batch.py` -- Batch

High-level batch execution with all operational concerns:

- **Concurrency**: Thread pool with configurable `max_concurrency`.
- **Checkpointing**: Atomic save/resume via `checkpoint.py`.
- **Retries**: Configurable retry with delay via `progress.py`.
- **Status monitoring**: Periodic progress display (every 30s).
- **Side effects**: Per-task logging.

Entry points:
- `run_domain(config)` -- Full pipeline: load tasks, filter, run batch, display metrics.
- `run_tasks(config, tasks)` -- Run a list of tasks with all batch features.
- `run_single_task(config, task)` -- Run one task with logging and side effects.

```python
from tau2.runner import run_domain
from tau2 import TextRunConfig

config = TextRunConfig(
    domain="indian_banking", agent="llm_agent", llm_agent="openai/gpt-4.1"
)
results = run_domain(config)
```

## Supporting Modules

- **`helpers.py`**: Task loading (`get_tasks`, `load_tasks`), run metadata (`get_info`, `make_run_name`), registry queries (`get_options`, `get_environment_info`).
- **`checkpoint.py`**: `try_resume()` for resuming from existing results, `create_checkpoint_saver()` for atomic saves.
- **`progress.py`**: `run_with_retry()` for retry logic, `StatusMonitor` for periodic progress display.

## Relationship to `tau2.run`

`tau2.run` is a thin facade over this package: it re-exports `run_domain`,
`run_tasks`, `run_single_task`, `run_simulation`, `get_tasks` and the other
entry points, so either import path reaches the same code.
