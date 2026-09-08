# Contributing

Thanks for your interest in the `indian_banking` benchmark. This guide covers how to
report issues and submit changes.

## Ways to contribute

- **Bug reports**: open an issue with the task id, the agent/user-simulator models and
  their versions, the command line you ran, and the relevant entry from the run's
  `results.json` if you have one.
- **Score reports**: reference results for new models are welcome; include the full
  command line, serving configuration, and the run-to-run variance note from the README.
- **Harness or domain fixes**: see the caveat below before changing anything the models
  can see.

## Setup

```bash
pip install -r requirements.txt
python scripts/fetch_data.py   # the tests read the domain data
pip install --no-deps -e .
pytest tests                   # 86 tests, no network or GPU needed
```

The README covers serving models locally and running the benchmark end to end.

## A caveat on changing the domain

Reference results depend on the exact text of the policy, task instructions, tool
docstrings and knowledge-base documents. A wording change that looks cosmetic can move
scores. Fixes to model-visible text are only accepted together with re-run reference
numbers, or clearly marked as score-breaking.

All task and database content is synthetic. Do not add real customer data, credentials,
or personal information.

## Pull requests

1. Fork the repository and create a branch.
2. Keep changes focused: one fix or one addition per PR.
3. Make sure `pytest tests` passes.
4. Open a PR describing what changed and why, and whether it affects scores.

By contributing you agree to the [Code of Conduct](CODE_OF_CONDUCT.md) and that your
contribution is licensed under the repository's MIT licence.
