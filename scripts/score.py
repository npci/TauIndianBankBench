#!/usr/bin/env python3
"""Score an `indian_banking` results.json.

    python scripts/score.py data/simulations/<run>/results.json
    python scripts/score.py --compare RUN_A.json RUN_B.json

Single-file mode prints the strict mean reward, the episode count, the mean per
task category, and the termination profile.

Compare mode restricts both runs to the task ids they share, so two runs that
covered different subsets (or one that is still in flight) are still compared
like for like.

"Strict" means a task counts as solved only if every trial of it scored a full
1.0; with the default single trial that is just the mean reward. Reading the
file needs nothing but the standard library, so it also works outside the
benchmark's virtualenv.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

# Task ids look like `<corpus>_<family>_<index>`, e.g. `rl_hpbal_0005`; the
# family's leading prefix picks the category. Longer prefixes win so that
# `sq2...` is not swallowed by `sq`.
CATEGORIES: tuple[tuple[str, str], ...] = (
    ("hp", "hp    happy path"),
    ("sq2", "sq2   multi-step scenario"),
    ("sq", "sq    scenario"),
    ("edga", "edga  edge case A"),
    ("edgb", "edgb  edge case B"),
    ("tl", "tl    tool lookup"),
)
_ORDER = [c for c, _ in CATEGORIES]
_LABEL = dict(CATEGORIES)


def category_of(task_id: str) -> str:
    """Map a task id to its category prefix, or 'other'."""
    parts = task_id.split("_")
    family = parts[1] if len(parts) > 1 else parts[0]
    best = None
    for prefix in _ORDER:
        if family.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return best or "other"


def _read_json(path: Path) -> dict:
    try:
        with path.open() as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        sys.exit(f"score.py: {path} is not valid JSON ({exc})")


def load(path: str | Path) -> dict:
    """Read a run in either storage format.

    "json": one results.json holding everything.
    "dir":  a results.json holding only the metadata, with one file per
            simulation in a sibling simulations/ directory.
    """
    path = Path(path)
    if not path.exists():
        sys.exit(f"score.py: no such file: {path}")
    if path.is_dir():
        candidate = path / "results.json"
        if not candidate.exists():
            sys.exit(f"score.py: {path} is a directory with no results.json in it")
        path = candidate

    data = _read_json(path)
    if not isinstance(data, dict) or "info" not in data:
        sys.exit(f"score.py: {path} does not look like a tau2 results file")

    if not data.get("simulations"):
        sims_dir = path.parent / "simulations"
        if sims_dir.is_dir():
            data["simulations"] = [
                _read_json(f) for f in sorted(sims_dir.glob("*.json"))
            ]
    if "simulations" not in data:
        sys.exit(f"score.py: {path} does not look like a tau2 results file")
    return data


def rewards_by_task(data: dict) -> dict[str, list[float]]:
    """Every scored episode's reward, grouped by task id."""
    out: dict[str, list[float]] = collections.defaultdict(list)
    for sim in data["simulations"]:
        info = sim.get("reward_info") or {}
        reward = info.get("reward")
        if reward is None:
            continue
        out[sim["task_id"]].append(float(reward))
    return dict(out)


def strict(rewards: list[float]) -> float:
    """1.0 only if every trial of the task scored a full 1.0."""
    return 1.0 if rewards and all(r >= 1.0 for r in rewards) else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def by_category(scores: dict[str, float]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = collections.defaultdict(list)
    for task_id, value in scores.items():
        grouped[category_of(task_id)].append(value)
    return grouped


def category_rows(grouped: dict[str, list[float]]):
    for key in _ORDER + ["other"]:
        if key in grouped:
            yield _LABEL.get(key, f"{key}   uncategorised"), grouped[key]


def report_one(path: str) -> None:
    data = load(path)
    per_task = rewards_by_task(data)
    if not per_task:
        sys.exit(f"score.py: {path} has no scored episodes yet")

    scores = {task_id: strict(rs) for task_id, rs in per_task.items()}
    episodes = sum(len(rs) for rs in per_task.values())
    trials = max(len(rs) for rs in per_task.values())

    info = data.get("info") or {}
    agent_info = info.get("agent_info") or {}
    user_info = info.get("user_info") or {}

    print(f"file            : {path}")
    if agent_info.get("llm"):
        print(f"agent           : {agent_info['llm']}")
    if user_info.get("llm"):
        print(f"user simulator  : {user_info['llm']}")
    print(f"tasks scored    : {len(scores)}")
    print(
        f"episodes scored : {episodes}"
        + (f"  ({trials} trials/task)" if trials > 1 else "")
    )
    print(f"strict mean     : {mean(list(scores.values())):.4f}")
    print(f"solved          : {int(sum(scores.values()))}/{len(scores)}")

    print("\nper category (strict mean, n)")
    for label, values in category_rows(by_category(scores)):
        print(f"  {label:26s} {mean(values):.4f}   n={len(values)}")

    print("\ntermination profile")
    terminations = collections.Counter(
        sim.get("termination_reason") or "unknown" for sim in data["simulations"]
    )
    total = sum(terminations.values())
    for reason, count in terminations.most_common():
        print(f"  {reason:26s} {count:5d}   {100 * count / total:5.1f}%")


def report_compare(path_a: str, path_b: str) -> None:
    a_scores = {t: strict(r) for t, r in rewards_by_task(load(path_a)).items()}
    b_scores = {t: strict(r) for t, r in rewards_by_task(load(path_b)).items()}
    shared = sorted(set(a_scores) & set(b_scores))
    if not shared:
        sys.exit("score.py: the two runs share no scored task ids")

    print(f"A : {path_a}   ({len(a_scores)} tasks scored)")
    print(f"B : {path_b}   ({len(b_scores)} tasks scored)")
    only_a, only_b = len(a_scores) - len(shared), len(b_scores) - len(shared)
    print(
        f"comparing the {len(shared)} shared task ids"
        + (f"  (A-only: {only_a}, B-only: {only_b})" if only_a or only_b else "")
    )

    a_mean = mean([a_scores[t] for t in shared])
    b_mean = mean([b_scores[t] for t in shared])
    print(
        f"\nstrict mean  A={a_mean:.4f}  B={b_mean:.4f}  delta={b_mean - a_mean:+.4f}"
    )

    gained = [t for t in shared if b_scores[t] > a_scores[t]]
    lost = [t for t in shared if b_scores[t] < a_scores[t]]
    print(f"A fail -> B pass : {len(gained)}")
    print(f"A pass -> B fail : {len(lost)}")

    grouped_a = by_category({t: a_scores[t] for t in shared})
    grouped_b = by_category({t: b_scores[t] for t in shared})
    print("\nper category (A, B, delta, n)")
    for label, values in category_rows(grouped_a):
        key = label.split()[0]
        b_values = grouped_b.get(key, [])
        am, bm = mean(values), mean(b_values)
        print(f"  {label:26s} {am:.4f}   {bm:.4f}   {bm - am:+.4f}   n={len(values)}")

    if lost:
        print("\nregressions (first 20)")
        for task_id in lost[:20]:
            print(f"  {task_id}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score an indian_banking results.json.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "results", nargs="?", help="path to a results.json (or its run directory)"
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="compare two results files on the task ids they share",
    )
    args = parser.parse_args()

    if args.compare:
        report_compare(*args.compare)
    elif args.results:
        report_one(args.results)
    else:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
