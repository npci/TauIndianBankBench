"""Tests for checkpoint save/resume logic."""

import json
import multiprocessing

from conftest import make_info, make_sim, make_task

from tau2.data_model.simulation import Results, TerminationReason
from tau2.runner.checkpoint import (
    create_checkpoint_replacer,
    create_checkpoint_saver,
    try_resume,
)


class TestTryResumeInfraErrorRemoval:
    """Test that try_resume properly handles infrastructure error retries."""

    def test_infra_errors_excluded_from_done_runs(self, tmp_path):
        """Infrastructure error sims should not be in done_runs so tasks get retried."""
        tasks = [make_task("t0"), make_task("t1"), make_task("t2")]
        info = make_info()

        prev_results = Results(
            info=info,
            tasks=tasks,
            simulations=[
                make_sim("t0", termination_reason=TerminationReason.USER_STOP),
                make_sim(
                    "t1",
                    termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
                ),
                make_sim("t2", termination_reason=TerminationReason.USER_STOP),
            ],
        )

        save_path = tmp_path / "results.json"
        with open(save_path, "w") as f:
            f.write(prev_results.model_dump_json(indent=2))

        new_results = Results(info=info, tasks=tasks, simulations=[])

        resumed, done_runs, _ = try_resume(
            save_path, new_results, tasks, num_trials=1, auto_resume=True
        )

        done_task_ids = {task_id for _, task_id, _ in done_runs}
        assert "t0" in done_task_ids
        assert "t1" not in done_task_ids, (
            "Infrastructure error task should NOT be in done_runs"
        )
        assert "t2" in done_task_ids

    def test_infra_errors_removed_from_resumed_simulations(self, tmp_path):
        """Infrastructure error sims should be removed from the resumed results."""
        tasks = [make_task("t0"), make_task("t1")]
        info = make_info()

        prev_results = Results(
            info=info,
            tasks=tasks,
            simulations=[
                make_sim("t0", termination_reason=TerminationReason.USER_STOP),
                make_sim(
                    "t1",
                    termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
                ),
            ],
        )

        save_path = tmp_path / "results.json"
        with open(save_path, "w") as f:
            f.write(prev_results.model_dump_json(indent=2))

        new_results = Results(info=info, tasks=tasks, simulations=[])

        resumed, _, _ = try_resume(
            save_path, new_results, tasks, num_trials=1, auto_resume=True
        )

        assert len(resumed.simulations) == 1
        assert resumed.simulations[0].task_id == "t0"

    def test_infra_errors_removed_from_disk(self, tmp_path):
        """After try_resume, the on-disk file should NOT contain infra error sims.

        An on-disk file that retains infra-error entries makes
        create_checkpoint_saver's duplicate check reject the retried results.
        """
        tasks = [make_task("t0"), make_task("t1")]
        info = make_info()

        prev_results = Results(
            info=info,
            tasks=tasks,
            simulations=[
                make_sim("t0", termination_reason=TerminationReason.USER_STOP),
                make_sim(
                    "t1",
                    termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
                ),
            ],
        )

        save_path = tmp_path / "results.json"
        with open(save_path, "w") as f:
            f.write(prev_results.model_dump_json(indent=2))

        new_results = Results(info=info, tasks=tasks, simulations=[])

        try_resume(save_path, new_results, tasks, num_trials=1, auto_resume=True)

        with open(save_path, "r") as f:
            on_disk = json.load(f)

        assert len(on_disk["simulations"]) == 1
        assert on_disk["simulations"][0]["task_id"] == "t0"

    def test_checkpoint_saver_works_after_infra_error_resume(self, tmp_path):
        """End-to-end: after resuming, saving a retried sim should succeed."""
        tasks = [make_task("t0"), make_task("t1")]
        info = make_info()
        seed = 42

        prev_results = Results(
            info=info,
            tasks=tasks,
            simulations=[
                make_sim("t0", seed=seed),
                make_sim(
                    "t1",
                    seed=seed,
                    termination_reason=TerminationReason.INFRASTRUCTURE_ERROR,
                ),
            ],
        )

        save_path = tmp_path / "results.json"
        with open(save_path, "w") as f:
            f.write(prev_results.model_dump_json(indent=2))

        new_results = Results(info=info, tasks=tasks, simulations=[])

        try_resume(save_path, new_results, tasks, num_trials=1, auto_resume=True)

        lock = multiprocessing.Lock()
        save_fn = create_checkpoint_saver(save_path, lock)

        retried_sim = make_sim("t1", seed=seed)
        save_fn(retried_sim)

        with open(save_path, "r") as f:
            on_disk = json.load(f)

        assert len(on_disk["simulations"]) == 2
        task_ids = {s["task_id"] for s in on_disk["simulations"]}
        assert task_ids == {"t0", "t1"}
        t1_sim = next(s for s in on_disk["simulations"] if s["task_id"] == "t1")
        assert t1_sim["termination_reason"] == "user_stop"

    def test_no_infra_errors_skips_resave(self, tmp_path):
        """When there are no infra errors and no new tasks, don't rewrite the file."""
        tasks = [make_task("t0")]
        info = make_info()

        prev_results = Results(
            info=info,
            tasks=tasks,
            simulations=[make_sim("t0")],
        )

        save_path = tmp_path / "results.json"
        with open(save_path, "w") as f:
            f.write(prev_results.model_dump_json(indent=2))

        original_mtime = save_path.stat().st_mtime

        new_results = Results(info=info, tasks=tasks, simulations=[])

        import time

        time.sleep(0.05)

        try_resume(save_path, new_results, tasks, num_trials=1, auto_resume=True)

        assert save_path.stat().st_mtime == original_mtime


class TestCheckpointSaver:
    """Tests for checkpoint save function."""

    def test_saves_new_simulation(self, tmp_path):
        save_path = tmp_path / "results.json"
        data = {"simulations": [], "info": {}, "tasks": []}
        with open(save_path, "w") as f:
            json.dump(data, f)

        lock = multiprocessing.Lock()
        save_fn = create_checkpoint_saver(save_path, lock)

        sim = make_sim("t0")
        save_fn(sim)

        with open(save_path, "r") as f:
            on_disk = json.load(f)

        assert len(on_disk["simulations"]) == 1
        assert on_disk["simulations"][0]["task_id"] == "t0"

    def test_skips_duplicate(self, tmp_path):
        sim = make_sim("t0")
        save_path = tmp_path / "results.json"
        data = {"simulations": [sim.model_dump()], "info": {}, "tasks": []}
        with open(save_path, "w") as f:
            json.dump(data, f)

        lock = multiprocessing.Lock()
        save_fn = create_checkpoint_saver(save_path, lock)

        save_fn(sim)

        with open(save_path, "r") as f:
            on_disk = json.load(f)

        assert len(on_disk["simulations"]) == 1


class TestCheckpointReplacer:
    """Tests for checkpoint replace function."""

    def test_replaces_existing_simulation(self, tmp_path):
        old_sim = make_sim(
            "t0", termination_reason=TerminationReason.INFRASTRUCTURE_ERROR
        )
        save_path = tmp_path / "results.json"
        data = {"simulations": [old_sim.model_dump()], "info": {}, "tasks": []}
        with open(save_path, "w") as f:
            json.dump(data, f)

        lock = multiprocessing.Lock()
        replace_fn = create_checkpoint_replacer(save_path, lock)

        new_sim = make_sim("t0")
        replace_fn((0, "t0", 42), new_sim)

        with open(save_path, "r") as f:
            on_disk = json.load(f)

        assert len(on_disk["simulations"]) == 1
        assert on_disk["simulations"][0]["termination_reason"] == "user_stop"
