import argparse
import json

from tau2.config import (
    DEFAULT_AGENT_IMPLEMENTATION,
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_LOG_MODE,
    DEFAULT_LLM_TEMPERATURE_AGENT,
    DEFAULT_LLM_TEMPERATURE_USER,
    DEFAULT_LLM_USER,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_TRIALS,
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_MIN_WAIT,
    DEFAULT_SEED,
    DEFAULT_USER_IMPLEMENTATION,
)
from tau2.data_model.simulation import TextRunConfig
from tau2.run import get_options, run_domain


def add_run_args(parser):
    """Add run arguments to a parser."""
    domains = get_options().domains
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=domains,
        help="The domain to run the simulation on",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="The number of times each task is run. Default is 1.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=DEFAULT_AGENT_IMPLEMENTATION,
        choices=get_options().agents,
        help=f"The agent implementation to use. Default is {DEFAULT_AGENT_IMPLEMENTATION}.",
    )
    parser.add_argument(
        "--agent-llm",
        type=str,
        default=DEFAULT_LLM_AGENT,
        help=f"The LLM to use for the agent. Default is {DEFAULT_LLM_AGENT}.",
    )
    parser.add_argument(
        "--agent-llm-args",
        type=json.loads,
        default={"temperature": DEFAULT_LLM_TEMPERATURE_AGENT},
        help=f"The arguments to pass to the LLM for the agent. Default is '{{\"temperature\": {DEFAULT_LLM_TEMPERATURE_AGENT}}}'.",
    )
    parser.add_argument(
        "--user",
        type=str,
        choices=get_options().users,
        default=DEFAULT_USER_IMPLEMENTATION,
        help=f"The user implementation to use. Default is {DEFAULT_USER_IMPLEMENTATION}.",
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default=DEFAULT_LLM_USER,
        help=f"The LLM to use for the user. Default is {DEFAULT_LLM_USER}.",
    )
    parser.add_argument(
        "--user-llm-args",
        type=json.loads,
        default={"temperature": DEFAULT_LLM_TEMPERATURE_USER},
        help=f"The arguments to pass to the LLM for the user. Default is '{{\"temperature\": {DEFAULT_LLM_TEMPERATURE_USER}}}'.",
    )
    parser.add_argument(
        "--task-set-name",
        type=str,
        default=None,
        choices=get_options().task_sets,
        help="The task set to run the simulation on. If not provided, will load default task set for the domain.",
    )
    parser.add_argument(
        "--task-split-name",
        type=str,
        default="base",
        help="The task split to run the simulation on. If not provided, will load 'base' split.",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        help="(Optional) run only the tasks with the given IDs. If not provided, will run all tasks.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="The number of tasks to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"The maximum number of steps to run the simulation. Default is {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help=f"The maximum number of tool errors allowed in a row in the simulation. Default is {DEFAULT_MAX_ERRORS}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Maximum wallclock time in seconds for each simulation. No timeout by default.",
    )
    parser.add_argument(
        "--save-to",
        type=str,
        required=False,
        help="The path to save the simulation results. Will be saved to data/simulations/<save_to>/results.json. If not provided, will save to <timestamp>_<domain>_<agent>_<user>. If the file already exists, it will try to resume the run.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"The maximum number of concurrent simulations to run. Default is {DEFAULT_MAX_CONCURRENCY}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"The seed to use for the simulation. Default is {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        help=f"The log level to use for the simulation. Default is {DEFAULT_LOG_LEVEL}.",
    )
    parser.add_argument(
        "--verbose-logs",
        action="store_true",
        default=False,
        help="Enable verbose logging: saves LLM call logs and per-task logs. "
        "Files are saved to the save directory (auto-generated if --save-to not specified).",
    )
    parser.add_argument(
        "--llm-log-mode",
        type=str,
        choices=["all", "latest"],
        default=DEFAULT_LLM_LOG_MODE,
        help="LLM debug logging mode. Only takes effect when --verbose-logs is enabled. "
        "'all' saves every LLM call (can generate many files), "
        "'latest' keeps only the most recent call of each type (saves space). "
        f"Default is '{DEFAULT_LLM_LOG_MODE}'. Ignored if --verbose-logs is not specified.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help=f"Maximum number of retries for failed tasks. Default is {DEFAULT_RETRY_ATTEMPTS}.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=DEFAULT_RETRY_MIN_WAIT,
        help=f"Delay in seconds between retries. Default is {DEFAULT_RETRY_MIN_WAIT}.",
    )
    parser.add_argument(
        "--enforce-communication-protocol",
        action="store_true",
        default=False,
        help="Enforce communication protocol rules (e.g., no mixed messages with text and tool calls). Default is False.",
    )
    # Resume mode
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=False,
        help="Automatically resume from existing save file without prompting (for non-interactive runs).",
    )


def _get_version() -> str:
    from tau2.utils.utils import get_tau2_version

    return get_tau2_version()


def run_intro():
    """Display a rich intro page describing what the benchmark can do."""
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    version = _get_version()

    # ── Banner ──────────────────────────────────────────────────────────
    banner = Text(justify="center")
    banner.append("\n")
    banner.append("tau2-bench", style="bold cyan")
    banner.append(f"  v{version}\n", style="dim")
    banner.append(
        "A simulation framework for evaluating\n"
        "conversational customer-service agents\n",
        style="italic",
    )
    console.print(
        Panel(
            banner,
            border_style="cyan",
            padding=(1, 4),
        )
    )

    # ── Domains ─────────────────────────────────────────────────────────
    domain_table = Table(
        title="Domains",
        box=box.SIMPLE_HEAVY,
        title_style="bold magenta",
        show_edge=False,
        pad_edge=False,
        padding=(0, 2),
    )
    domain_table.add_column("Domain", style="bold", no_wrap=True)
    domain_table.add_column("Description")
    domain_table.add_row(
        "indian_banking",
        "Indian retail banking: accounts, deposits, loans, cards, mandates",
    )
    console.print(domain_table)
    console.print()

    # ── Commands ────────────────────────────────────────────────────────
    cmd_table = Table(
        title="Commands",
        box=box.SIMPLE_HEAVY,
        title_style="bold magenta",
        show_edge=False,
        pad_edge=False,
        padding=(0, 2),
    )
    cmd_table.add_column("Command", style="bold green", no_wrap=True)
    cmd_table.add_column("What it does")
    cmd_table.add_row("tau2 run", "Run a benchmark evaluation against a domain")
    cmd_table.add_row("tau2 view", "Browse and inspect simulation results")
    cmd_table.add_row(
        "tau2 domain <name>",
        "Serve a domain's policy + tool docs at http://127.0.0.1:8004/redoc",
    )
    cmd_table.add_row(
        "tau2 evaluate-trajs <paths>", "Re-evaluate trajectories and recompute rewards"
    )
    cmd_table.add_row("tau2 check-data", "Verify data directory is set up correctly")
    cmd_table.add_row(
        "tau2 convert-results <path>", "Switch results between storage formats"
    )
    cmd_table.add_row("tau2 intro", "Show this page")
    console.print(cmd_table)
    console.print()

    # ── Quick Start ─────────────────────────────────────────────────────
    from rich.syntax import Syntax

    quick_start = (
        "# 1. Verify your setup\n"
        "tau2 check-data\n"
        "\n"
        "# 2. Run an evaluation\n"
        "tau2 run --domain indian_banking --agent-llm openai/gpt-4.1 "
        "--user-llm openai/gpt-4.1 --num-trials 1 --num-tasks 5\n"
        "\n"
        "# 3. Browse results\n"
        "tau2 view\n"
    )
    console.print(
        Panel(
            Syntax(quick_start, "bash", theme="monokai", line_numbers=False),
            title="[bold yellow]Quick Start[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
        )
    )

    # ── Tip ─────────────────────────────────────────────────────────────
    console.print(
        "\n[dim]Tip: Run any command with [bold]--help[/bold] for detailed usage, "
        "e.g. [bold green]tau2 run --help[/bold green][/dim]\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Tau2 command line interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run a benchmark")
    add_run_args(run_parser)

    def run_command(args):
        # Set global LLM log mode (used by verbose logging)
        from tau2.utils.llm_utils import set_llm_log_mode

        set_llm_log_mode(args.llm_log_mode)

        config = TextRunConfig(
            domain=args.domain,
            task_set_name=args.task_set_name,
            task_split_name=args.task_split_name,
            task_ids=args.task_ids,
            num_tasks=args.num_tasks,
            llm_user=args.user_llm,
            llm_args_user=args.user_llm_args,
            num_trials=args.num_trials,
            max_errors=args.max_errors,
            timeout=args.timeout,
            save_to=args.save_to,
            max_concurrency=args.max_concurrency,
            seed=args.seed,
            log_level=args.log_level,
            verbose_logs=args.verbose_logs,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            auto_resume=args.auto_resume,
            agent=args.agent,
            llm_agent=args.agent_llm,
            llm_args_agent=args.agent_llm_args,
            user=args.user,
            max_steps=args.max_steps,
            enforce_communication_protocol=args.enforce_communication_protocol,
        )

        return run_domain(config)

    run_parser.set_defaults(func=run_command)

    # View command
    view_parser = subparsers.add_parser("view", help="View simulation results")
    view_parser.add_argument(
        "--dir",
        type=str,
        help="Directory containing simulation files. Defaults to data/simulations if not specified.",
    )
    view_parser.add_argument(
        "--file",
        type=str,
        help="Path to the simulation results file to view",
    )
    view_parser.add_argument(
        "--only-show-failed",
        action="store_true",
        help="Only show failed tasks.",
    )
    view_parser.add_argument(
        "--only-show-all-failed",
        action="store_true",
        help="Only show tasks that failed in all trials.",
    )
    view_parser.set_defaults(func=lambda args: run_view_simulations(args))

    # Domain command
    domain_parser = subparsers.add_parser("domain", help="Show domain documentation")
    domain_parser.add_argument(
        "domain",
        type=str,
        help="Name of the domain to show documentation for (e.g., 'indian_banking')",
    )
    domain_parser.set_defaults(func=lambda args: run_show_domain(args))

    # Intro command
    intro_parser = subparsers.add_parser(
        "intro", help="Show an overview of the benchmark and available commands"
    )
    intro_parser.set_defaults(func=lambda args: run_intro())

    # Check data command
    check_data_parser = subparsers.add_parser(
        "check-data", help="Check if data directory is properly configured"
    )
    check_data_parser.set_defaults(func=lambda args: run_check_data())

    # Evaluate trajectories command
    evaluate_parser = subparsers.add_parser(
        "evaluate-trajs", help="Evaluate trajectories and update rewards"
    )
    evaluate_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    evaluate_parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save updated trajectory files with recomputed rewards. If not provided, only displays metrics.",
    )
    evaluate_parser.set_defaults(func=lambda args: run_evaluate_trajectories(args))

    # Convert results format command
    convert_parser = subparsers.add_parser(
        "convert-results",
        help="Convert simulation results between storage formats",
    )
    convert_parser.add_argument(
        "path",
        help="Path to results.json or results directory to convert",
    )
    convert_parser.add_argument(
        "--to",
        dest="target_format",
        choices=["json", "dir"],
        default=None,
        help="Target format: 'json' (monolithic) or 'dir' (directory with individual sim files). "
        "If omitted, converts to the opposite of the current format.",
    )
    convert_parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating a .bak backup of the original file",
    )
    convert_parser.set_defaults(func=lambda args: run_convert_results(args))

    args = parser.parse_args()
    if not hasattr(args, "func"):
        run_intro()
        return

    args.func(args)


def run_view_simulations(args):
    from tau2.scripts.view_simulations import main as view_main

    view_main(
        sim_file=args.file,
        only_show_failed=args.only_show_failed,
        only_show_all_failed=args.only_show_all_failed,
        sim_dir=args.dir,
    )


def run_show_domain(args):
    from tau2.scripts.show_domain_doc import main as domain_main

    domain_main(args.domain)


def run_check_data():
    from tau2.scripts.check_data import main as check_data_main

    check_data_main()


def run_evaluate_trajectories(args):
    import sys

    from loguru import logger

    from tau2.scripts.evaluate_trajectories import evaluate_trajectories

    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])

    evaluate_trajectories(args.paths, args.output_dir)


def run_convert_results(args):
    """Convert simulation results between storage formats."""
    import shutil
    from pathlib import Path

    from tau2.data_model.simulation import Results

    path = Path(args.path)
    current_fmt = Results._detect_format(path)
    target_fmt = args.target_format

    if target_fmt is None:
        target_fmt = "json" if current_fmt == "dir" else "dir"

    if current_fmt == target_fmt:
        print(f"Results at {path} are already in '{target_fmt}' format.")
        return

    print(f"Converting {path}: '{current_fmt}' -> '{target_fmt}'")
    results = Results.load(path)

    meta_path = path if path.suffix == ".json" else path / "results.json"

    if not args.no_backup:
        if current_fmt == "json":
            backup = meta_path.with_suffix(".json.bak")
            shutil.copy2(meta_path, backup)
            print(f"  Backup: {backup}")
        else:
            sims_dir = meta_path.parent / "simulations"
            backup_dir = meta_path.parent / "simulations.bak"
            if sims_dir.exists():
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                shutil.copytree(sims_dir, backup_dir)
            backup = meta_path.with_suffix(".json.bak")
            shutil.copy2(meta_path, backup)
            print(f"  Backup: {backup}")
            if backup_dir.exists():
                print(f"  Backup: {backup_dir}")

    if target_fmt == "dir":
        results.save(meta_path, format="dir")
    else:
        sims_dir = meta_path.parent / "simulations"
        results.save(meta_path, format="json")
        if sims_dir.exists():
            shutil.rmtree(sims_dir)

    n = len(results.simulations)
    print(f"  Done. {n} simulation(s) converted to '{target_fmt}' format.")


if __name__ == "__main__":
    main()
