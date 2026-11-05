"""Explicit entry point for the independent XM CZ workflow; no default run."""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    for command in ("validate", "run", "plot"):
        sub = commands.add_parser(command)
        sub.add_argument("config", type=Path)
        if command == "run":
            sub.add_argument("--dry-run", action="store_true")
            sub.add_argument("--workers", type=int)
            sub.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0
    from core.runtime import configure_single_threaded_blas

    configure_single_threaded_blas()
    from xm_cz.config import validate_payload
    from xm_cz.runner import make_tasks, run

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    tasks = make_tasks(config)
    for task in tasks:
        validate_payload(task["payload"])
    if args.command == "validate":
        print(f"Validated {len(tasks)} grid points; no calculation launched")
        return 0
    if args.command == "plot":
        from xm_cz.plotting import plot_stage

        print(json.dumps(plot_stage(args.config), indent=2))
        return 0
    code = run(
        args.config,
        workers=args.workers,
        retry_failed=args.retry_failed,
        dry_run=args.dry_run,
    )
    if not args.dry_run and code == 0 and "grid" in config:
        from xm_cz.plotting import plot_stage

        print(json.dumps(plot_stage(args.config), indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
