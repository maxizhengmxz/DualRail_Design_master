"""Explicit staged CLI. No arguments prints help and launches no simulation."""

import argparse
import json
from pathlib import Path

from coupler_turn_off.config import OBJECTIVES, fingerprint, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("check", "nominal", "optimize", "audit", "sweep", "plot"),
        nargs="?",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--candidate", help="Explicit geometry id from the nominal search"
    )
    parser.add_argument(
        "--root",
        type=int,
        help="Root index within the selected candidate; inspect nominal.json first",
    )
    parser.add_argument(
        "--objective",
        choices=OBJECTIVES,
        help="Required for optimize/audit/sweep/plot; never inherited",
    )
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "output"
    )
    parser.add_argument("--maxiter", type=int, default=40)
    args = parser.parse_args()
    if args.stage is None:
        parser.print_help()
        return
    if args.config is None:
        parser.error(
            "--config is required; request the readout-optimized pair and direct coupling first"
        )
    if args.stage in ("optimize", "audit", "sweep", "plot") and (
        args.objective is None or args.candidate is None or args.root is None
    ):
        parser.error(
            "Supply --candidate, --root and --objective after asking the user which minimization goal to use"
        )
    if args.root is not None and args.root < 0:
        parser.error("--root must be nonnegative")
    if args.maxiter < 1:
        parser.error("--maxiter must be positive")
    try:
        config = load(args.config)
        if args.stage == "check":
            print(
                "Inputs complete. Next: nominal; then ask the user to choose physical, logical or combined."
            )
            return
        # Heavy numerical imports happen only after input validation.
        from threadpoolctl import threadpool_limits

        from coupler_turn_off.engine import Case, save

        folder = args.output / config["case_id"] / fingerprint(config)
        save(folder / "inputs.json", config)
        with threadpool_limits(limits=1):
            if args.stage == "nominal":
                records = []
                for candidate in config["coupler_candidates"]:
                    try:
                        roots = Case(config, candidate).nominal_roots()
                        records.append(
                            dict(candidate=candidate, roots=roots, error=None)
                        )
                    except ValueError as exc:
                        records.append(
                            dict(candidate=candidate, roots=[], error=str(exc))
                        )
                    save(folder / "nominal.json", records)
                print(json.dumps(records, indent=2))
                return
            candidate = next(
                (c for c in config["coupler_candidates"] if c["id"] == args.candidate),
                None,
            )
            if candidate is None:
                raise ValueError("Unknown candidate id")
            case = Case(config, candidate)
            selected = folder / args.candidate / f"root-{args.root}" / args.objective
            if args.stage == "optimize":
                if (selected / "optimized.json").exists():
                    raise ValueError(
                        "This optimization already exists; inspect it or choose a new output directory"
                    )
                records = json.loads((folder / "nominal.json").read_text())
                root = next(
                    r for r in records if r["candidate"]["id"] == args.candidate
                )["roots"][args.root]
                record = case.optimize(
                    root["ej_ghz"], args.objective, selected, args.maxiter
                )
                print(
                    json.dumps(
                        dict(
                            output=str(selected),
                            success=record["optimizer_success"],
                            constraints_pass=record["nominal_constraints_pass"],
                        )
                    )
                )
            else:
                record = json.loads((selected / "optimized.json").read_text())
                if args.stage == "audit":
                    save(selected / "audit.json", case.audit(record["ej_ghz"]))
                else:
                    from coupler_turn_off.sweep import plot, sweep

                    if args.stage == "sweep":
                        sweep(case, record["ej_ghz"], selected)
                    else:
                        plot(case, selected)
                print(selected)
    except (ValueError, KeyError, IndexError, FileNotFoundError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
