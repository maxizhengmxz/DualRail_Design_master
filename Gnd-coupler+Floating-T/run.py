"""CLI entry point; no numerical run without an explicit command/config."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from floating_coupler import build_model, scan_coupler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    for command in ("check", "scan"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--subset", action="store_true", help="Keep loaded A2-C-B1 coefficients")
        sub.add_argument("--output", type=Path, help="Optional JSON output (use output/)")
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model = build_model(config)
    if args.subset:
        model = model.subset()
    result = {"input": config, "electrostatics": model.electrostatics,
              "dynamical_node_order": list(model.names)}
    if args.command == "scan":
        scan = config["scan"]
        result["scan"] = scan_coupler(model, scan["ej_start_ghz"], scan["ej_stop_ghz"],
                                      scan["points"], scan.get("minimum_overlap", 0.8))
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
        print(args.output)
    else:
        print(encoded)


if __name__ == "__main__":
    main()
