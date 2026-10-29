"""Explicit approximate CZ commands; running without arguments shows help."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    select = commands.add_parser("select-channel", help="select minimum S1 among verified common-g candidates; does not compute spectra")
    select.add_argument("config")
    select.add_argument("--output", required=True)
    single = commands.add_parser("simulate", help="detailed single-point calculation")
    single.add_argument("config", nargs="?", default=str(HERE / "configs/example_config.json"))
    search = commands.add_parser("search", help="constrained EJ candidate search")
    search.add_argument("config", nargs="?", default=str(HERE / "configs/gate_search.json"))
    search.add_argument("--output", default=str(HERE / "output/gate_points.nc"))
    sweep = commands.add_parser("sweep", help="one explicitly selected sweep")
    sweep.add_argument("kind", choices=("parameter", "ec_a2", "detuning", "ej_detuning", "signed_detuning", "coupling"))
    sweep.add_argument("config", nargs="?", default=str(HERE / "configs/parameter_sweep_config.json"))
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    if args.command == "select-channel":
        from transmon_cz.channel_policy import select_channel, off_target_definition
        raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
        chosen = select_channel(raw["candidates"], **raw["selection_conditions"])
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"channel_selection": chosen.to_dict(),
                          "coupler_handoff_off_target":off_target_definition('five')}, indent=2), encoding="utf-8")
        print(json.dumps(chosen.to_dict(), indent=2))
        return
    from core.runtime import configure_single_threaded_blas
    configure_single_threaded_blas()
    if args.command == "simulate":
        from transmon_cz.experiments.run_simulation import main as simulate
        sys.argv = [sys.argv[0], args.config]
        simulate()
    elif args.command == "search":
        import numpy as np
        from transmon_cz.gate_search import find_gate_operating_points, save_gate_operating_points
        raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
        grid = raw["ej_grid_GHz"]
        result = find_gate_operating_points(
            raw, np.linspace(grid["start"], grid["stop"], grid["count"]),
            evolution=raw["evolution"], **raw["gate_search"],
        )
        save_gate_operating_points(result, args.output)
        print(f"Saved {result.sizes['gate_operating_point']} candidates to {args.output}")
    else:
        from transmon_cz.experiments import sweeps
        name = {"parameter": "run_parameter_sweep", "ec_a2": "run_ec_a2_sweep",
                "detuning": "run_detuning_sweep", "ej_detuning": "run_ej_a2_detuning_sweep",
                "signed_detuning": "run_ej_a2_detuning_sweep", "coupling": "run_coupling_sweep"}[args.kind]
        kwargs = {"config_key": "ej_a2_signed_detuning_sweep"} if args.kind == "signed_detuning" else {}
        getattr(sweeps, name)(args.config, **kwargs)


if __name__ == "__main__":
    main()
