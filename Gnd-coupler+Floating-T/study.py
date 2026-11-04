"""Explicit optional-study commands. Importing never launches a study."""

from pathlib import Path
import argparse
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    robust = commands.add_parser(
        "robustness", help="Explicit capacitor-variation recovery study"
    )
    robust.add_argument(
        "--input", required=True, help="Self-contained nominal fabrication-input.json"
    )
    robust.add_argument("--output", required=True)
    robust.add_argument(
        "--sampling", choices=["independent", "max-bins"], default="independent"
    )
    robust.add_argument("--deviation-percent", type=float, default=5.0)
    robust.add_argument("--bin-width-percent", type=float, default=0.25)
    robust.add_argument("--target-per-bin", type=int, default=200)
    robust.add_argument("--samples", type=int, default=200)
    robust.add_argument(
        "--reuse", help="Optional robustness-data.tar.xz to retain and credit"
    )
    robust.add_argument("--workers", type=int, choices=[1, 2, 3, 4], default=2)
    robust.add_argument("--hours", type=float, default=48.0)
    robust.add_argument("--seed", type=int, default=20261103)
    robust.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate input and show sample plan only",
    )
    deco = commands.add_parser(
        "decoherence", help="Explicit independent-rail flux-noise study"
    )
    deco.add_argument("--input", required=True)
    deco.add_argument("--output", required=True)
    deco.add_argument(
        "--a-rail-tuning",
        type=float,
        nargs=2,
        default=[4.0, 5.0],
        metavar=("MIN", "MAX"),
        help="A1/A2 SQUID range for isolated A-idle dephasing and the central AB-noise gate estimate",
    )
    deco.add_argument(
        "--rail-tuning", type=float, nargs=2, default=[4.4, 5.8], metavar=("MIN", "MAX")
    )
    deco.add_argument(
        "--coupler-tuning",
        type=float,
        nargs=2,
        default=[6.5, 11.0],
        metavar=("MIN", "MAX"),
    )
    deco.add_argument(
        "--amplitudes",
        type=float,
        nargs=2,
        default=[1.0, 100.0],
        metavar=("MIN", "MAX"),
    )
    deco.add_argument("--low-cutoff", type=float, default=1.0)
    deco.add_argument("--gate-time", type=float, default=100.0)
    deco.add_argument("--levels", type=int, default=7)
    deco.add_argument("--response-points", type=int, default=41)
    deco.add_argument(
        "--participation",
        action="store_true",
        help="Also evaluate five-mode physical-basis coupler participation at OFF",
    )
    report = commands.add_parser(
        "report", help="Render saved results without optimization"
    )
    report.add_argument("--kind", choices=["robustness", "decoherence"], required=True)
    report.add_argument(
        "--study", required=True, help="Study folder or retained robustness .tar.xz"
    )
    report.add_argument("--output", help="Required when rendering an archive")
    report.add_argument("--design-report", help="Existing design PDF whose hardware cover should be refreshed (decoherence reporting only)")
    archive = commands.add_parser(
        "verify-archive", help="Verify archive checksums and outcome counts"
    )
    archive.add_argument("archive")
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "robustness":
        if (
            args.hours <= 0
            or args.samples < 1
            or args.target_per_bin < 1
            or not 0 < args.bin_width_percent <= args.deviation_percent <= 5
        ):
            raise ValueError(
                "Require positive budget/counts and 0<bin width<=deviation<=5%"
            )
        if args.dry_run:
            from floating_coupler.fabrication import parameter_names

            base = json.loads(Path(args.input).read_text())
            return dict(
                command="robustness",
                launched=False,
                parameters=parameter_names(base),
                sampling=args.sampling,
                deviation_percent=args.deviation_percent,
                workers=args.workers,
            )
        from studies.robustness import run

        return run(args)
    if args.command == "decoherence":
        if (
            not 0 < args.amplitudes[0] <= args.amplitudes[1]
            or args.low_cutoff <= 0
            or args.gate_time <= 0
            or args.levels < 4
            or args.response_points < 3
        ):
            raise ValueError("Invalid noise, gate time or basis inputs")
        from studies.decoherence import run

        return run(args)
    if args.command == "verify-archive":
        from studies.archive import verify_archive

        return verify_archive(args.archive)
    if args.kind == "decoherence":
        from studies.decoherence_report import generate

        return generate(args.study, design_report=args.design_report)
    from studies.robustness_report import generate

    study = Path(args.study)
    if study.is_file():
        if not args.output:
            raise ValueError("--output is required to report an archive")
        folder = Path(args.output)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "campaign.json").write_text(
            json.dumps(
                dict(study="robustness", reuse_archive=str(study.resolve())), indent=2
            )
        )
        (folder / "status.json").write_text(
            json.dumps(dict(state="completed", calculation_errors=0), indent=2)
        )
        return generate(folder)
    return generate(study)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
