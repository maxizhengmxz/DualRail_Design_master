"""Run active workflow tests from this checkout; --legacy includes archived tests.

Entry-point path setup makes the workflow-local logical_resonance_gate package
importable without core forwarding imports or an additional package install.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEGACY_CZ = ROOT / "legacy" / "transmon-based CZ"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true", help="also run archived tests")
    args = parser.parse_args()
    directories = [ROOT / "tests", ROOT / "readout_workflow" / "tests",
                   ROOT / "coupler turn-off" / "tests", ROOT / "2Q-gate_XM" / "tests",
                   ROOT / "Gnd-coupler+Floating-T" / "tests"]
    if args.legacy:
        directories += [LEGACY_CZ / "tests", ROOT / "legacy" / "tests"]
    # Separate processes prevent duplicate test-module names across workflows
    # from sharing imports, discovery state or numerical runtime settings.
    failed = False
    for directory in directories:
        env = os.environ.copy()
        paths = [ROOT, ROOT / "logical-resonance gate", ROOT / "coupler turn-off",
                 ROOT / "2Q-gate_XM", ROOT / "Gnd-coupler+Floating-T"]
        if directory.parent.name in ("legacy", "transmon-based CZ"):
            paths.append(LEGACY_CZ)
        env["PYTHONPATH"] = os.pathsep.join(map(str, paths))
        print(f"\nTesting {directory.relative_to(ROOT)}", flush=True)
        result = subprocess.run([sys.executable, "-m", "unittest", "discover",
                                 "-s", str(directory)], cwd=ROOT, env=env)
        failed |= result.returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
