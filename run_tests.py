"""Run active workflow tests from this checkout; --legacy includes archived tests.

Entry-point path setup makes the workflow-local logical_resonance_gate package
importable without core forwarding imports or an additional package install.
"""

import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parent
for source in (ROOT, ROOT / "logical-resonance gate", ROOT / "coupler turn-off", ROOT / "transmon-based CZ"):
    sys.path.insert(0, str(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true", help="also run archived tests")
    args = parser.parse_args()
    directories = [ROOT / "tests", ROOT / "readout_workflow" / "tests",
                   ROOT / "coupler turn-off" / "tests", ROOT / "transmon-based CZ" / "tests"]
    if args.legacy:
        directories.append(ROOT / "legacy" / "tests")
    suite = unittest.TestSuite(
        unittest.TestLoader().discover(str(directory)) for directory in directories
    )
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
