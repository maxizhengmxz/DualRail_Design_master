"""Run focused coupler tests without saved inputs, outputs or optimization jobs."""

import sys
import unittest
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER))
sys.path.insert(1, str(FOLDER.parent))

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(FOLDER / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
