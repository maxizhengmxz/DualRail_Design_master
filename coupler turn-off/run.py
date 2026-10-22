"""Run the coupler-off CLI; no arguments starts no calculation.

From the repository root: python "coupler turn-off/run.py" --help
The shared readout primitives remain in the sibling readout_workflow package.
"""

import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER))
sys.path.insert(1, str(FOLDER.parent))

from coupler_turn_off.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
