#!/usr/bin/env python3
"""Wait for rail-A step-2 direct-EC completion and produce canonical plots."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from plot_direct_ec_readout_map import run

HERE = Path(__file__).resolve().parent
CAMPAIGN = HERE / "readout" / "direct_ec" / "rail_a_step2"
PLOTS = HERE / "readout" / "plots"


def main() -> None:
    while True:
        progress_path = CAMPAIGN / "progress.json"
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress["phase"] == "complete":
                break
        time.sleep(10)
    unique = PLOTS / "rail_a_independent_ec_readout_map_step2"
    summary = run(CAMPAIGN, unique, "Rail A readout: 140-160 MHz")
    canonical = PLOTS / "rail_a_independent_ec_readout_map"
    for suffix in (".png", ".pdf", ".json"):
        shutil.copy2(unique.with_suffix(suffix), canonical.with_suffix(suffix))
    (CAMPAIGN / "plot_complete.json").write_text(json.dumps({"status": "complete", "summary": summary}, indent=2) + "\n")


if __name__ == "__main__":
    main()
