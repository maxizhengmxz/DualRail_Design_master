"""Shared stable definitions only; no original simulation/search imports."""

import sys
from pathlib import Path

TOY_ROOT = Path(__file__).resolve().parents[2] / "transmon-based CZ"
if str(TOY_ROOT) not in sys.path:
    sys.path.insert(0, str(TOY_ROOT))

from transmon_cz.channel_policy import (  # noqa: E402,F401
    DEFINITION_VERSION,
    LOGICAL_ORDER,
    SELECTION_RULE,
    S1Candidate,
    effective_logical_exchange,
    interaction_scores,
    off_target_definition,
    one_excitation_indices,
    select_channel,
)
