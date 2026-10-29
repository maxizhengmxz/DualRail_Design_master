"""Import channel definitions from the toy workflow; define no local convention."""
import sys
from pathlib import Path
TOY_ROOT=Path(__file__).resolve().parents[2]/'transmon-based CZ'
if str(TOY_ROOT) not in sys.path:sys.path.insert(0,str(TOY_ROOT))
from transmon_cz.channel_policy import (DEFINITION_VERSION,LOGICAL_ORDER,PAIR_ORDER,
    AUXILIARY_OCCUPATIONS,SELECTION_RULE,SCORE_KIND,Crossing,S1Candidate,ChannelSelection,
    one_excitation_indices,require_selection,selection_from_request,
    select_channel,channel_indices,require_same_channel,interaction_scores,
    OFF_DEFINITION_VERSION,OFF_TARGET,THREE_BODY_OFF_TARGET,off_target_definition,require_off_definition,
    logical_energy_frame,logical_exchange_from_eigenpairs,effective_logical_exchange,
    logical_exchange_definition,physical_exchange_from_eigenpairs)
