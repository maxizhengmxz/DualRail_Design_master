"""Shared logical definitions, S1/S2 scoring and fixed-strength CZ selection.

Spectra and feasible resonance candidates are supplied by the physics workflow.
Crossing order is a path diagnostic, never the selection objective.
"""
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

DEFINITION_VERSION = 'dual-rail-cz-s1-v2'
LOGICAL_ORDER = ('00', '01', '10', '11')
PAIR_ORDER = ('A', 'B')
AUXILIARY_OCCUPATIONS = {'A20': '2000', 'A11': '1100', 'A02': '0200',
                         'B20': '0020', 'B11': '0011', 'B02': '0002'}
SELECTION_RULE = 'minimum_S1_at_common_intended_g'
SCORE_KIND = 'evaluated_at_common_target_g'
OFF_DEFINITION_VERSION = 'physical-single-excitation-off-v1'
OFF_TARGET = 'physical_0100_0010_exchange_zero'
THREE_BODY_OFF_TARGET = 'physical_01_10_exchange_zero'


def off_target_definition(model='five'):
    """OFF isolation is independent of the S1-selected ON CZ auxiliary."""
    if model not in ('three','five','seven'):
        raise ValueError('OFF model must be three, five or seven')
    return dict(version=OFF_DEFINITION_VERSION,model=model,
                target=THREE_BODY_OFF_TARGET if model=='three' else OFF_TARGET,
                source='01' if model=='three' else '0100',
                destination='10' if model=='three' else '0010',
                status='established_seed' if model=='three' else 'temporary_open_question',
                controls='fixed_idle_rails_and_capacitors;vary_coupler_EJ',
                frame=('physical_A2_B1_with_C_in_ground' if model=='three'
                       else 'canonical_four_physical_single_excitation_states;C_ground;resonators_vacuum'),
                auxiliary_policy='virtual_paths_included;logical_exchange_CZ_channels_and_ZZ_are_residual_diagnostics')


def require_off_definition(value,model='five'):
    if value!=off_target_definition(model):
        raise ValueError('Superseded or missing OFF definition: temporary physical 0100-0010 target in five/seven modes; physical 01-10 in three modes. Recompute, do not relabel old results.')
    return value


def logical_exchange_definition(model='five'):
    """Preserve the previous observable as a diagnostic, not an OFF proof.

    TEMPORARY-OFF: restore this observable as the optimization goal only after
    the user revisits the choice documented in Gnd-coupler+Floating-T/OFF_POLICY.md.
    """
    return dict(version='logical-exchange-diagnostic-v1',model=model,
                target='logical_01_10_exchange',role='residual_diagnostic',
                frame='canonical_four_logical_states_then_local_energy_axes')


def physical_exchange_from_eigenpairs(energies,projection,model='five',minimum_overlap=.9):
    """Full-state downfolding onto 1000,0100,0010,0001, without local rotations.

    All four physical references must be retained together: a two-reference
    elimination would incorrectly eliminate the intentionally hybridized rails.
    Coupler and resonators are in ground/vacuum reference states, but their
    virtual excitations remain in the supplied full dressed eigenstates.
    """
    from scipy.optimize import linear_sum_assignment
    energies=np.asarray(energies);projection=np.asarray(projection)
    if model not in ('five','seven') or projection.shape!=(4,len(energies)):
        raise ValueError('Supply four physical single-excitation projections for five/seven modes')
    _,columns=linear_sum_assignment(-abs(projection)**2)
    u,s,vh=np.linalg.svd(projection[:,columns]);rotation=u@vh
    effective=(rotation*energies[columns])@rotation.conj().T
    overlap=float(s.min()**2)
    return dict(off_definition=off_target_definition(model),g_mhz=float(effective[1,2].real*1000),
                detuning_mhz=float((effective[1,1]-effective[2,2]).real*1000),
                reference_order=['1000','0100','0010','0001'],
                effective_hamiltonian_ghz=effective.real.tolist(),
                minimum_subspace_overlap=overlap,valid=bool(overlap>=minimum_overlap),
                selected_eigenstate_indices=columns.tolist())


def logical_energy_frame(logical_ghz):
    """Local single-rail rotations only; never diagonalize joint logical exchange."""
    from itertools import product
    from scipy.linalg import eigh
    logical_ghz=np.asarray(logical_ghz)
    paulis={'I':np.eye(2),'X':np.array([[0,1],[1,0]]),
            'Y':np.array([[0,-1j],[1j,0]]),'Z':np.diag([1,-1])}
    coefficients={a+b:float(np.trace(np.kron(paulis[a],paulis[b])@logical_ghz).real/4*1000)
                  for a,b in product(paulis,repeat=2)}
    ha=sum(coefficients[a+'I']*paulis[a] for a in 'XYZ')
    hb=sum(coefficients['I'+a]*paulis[a] for a in 'XYZ')
    ea,ua=eigh(ha);eb,ub=eigh(hb)
    for rotation in (ua,ub):
        for column in range(2):
            pivot=np.argmax(abs(rotation[:,column]))
            rotation[:,column]*=1 if rotation[pivot,column].real>=0 else -1
    transform=np.kron(ua,ub)
    rotated=(transform.conj().T@logical_ghz@transform).real*1000
    rotated_coefficients={a+b:float(np.trace(np.kron(paulis[a],paulis[b])@rotated).real/4)
                          for a,b in product(paulis,repeat=2)}
    return dict(nominal_frame_pauli_mhz=coefficients,energy_frame_pauli_mhz=rotated_coefficients,
                exchange_mhz=float(rotated[1,2]),double_flip_mhz=float(rotated[0,3]),
                zz_mhz=float(rotated[3,3]-rotated[2,2]-rotated[1,1]+rotated[0,0]),
                local_splittings_mhz=[float(ea[1]-ea[0]),float(eb[1]-eb[0])],
                local_rotations=[ua.real.tolist(),ub.real.tolist()])


def logical_exchange_from_eigenpairs(energies,logical_projection,minimum_overlap=.9):
    """Project full dressed eigenstates directly onto the four logical references.

    Symmetric orthogonalization includes virtual coupler/auxiliary paths before
    local-axis alignment. It does not replace the Hamiltonian by its eigenbasis,
    which would trivially erase every off-diagonal matrix element.
    """
    from scipy.optimize import linear_sum_assignment
    energies=np.asarray(energies);projection=np.asarray(logical_projection)
    if energies.ndim!=1 or projection.shape!=(4,len(energies)):
        raise ValueError('Supply four logical-reference projections in 00,01,10,11 order')
    _,columns=linear_sum_assignment(-abs(projection)**2)
    selected=projection[:,columns]
    u,s,vh=np.linalg.svd(selected);rotation=u@vh
    effective=(rotation*energies[columns])@rotation.conj().T
    frame=logical_energy_frame(effective)
    ua,ub=map(np.asarray,frame['local_rotations']);transform=np.kron(ua,ub)
    rotated=(transform.T@effective@transform).real
    overlap=float(s.min()**2)
    return dict(observable_definition=logical_exchange_definition('five'),g_mhz=float(rotated[1,2]*1000),
                detuning_mhz=float((rotated[1,1]-rotated[2,2])*1000),
                logical_reference_order=list(LOGICAL_ORDER),effective_hamiltonian_ghz=effective.real.tolist(),
                local_energy_hamiltonian_ghz=rotated.tolist(),local_energy_frame=frame,
                minimum_subspace_overlap=overlap,valid=bool(overlap>=minimum_overlap),
                assignment_weights=[float(abs(projection[i,k])**2) for i,k in enumerate(columns)],
                selected_eigenstate_indices=columns.tolist(),
                scope='Direct full-eigenstate four-logical-state effective exchange; virtual auxiliaries included')


def effective_logical_exchange(hamiltonian,logical_indices,minimum_overlap=.9):
    from scipy.linalg import eigh
    energies,vectors=eigh(np.asarray(hamiltonian))
    return logical_exchange_from_eigenpairs(energies,vectors[list(logical_indices),:],minimum_overlap)


def one_excitation_indices(energies, weights):
    """Logical 0/1 are the lower/upper two one-excitation-like modes."""
    energies, weights = np.asarray(energies), np.asarray(weights)
    if energies.ndim != 1 or weights.shape != energies.shape or len(energies) < 2:
        raise ValueError('Supply matching energy and one-excitation-weight vectors')
    if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(weights)):
        raise ValueError('Non-finite pair-state data')
    candidates = np.argsort(weights)[-2:]
    return tuple(sorted(map(int, candidates), key=lambda i: energies[i]))


def interaction_scores(energies, interaction, logical_indices, intended_source,
                       auxiliary, *, active_mask=None):
    """Toy S1 = mean logical-origin sum of unintended |g/delta| squared.

    The intended resonant edge is excluded. S2 scores auxiliary-origin edges
    separately and never participates in ranking. Caller supplies the stated
    reference basis and numerical nonzero mask; no physics truncation is hidden.
    Energies and interaction coefficients must use the same cyclic units.
    """
    energies = np.asarray(energies, dtype=float)
    interaction = np.asarray(interaction)
    logical = list(map(int, logical_indices))
    n = len(energies)
    if interaction.shape != (n, n) or len(logical) != 4 or len(set(logical)) != 4:
        raise ValueError('Supply four distinct logical indices and a square interaction')
    if intended_source not in logical or auxiliary in logical:
        raise ValueError('Specify a logical source and a distinct nonlogical auxiliary')
    if any(i < 0 or i >= n for i in logical + [auxiliary]):
        raise ValueError('Reference index out of range')
    if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(interaction)):
        raise ValueError('Non-finite scoring Hamiltonian')
    active = abs(interaction) > 0 if active_mask is None else np.asarray(active_mask, dtype=bool)
    if active.shape != interaction.shape:
        raise ValueError('Invalid numerical nonzero mask')
    scores, maxima, parts = [], [], []
    for source in logical + [auxiliary]:
        mask = active[source].copy()
        mask[source] = False
        if source == intended_source:
            mask[auxiliary] = False
        if source == auxiliary:
            mask[intended_source] = False
        destinations = np.flatnonzero(mask)
        delta = abs(energies[source] - energies[mask])
        if np.any(delta <= 1e-12):
            if source != auxiliary:
                raise ValueError('logical_unintended_degeneracy')
            scores.append(None)
            maxima.append(None)
            parts.append({'logical': None, 'nonlogical': None, 'auxiliary_degeneracy': True})
            continue
        ratios = abs(interaction[source, mask]) / delta
        squared = ratios**2
        is_logical = np.isin(destinations, logical)
        scores.append(float(squared.sum()))
        maxima.append(float(max(ratios, default=0)))
        parts.append({'logical': float(squared[is_logical].sum()),
                      'nonlogical': float(squared[~is_logical].sum())})
    return {'S1': float(np.mean(scores[:4])), 'S2': scores[4],
            'S1_by_input': scores[:4], 'score_parts': parts,
            'maximum_logical_ratio': max(maxima[:4]),
            'maximum_auxiliary_ratio': maxima[4]}


@dataclass(frozen=True)
class Crossing:
    """Path diagnostic only; cannot be passed off as an S1 candidate."""
    source_logical: str
    auxiliary: str
    path_fraction: float
    moved_frequency_ghz: float
    usable: bool = True
    relevant: bool = True
    tracking_verified: bool = False


@dataclass(frozen=True)
class S1Candidate:
    source_logical: str
    auxiliary: str
    moved_frequency_ghz: float
    s1: float
    intended_g_mhz: float
    maximum_logical_ratio: float
    score_basis_id: str
    comparison_id: str
    stationary_frequency_ghz: float | None = None
    feasible: bool = True
    tracking_verified: bool = False
    score_kind: str = SCORE_KIND


@dataclass(frozen=True)
class ChannelSelection:
    source_logical: str
    auxiliary: str
    moved_pair: str
    gate_frequency_ghz: float
    s1: float
    intended_g_mhz: float
    target_g_mhz: float
    maximum_logical_ratio: float
    score_basis_id: str
    comparison_id: str
    idle_frequency_ghz: float | None = None
    stationary_frequency_ghz: float | None = None
    logical_ratio_limit: float = .2
    definition_version: str = DEFINITION_VERSION
    selection_rule: str = SELECTION_RULE
    score_kind: str = SCORE_KIND

    def __post_init__(self):
        if self.source_logical not in LOGICAL_ORDER or self.auxiliary not in AUXILIARY_OCCUPATIONS:
            raise ValueError('Explicit logical source and physical auxiliary are required')
        if self.moved_pair not in PAIR_ORDER:
            raise ValueError('Specify the moved pair A or B')
        if self.definition_version != DEFINITION_VERSION or self.selection_rule != SELECTION_RULE:
            raise ValueError('Superseded channel policy: rerank verified candidates by S1 at common intended g')
        if self.score_kind != SCORE_KIND:
            raise ValueError('A normalized S1 estimate is diagnostic, not a verified common-g selection')
        frequencies = [self.gate_frequency_ghz]
        frequencies += [x for x in (self.idle_frequency_ghz, self.stationary_frequency_ghz) if x is not None]
        if not all(np.isfinite(x) and x > 0 for x in frequencies):
            raise ValueError('Frequencies must be positive and finite')
        if not all(np.isfinite(x) and x >= 0 for x in (self.s1, self.maximum_logical_ratio)):
            raise ValueError('S1 and unintended mixing ratio must be finite and nonnegative')
        if not (np.isfinite(self.target_g_mhz) and self.target_g_mhz > 0
                and np.isfinite(self.intended_g_mhz) and self.intended_g_mhz > 0):
            raise ValueError('Supply positive intended and comparison coupling strengths')
        if not np.isclose(self.intended_g_mhz, self.target_g_mhz, rtol=1e-5, atol=1e-8):
            raise ValueError('Candidates must be evaluated at the common target g; do not use unequal-g raw S1')
        if not np.isfinite(self.logical_ratio_limit) or self.logical_ratio_limit <= 0:
            raise ValueError('Invalid logical-origin mixing limit')
        if self.maximum_logical_ratio >= self.logical_ratio_limit:
            raise ValueError('Candidate violates the logical-origin |g/delta| screen')
        if not self.score_basis_id or not self.comparison_id:
            raise ValueError('Record the score basis and common comparison conditions')

    def to_dict(self):
        return dict(asdict(self), physical_auxiliary=AUXILIARY_OCCUPATIONS[self.auxiliary],
                    logical_order=list(PAIR_ORDER))


def require_selection(value):
    if isinstance(value, ChannelSelection):
        return value
    if not isinstance(value, Mapping):
        raise ValueError('Missing S1-selected CZ channel from the toy workflow; auxiliary 0020 alone does not specify a source')
    if value.get('definition_version', DEFINITION_VERSION) != DEFINITION_VERSION:
        raise ValueError('Superseded channel policy: rerank by S1; old saved results are not relabeled')
    fields = ChannelSelection.__dataclass_fields__
    try:
        result = ChannelSelection(**{k:v for k,v in value.items() if k in fields})
    except TypeError as exc:
        raise ValueError('Incomplete verified S1 channel-selection record') from exc
    if value.get('logical_order', list(PAIR_ORDER)) != list(PAIR_ORDER):
        raise ValueError('Logical bit order must be A,B')
    if value.get('physical_auxiliary', AUXILIARY_OCCUPATIONS[result.auxiliary]) != AUXILIARY_OCCUPATIONS[result.auxiliary]:
        raise ValueError('Auxiliary occupation disagrees with branch label')
    return result


def selection_from_request(request):
    return require_selection(request.get('channel_selection') or request.get('gate_target', {}).get('channel_selection'))


def select_channel(candidates: Sequence[S1Candidate], *, moved_pair, auxiliary,
                   target_g_mhz, idle_frequency_ghz=None, logical_ratio_limit=.2):
    """Choose the single feasible minimum S1 at common g and comparison basis.

    No crossing-order or S2 tie-break. The caller performs spectral search,
    candidate feasibility and state tracking. Equal-score unresolved choices
    fail explicitly. Uniformly rescaled saved scores cannot certify a design.
    """
    events = []
    for value in candidates:
        if isinstance(value, S1Candidate):
            event = value
        elif isinstance(value, Mapping):
            try:
                event = S1Candidate(**value)
            except TypeError as exc:
                raise ValueError('Supply evaluated S1 candidates, not crossing-only records') from exc
        else:
            raise ValueError('Crossing order does not select a resonance; supply evaluated S1 candidates')
        if event.auxiliary != auxiliary or not event.feasible:
            continue
        if not event.tracking_verified:
            raise ValueError('Unverified candidate state identity')
        if np.isfinite(event.maximum_logical_ratio) and event.maximum_logical_ratio >= logical_ratio_limit:
            continue
        record = ChannelSelection(event.source_logical, event.auxiliary, moved_pair,
                                  event.moved_frequency_ghz, event.s1, event.intended_g_mhz,
                                  target_g_mhz, event.maximum_logical_ratio,
                                  event.score_basis_id, event.comparison_id,
                                  idle_frequency_ghz, event.stationary_frequency_ghz,
                                  logical_ratio_limit, score_kind=event.score_kind)
        events.append(record)
    if not events:
        raise ValueError('No feasible S1 candidate at the common target g')
    if len({(e.score_basis_id, e.comparison_id) for e in events}) != 1:
        raise ValueError('S1 candidates must share the same basis and comparison conditions')
    events.sort(key=lambda event: event.s1)
    if len(events) > 1 and np.isclose(events[0].s1, events[1].s1, rtol=1e-12, atol=1e-15) and events[0] != events[1]:
        raise ValueError('Unresolved equal-S1 candidates; do not substitute first-crossing selection')
    return events[0]


def channel_indices(reference_labels, selection):
    selected = require_selection(selection)
    try:
        return reference_labels.index(selected.source_logical), reference_labels.index(selected.auxiliary)
    except ValueError as exc:
        raise ValueError('Selected physical branches are absent from the reference space') from exc


def require_same_channel(on, off):
    a, b = map(require_selection, (on, off))
    if a != b:
        raise ValueError('ON design provenance must match; the OFF logical-exchange target is defined separately')
    return a
