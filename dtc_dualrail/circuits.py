"""Circuit factory helpers."""

import inspect

import numpy as np
import scqubits as scq

DEFAULT_TRANSFORMATION_MATRIX = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
)
DEFAULT_SYSTEM_HIERARCHY = [[1, 2], [3]]
DEFAULT_SUBSYSTEM_TRUNC_DIMS = [40, 18]
FIVE_TRANSMON_FLUX_MAP = {
    "A1": "Φ1",
    "A2": "Φ2",
    "C": "Φ3",
    "B1": "Φ4",
    "B2": "Φ5",
}


def _circuit_factory_name(circuit_factory):
    return getattr(circuit_factory, "__name__", circuit_factory.__class__.__name__)


def _validate_circuit_factory_params(circuit_factory, poi_params):
    signature = inspect.signature(circuit_factory)
    parameters = signature.parameters
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    accepted_params = {
        name
        for name, param in parameters.items()
        if param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    required_params = {
        name
        for name, param in parameters.items()
        if param.default is inspect.Parameter.empty
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    for poi_idx, params in enumerate(poi_params):
        param_names = set(params)
        if not accepts_kwargs:
            unknown_params = sorted(param_names - accepted_params)
            if unknown_params:
                raise ValueError(
                    f"{_circuit_factory_name(circuit_factory)} cannot accept unknown "
                    f"parameters for poi {poi_idx}: {unknown_params}"
                )

        missing_params = sorted(required_params - param_names)
        if missing_params:
            raise ValueError(
                f"{_circuit_factory_name(circuit_factory)} is missing required "
                f"parameters for poi {poi_idx}: {missing_params}"
            )


def _configured_dual_rail_circuit(circuit_yaml: str) -> scq.Circuit:
    circuit = scq.Circuit(circuit_yaml, from_file=False, ext_basis="harmonic")
    circuit.cutoff_n_1 = 20
    circuit.cutoff_n_2 = 20
    circuit.cutoff_ext_3 = 20
    circuit.configure(
        transformation_matrix=DEFAULT_TRANSFORMATION_MATRIX,
        system_hierarchy=DEFAULT_SYSTEM_HIERARCHY,
        subsystem_trunc_dims=DEFAULT_SUBSYSTEM_TRUNC_DIMS,
    )
    return circuit


def _configured_five_transmon_circuit(circuit_yaml: str) -> scq.Circuit:
    circuit = scq.Circuit(circuit_yaml, from_file=False, ext_basis="harmonic")
    for index in range(1, 6):
        setattr(circuit, f"cutoff_n_{index}", 20)
    circuit.configure(
        transformation_matrix=np.eye(5),
        system_hierarchy=[[1], [2], [3], [4], [5]],
        subsystem_trunc_dims=[3, 3, 4, 3, 3],
    )
    object.__setattr__(circuit, "dualrail_flux_map", dict(FIVE_TRANSMON_FLUX_MAP))
    return circuit


def gen_five_transmon_dual_rail(
    *,
    rail_ej_pairs_ghz,
    rail_cj_pairs_ff,
    rail_charging_energy_ghz,
    coupler_ej_pair_ghz,
    coupler_cj_pair_ff,
    coupler_charging_energy_ghz,
    coupling_capacitance_ff,
    parasitic_capacitance_ff=None,
) -> scq.Circuit:
    """Build the direct A1--A2--C--B1--B2 reference circuit.

    The returned circuit records a stable logical-name to generated external-
    flux mapping in ``dualrail_flux_map``.  Coupling keys use ``A1_A2``,
    ``A2_C``, ``C_B1``, and ``B1_B2``; optional parasitics use any other pair
    from the same five labels, separated by an underscore.
    """

    rail_ej_pairs_ghz = tuple(tuple(values) for values in rail_ej_pairs_ghz)
    rail_cj_pairs_ff = tuple(tuple(values) for values in rail_cj_pairs_ff)
    rail_charging_energy_ghz = tuple(rail_charging_energy_ghz)
    if not (
        len(rail_ej_pairs_ghz)
        == len(rail_cj_pairs_ff)
        == len(rail_charging_energy_ghz)
        == 4
    ):
        raise ValueError("rail parameter collections must each contain four entries")
    if any(len(values) != 2 for values in (*rail_ej_pairs_ghz, *rail_cj_pairs_ff)):
        raise ValueError("rail junction entries must be pairs")
    coupler_ej_pair_ghz = tuple(coupler_ej_pair_ghz)
    coupler_cj_pair_ff = tuple(coupler_cj_pair_ff)
    if len(coupler_ej_pair_ghz) != 2 or len(coupler_cj_pair_ff) != 2:
        raise ValueError("coupler junction parameters must be pairs")
    required_links = {"A1_A2", "A2_C", "C_B1", "B1_B2"}
    missing = sorted(required_links - set(coupling_capacitance_ff))
    if missing:
        raise ValueError(f"coupling_capacitance_ff is missing {missing}")

    numeric = [
        *[value for pair in rail_ej_pairs_ghz for value in pair],
        *[value for pair in rail_cj_pairs_ff for value in pair],
        *rail_charging_energy_ghz,
        *coupler_ej_pair_ghz,
        *coupler_cj_pair_ff,
        coupler_charging_energy_ghz,
        *coupling_capacitance_ff.values(),
        *(parasitic_capacitance_ff or {}).values(),
    ]
    if np.any(~np.isfinite(numeric)) or np.any(np.asarray(numeric) <= 0):
        raise ValueError("all direct-circuit energies and capacitances must be positive and finite")

    branches = ["# direct-five-transmon-dual-rail", "branches:"]
    rail_specs = [
        ("A1", 1, rail_ej_pairs_ghz[0], rail_cj_pairs_ff[0], rail_charging_energy_ghz[0]),
        ("A2", 2, rail_ej_pairs_ghz[1], rail_cj_pairs_ff[1], rail_charging_energy_ghz[1]),
        (
            "C",
            3,
            coupler_ej_pair_ghz,
            coupler_cj_pair_ff,
            coupler_charging_energy_ghz,
        ),
        ("B1", 4, rail_ej_pairs_ghz[2], rail_cj_pairs_ff[2], rail_charging_energy_ghz[2]),
        ("B2", 5, rail_ej_pairs_ghz[3], rail_cj_pairs_ff[3], rail_charging_energy_ghz[3]),
    ]
    for label, node, ej, cj, charging in rail_specs:
        branches.extend(
            (
                f"- [JJ, 0,{node}, EJ_{label}_L={ej[0]}GHz, CJ_{label}_L={cj[0]}fF]",
                f"- [JJ, 0,{node}, EJ_{label}_R={ej[1]}GHz, CJ_{label}_R={cj[1]}fF]",
                f"- [C, 0,{node}, EC_{label}={charging}GHz]",
            )
        )
    link_nodes = {
        "A1_A2": (1, 2, "C_A1A2"),
        "A2_C": (2, 3, "C_A2C"),
        "C_B1": (3, 4, "C_CB1"),
        "B1_B2": (4, 5, "C_B1B2"),
    }
    for key, (left, right, parameter) in link_nodes.items():
        branches.append(
            f"- [C, {left},{right}, {parameter}={coupling_capacitance_ff[key]}fF]"
        )
    label_node = {"A1": 1, "A2": 2, "C": 3, "B1": 4, "B2": 5}
    for key, capacitance in (parasitic_capacitance_ff or {}).items():
        try:
            left_label, right_label = key.split("_")
            left, right = label_node[left_label], label_node[right_label]
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid parasitic link {key!r}") from error
        branches.append(f"- [C, {left},{right}, C_{left_label}{right_label}={capacitance}fF]")
    circuit = _configured_five_transmon_circuit("\n".join(branches))
    missing_flux = [
        name for name in FIVE_TRANSMON_FLUX_MAP.values() if not hasattr(circuit, name)
    ]
    # Test doubles do not synthesize symbolic circuit attributes.  Real Circuit
    # instances must expose every generated loop so ordering cannot drift.
    if hasattr(circuit, "external_fluxes") and missing_flux:
        raise RuntimeError(f"direct circuit is missing external fluxes {missing_flux}")
    return circuit


def gen_dtc(
    Ej1: float = 14,
    Cj1: float = 5,
    C1: float = 194,
    Ej2: float = 14,
    Cj2: float = 5,
    C2: float = 194,
    Ej3: float = 2,
    Cj3: float = 0.2,
    C12: float = 3,
    C1r: float = 15,
    C2r: float = 15,
    Cr: float = 385,
    Lr: float = 1.56,
):
    dtc_yaml = f"""# dtc-dual-rail
        branches:
        - [JJ, 0,1, Ej1={Ej1}GHz, Cj1={Cj1}fF]
        - [C, 0,1, C1={C1}MHz]
        ###
        - [JJ, 0,2, Ej2={Ej2}GHz, Cj2={Cj2}fF]
        - [C, 0,2, C2={C2}MHz]
        ###
        - [JJ, 1,2, Ej3={Ej3}nA, Cj3={Cj3}fF]
        - [C, 1,2, C12={C12}fF]
        ###
        - [C, 1,3, {C1r}fF]
        - [C, 2,3, {C2r}fF]
        ###
        - [C, 3,0, {Cr}fF]
        - [L, 3,0, {Lr}nH]
    """

    return _configured_dual_rail_circuit(dtc_yaml)


def gen_dt(
    Ej1: float = 14,
    Cj1: float = 5,
    C1: float = 194,
    Ej2: float = 14,
    Cj2: float = 5,
    C2: float = 194,
    C12: float = 3,
    C1r: float = 15,
    C2r: float = 15,
    Cr: float = 385,
    Lr: float = 1.56,
):
    dt_yaml = f"""# dtc-dual-rail
        branches:
        - [JJ, 0,1, Ej1={Ej1}GHz, Cj1={Cj1}fF]
        - [C, 0,1, C1={C1}MHz]
        ###
        - [JJ, 0,2, Ej2={Ej2}GHz, Cj2={Cj2}fF]
        - [C, 0,2, C2={C2}MHz]
        ###
        - [C, 1,2, C12={C12}fF]
        ###
        - [C, 1,3, {C1r}fF]
        - [C, 2,3, {C2r}fF]
        ###
        - [C, 3,0, {Cr}fF]
        - [L, 3,0, {Lr}nH]
    """

    return _configured_dual_rail_circuit(dt_yaml)


def gen_tdt(
    Ej1: float = 14,
    Cj1: float = 5,
    Ej1_2: float = 14,
    Cj1_2: float = 5,
    C1: float = 194,
    Ej2: float = 14,
    Cj2: float = 5,
    Ej2_2: float = 14,
    Cj2_2: float = 5,
    C2: float = 194,
    C12: float = 3,
    C1r: float = 15,
    C2r: float = 15,
    Cr: float = 385,
    Lr: float = 1.56,
) -> scq.Circuit:
    dt_yaml = f"""# dtc-dual-rail
        branches:
        - [JJ, 0,1, Ej1={Ej1}GHz, Cj1={Cj1}fF]
        - [JJ, 0,1, Ej1_2={Ej1_2}GHz, Cj1_2={Cj1_2}fF]
        - [C, 0,1, C1={C1}MHz]
        ###
        - [JJ, 0,2, Ej2={Ej2}GHz, Cj2={Cj2}fF]
        - [JJ, 0,2, Ej2_2={Ej2_2}GHz, Cj2_2={Cj2_2}fF]
        - [C, 0,2, C2={C2}MHz]
        ###
        - [C, 1,2, C12={C12}fF]
        ###
        - [C, 1,3, {C1r}fF]
        - [C, 2,3, {C2r}fF]
        ###
        - [C, 3,0, {Cr}fF]
        - [L, 3,0, {Lr}nH]
    """

    return _configured_dual_rail_circuit(dt_yaml)


__all__ = [
    "_circuit_factory_name",
    "_configured_dual_rail_circuit",
    "_validate_circuit_factory_params",
    "gen_dt",
    "gen_dtc",
    "gen_five_transmon_dual_rail",
    "gen_tdt",
]
