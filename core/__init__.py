"""Shared physics and utilities only; no gate or legacy forwarding imports."""

from importlib import import_module

_EXPORTS = {'gen_dt': ('core.circuits', 'gen_dt'),
 'gen_dtc': ('core.circuits', 'gen_dtc'),
 'gen_five_transmon_dual_rail': ('core.circuits', 'gen_five_transmon_dual_rail'),
 'gen_tdt': ('core.circuits', 'gen_tdt'),
 'BRANCH_INDEX': ('core.constants', 'BRANCH_INDEX'),
 'BRANCH_LABELS': ('core.constants', 'BRANCH_LABELS'),
 'DATA_DIR': ('core.constants', 'DATA_DIR'),
 'DEFAULT_PARAM_REL_SIGMA': ('core.constants', 'DEFAULT_PARAM_REL_SIGMA'),
 'DEFAULT_PARAM_SIGMA': ('core.constants', 'DEFAULT_PARAM_SIGMA'),
 'FLUX_PARAM': ('core.constants', 'FLUX_PARAM'),
 'TRACKING_SCHEMA': ('core.constants', 'TRACKING_SCHEMA'),
 'save_dataset': ('core.io', 'save_dataset'),
 'evaluate_dispersive_shifts_at_flux_pair': ('core.operating_point_dispersive',
                                             'evaluate_dispersive_shifts_at_flux_pair'),
 'load_or_compute_operating_point_dispersive_sweep': ('core.operating_point_dispersive',
                                                      'load_or_compute_operating_point_dispersive_sweep'),
 'run_operating_point_dispersive_sweep': ('core.operating_point_dispersive',
                                          'run_operating_point_dispersive_sweep'),
 'evaluate_rail_hybridization': ('core.operating_points',
                                 'evaluate_rail_hybridization'),
 'find_operating_points': ('core.operating_points', 'find_operating_points'),
 'load_or_compute_operating_points': ('core.operating_points',
                                      'load_or_compute_operating_points'),
 'track_levels_by_overlap': ('core.tracking', 'track_levels_by_overlap'),
 'Basis': ('core.transmon', 'Basis'),
 'transmon_spectrum': ('core.transmon', 'transmon_spectrum'),
 'charge_coupling_ghz': ('core.electrostatics', 'charge_coupling_ghz'),
 'exchange_j01_ghz': ('core.electrostatics', 'exchange_j01_ghz'),
 'loaded_charging_energies_ghz': ('core.electrostatics',
                                  'loaded_charging_energies_ghz')}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
