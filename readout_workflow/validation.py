"""Shared reported quantities; all chi inputs are frequency shifts in GHz."""

from . import calculation as study
from core.transmon import Basis, transmon_spectrum
from core.electrostatics import exchange_j01_ghz

AUDIT = Basis(25, 12, 10, 60)


def audit_point(point, sign=True):
    """Recompute unchanged physical inputs and attach local exchange coupling."""
    model = study.ReadoutModel(study.Device(**point["device"]), AUDIT)
    result = study.annotate(model, point["ej1_ghz"], point["ej2_ghz"])
    result["x"] = study.x_from_point(result)
    n1 = transmon_spectrum(result["ej1_ghz"], model.ec_ghz[0], AUDIT)[1]
    n2 = transmon_spectrum(result["ej2_ghz"], model.ec_ghz[1], AUDIT)[1]
    result["J12_ghz"] = float(model.J_ghz[0, 1])
    result["J01_12_mhz"] = 1000 * exchange_j01_ghz(
        model.J_ghz[0, 1], n1[0, 1], n2[0, 1]
    )
    result["g12_over_2pi_mhz"] = result["J01_12_mhz"]  # Legacy alias.
    if sign:
        result = study.check_chi_sign_stability(result, model)
    return result


def metrics(point):
    """Keep signed sum/difference and explicitly distinguish their magnitudes."""
    minus, plus = point["chi_m_ghz"], point["chi_p_ghz"]
    return dict(
        chi_minus_mhz=1000 * minus,
        chi_plus_mhz=1000 * plus,
        delta_chi_khz=1e6 * (plus - minus),
        chi_sum_mhz=1000 * (plus + minus),
        absolute_chi_sum_mhz=1000 * abs(plus + minus),
        absolute_mean_chi_mhz=study.absolute_mean_chi_mhz(point),
        ratio=point["ratio"],
    )
