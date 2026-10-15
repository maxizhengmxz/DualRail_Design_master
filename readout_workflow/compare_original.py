"""Compare cached original SQUID results with the equivalent single-junction model.

Run with the parent project's environment (xarray is needed to read its caches).
Junction capacitances are absorbed into the shunts; no junction C is introduced.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from scipy.constants import elementary_charge as e, h
from scipy.optimize import brentq, minimize_scalar
from threadpoolctl import threadpool_limits
import xarray as xr

from guided_search import predict_resonances
from model import Basis, Device, ReadoutModel


def effective_ej(a, b, flux):
    return float(np.sqrt(a*a+b*b+2*a*b*np.cos(2*np.pi*flux)))


def original_device(params):
    return Device(
        C1_shunt_fF=e**2/(2*h*params["C1"]*1e6)*1e15+params["Cj1"]+params["Cj1_2"],
        C2_shunt_fF=e**2/(2*h*params["C2"]*1e6)*1e15+params["Cj2"]+params["Cj2_2"],
        C12_fF=params["C12"], C1r_fF=params["C1r"], C2r_fF=params["C2r"],
        Cr_fF=params["Cr"], Lr_nH=params["Lr"],
    )


def resonance_point(model, ej1):
    roots, _ = predict_resonances(model, ej1, "EJ2_GHz", (2, 25))
    if len(roots) != 1:
        raise RuntimeError(f"Expected one resonance prediction for EJ1={ej1}")
    center = roots[0]
    result = minimize_scalar(lambda v: model.evaluate(ej1, v)["splitting_ghz"],
                             method="bounded", bounds=(0.85*center, 1.15*center),
                             options={"xatol": 1e-7})
    if not result.success:
        raise RuntimeError("Crossing refinement failed")
    return model.evaluate(ej1, result.x)


def compare():
    root = Path(__file__).resolve().parent
    report = {}
    with threadpool_limits(limits=1):
        for name in ("tdt1", "tdt2"):
            cached = xr.load_dataset(root.parent/"data"/f"double_tdt_operating_point_dispersive_{name}.nc")
            params = dict(zip(cached.param.values.tolist(), cached.param_value.values.tolist()))
            device = original_device(params)
            model = ReadoutModel(device)
            rows = []
            for i in range(cached.sizes["operating_point"]):
                old = cached.isel(operating_point=i)
                ej1 = effective_ej(params["Ej1"], params["Ej1_2"], float(old.phi_1))
                ej2 = effective_ej(params["Ej2"], params["Ej2_2"], float(old.phi_2))
                new = model.evaluate(ej1, ej2)
                rows.append({"old_frequency_ghz": float(old.operating_frequency_ghz),
                             "old_ratio": float(old.abs_delta_chi_over_chi_p),
                             "old_chi_p_ghz": float(old.chi_p), "old_chi_m_ghz": float(old.chi_m),
                             **new})
            ej1_target = brentq(lambda ej1: resonance_point(model, ej1)["operating_frequency_ghz"]-4.4,
                                5, 18, xtol=1e-7)
            target = resonance_point(model, ej1_target)
            larger = ReadoutModel(device, Basis(ncut=20, transmon_levels=9, resonator_levels=8, eigenlevels=30))
            target_large = larger.evaluate(target["ej1_ghz"], target["ej2_ghz"])
            best = min(rows, key=lambda row: row["old_ratio"])
            entry = {"original_params": params, "equivalent_device": asdict(device),
                     "loaded_ec_mhz": (model.ec_ghz[:2]*1e3).tolist(),
                     "bare_resonator_frequency_ghz": 1/(2*np.pi*np.sqrt(device.Lr_nH*1e-9*device.Cr_fF*1e-15))/1e9,
                     "mode_impedance_ohm": float(np.sqrt(device.Lr_nH*1e-9/(device.Cr_fF*1e-15))),
                     "target_4p4ghz": target, "target_larger_basis": target_large,
                     "best_original_cached_point": best, "comparison_rows": rows}
            report[name] = entry
            print(name, "effective shunts fF:", device.C1_shunt_fF, device.C2_shunt_fF)
            print("loaded EC/h MHz:", entry["loaded_ec_mhz"])
            print("4.4 GHz target:", target)
            print("Larger-basis ratio:", target_large["abs_delta_chi_over_chi_p"])
            print("Best original cache:", best)
            print("Max chi discrepancy MHz:", max(max(abs(r["chi_p_ghz"]-r["old_chi_p_ghz"]),abs(r["chi_m_ghz"]-r["old_chi_m_ghz"])) for r in rows)*1e3)
    (root/"data").mkdir(exist_ok=True)
    (root/"data"/"original_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    compare()
