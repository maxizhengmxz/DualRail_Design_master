"""Original cached-device presets, converted to ideal single-junction circuits."""
import numpy as np
from scipy.constants import elementary_charge as e, h


def original_preset(name):
    # Source: data/double_tdt_operating_point_dispersive_tdt{1,2}.nc param_value.
    settings = {
        "tdt1": dict(ec1_mhz=354., ec2_mhz=434., c1r=15., c2r=15., cr=531., lr=1.33,
                     ej1_4p4=11.516088299896243, ej2_4p4=10.275189250453693),
        "tdt2": dict(ec1_mhz=294., ec2_mhz=364., c1r=13., c2r=13., cr=531.*1.01, lr=1.33*1.01,
                     ej1_4p4=12.633368460744236, ej2_4p4=11.067761635367075),
    }
    if name not in settings:
        raise ValueError("Choose tdt1 or tdt2")
    p = settings[name]
    # The two original 5 fF junction capacitors per transmon become part of
    # its one explicit shunt. This preserves C exactly without junction C elements.
    return {
        "C1_shunt_fF": e**2/(2*h*p["ec1_mhz"]*1e6)*1e15+10.,
        "C2_shunt_fF": e**2/(2*h*p["ec2_mhz"]*1e6)*1e15+10.,
        "C12_fF": 2.3, "C1r_fF": p["c1r"], "C2r_fF": p["c2r"],
        "resonator_frequency_ghz": 1/(2*np.pi*np.sqrt(p["cr"]*1e-15*p["lr"]*1e-9))/1e9,
        "resonator_mode_impedance_ohm": float(np.sqrt(p["lr"]*1e-9/(p["cr"]*1e-15))),
        "original_ec_mhz": (p["ec1_mhz"], p["ec2_mhz"]),
        "ej1_4p4_ghz": p["ej1_4p4"], "ej2_4p4_ghz": p["ej2_4p4"],
    }
