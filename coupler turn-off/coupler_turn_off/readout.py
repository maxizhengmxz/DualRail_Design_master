"""Loaded isolated-pair dispersive diagnostics, not a readout optimizer.

These checks include electrostatic loading, but omit coupler dynamics.
The full-circuit conditional chi/hybridization checks come from BudgetModel.
"""

import numpy as np

from readout_workflow.calculation import Basis, Device, ReadoutModel, annotate, failures


def block_checks(model):
    """Original readout screens on isolated dressed pair blocks, not full-network certificates."""
    reports = []
    for nodes, ej in [
        ([0, 1, 2], model.rail_ej_ghz[:2]),
        ([4, 5, 6], model.rail_ej_ghz[2:]),
    ]:
        matrix = np.linalg.inv(model.cinv[np.ix_(nodes, nodes)]) * 1e15
        ground = matrix.sum(axis=1)
        device = Device(
            C1_shunt_fF=ground[0],
            C2_shunt_fF=ground[1],
            Cr_fF=ground[2],
            C12_fF=-matrix[0, 1],
            C1r_fF=-matrix[0, 2],
            C2r_fF=-matrix[1, 2],
            Lr_nH=model.resonator_inductances_nh[0 if nodes[0] == 0 else 1],
        )
        p = annotate(ReadoutModel(device, Basis(25, 8, 6, 40)), *ej)
        reports.append(
            dict(
                ej_ghz=list(ej),
                loaded_ec_mhz=[p["loaded_ec1_mhz"], p["loaded_ec2_mhz"]],
                ej_over_ec=[p["ej1_over_ec1"], p["ej2_over_ec2"]],
                hybridization=p["hybridization"],
                logical_frequency_mhz=1000 * p["splitting_ghz"],
                operation_frequency_ghz=p["operating_frequency_ghz"],
                resonator_frequency_ghz=p["dressed_r_ghz"],
                dispersive_ratio=p["lambda_max"],
                assignment_overlap=p["min_assignment_overlap"],
                failures_excluding_sign_grid=failures(p),
                scope="isolated pair block with full-network electrostatic loading; excludes coupler dynamics",
            )
        )
    return reports
