"""Stages 2-5: electrostatic budget, physical root, EJ-only search and audits.

All five EJs are optimization variables. The chosen capacitance matrix is
constructed once per geometry and is independent of EJ. No readout optimizer
is called: the imported readout module supplies diagnostics only.
"""

import json
from pathlib import Path

import numpy as np
from scipy.constants import e, h
from scipy.optimize import brentq, minimize

from coupler_turn_off.circuit import BudgetModel, capacitance_matrix
from coupler_turn_off.logical import energy_exchange
from coupler_turn_off.readout import block_checks
from readout_workflow.calculation import Basis, _transmon


def save(path, value):
    """Atomic checkpoint; interrupted writes leave the previous JSON intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def objective_mhz(result, objective):
    """Cyclic H/h magnitudes; combined gives equal quadratic weight to J1/ZZ."""
    if objective == "physical":
        return abs(result["physical_signed_exchange_mhz"])
    j = result["energy_basis_logical_exchange_mhz"]
    if objective == "logical":
        return abs(j)
    if objective == "combined":
        return float(np.hypot(j, result["energy_basis_logical_zz_mhz"]))
    raise ValueError(f"Unknown objective: {objective}")


def resolve_direct_capacitance(config):
    """Convert a supplied bare rate using the disconnected-mediator circuit.

    A rate means |<10|H/h|01>| at the supplied nominal rail EJs, with both
    coupler links removed. It is NOT Jport after virtual coupler exchange.
    The capacitance is solved with loaded rail EC preserved, not an assumed
    linear rate/capacitance proportionality. The resolved C stays fixed later.
    """
    direct = config["direct_a2_b1"]
    if direct.get("capacitance_ff") is not None:
        return float(direct["capacitance_ff"])
    point = config["pair"]
    target = direct["bare_exchange_mhz"]
    if target == 0:
        return 0.0
    basis = Basis(25, 8, 3, 12)
    n2 = abs(
        _transmon(point["ej2_ghz"], point["loaded_ec2_mhz"] / 1000, basis)[1][0, 1]
    )
    n1 = abs(
        _transmon(point["ej1_ghz"], point["loaded_ec1_mhz"] / 1000, basis)[1][0, 1]
    )

    def residual(cap):
        matrix, _ = capacitance_matrix(
            point, 0, config["coupler_candidates"][0]["ec_ghz"], cap
        )
        coefficient_ghz = 4 * e**2 * np.linalg.inv(matrix * 1e-15)[1, 4] / (h * 1e9)
        return 1000 * abs(coefficient_ghz * n1 * n2) - target

    maximum = 0.99 * min(point["device"]["C1_shunt_fF"], point["device"]["C2_shunt_fF"])
    upper = min(0.001, maximum)
    while residual(upper) < 0 and upper < maximum:
        upper = min(upper * 2, maximum)
    if residual(upper) < 0:
        raise ValueError(
            "Requested direct rate exceeds the supported capacitance budget"
        )
    return float(brentq(residual, 0, upper, xtol=1e-12))


class Case:
    """One supplied pair and one fixed coupler geometry, with no disk defaults."""

    def __init__(self, config, candidate):
        self.config, self.candidate = config, candidate
        self.point = config["pair"]
        self.direct_ff = resolve_direct_capacitance(config)
        self.nominal_rail_ej = [self.point["ej1_ghz"], self.point["ej2_ghz"]] * 2
        self.bounds = np.asarray(config["ej_search_bounds_ghz"], dtype=float).copy()
        screens = config["screens"]
        ecs = (
            np.array([self.point["loaded_ec1_mhz"], self.point["loaded_ec2_mhz"]] * 2)
            / 1000
        )
        self.bounds[:4, 0] = np.maximum(
            self.bounds[:4, 0], ecs * screens["rail_ej_over_ec_min"] * (1 + 1e-9)
        )
        self.bounds[:4, 1] = np.minimum(
            self.bounds[:4, 1], screens["rail_ej_max_ghz"] * (1 - 1e-9)
        )
        if screens["coupler_ej_max_ghz"] is not None:
            self.bounds[4, 1] = min(
                self.bounds[4, 1], screens["coupler_ej_max_ghz"] * (1 - 1e-9)
            )
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError(
                "EJ search bounds and hardware constraints have no intersection"
            )
        if candidate["ec_ghz"] * 1000 >= screens["coupler_ec_max_mhz"]:
            raise ValueError("Candidate violates the coupler EC upper bound")
        low, high = screens["rail_ec_bounds_mhz"]
        if np.any(ecs * 1000 < low) or np.any(ecs * 1000 >= high):
            raise ValueError("Supplied loaded rail EC is outside the agreed bounds")

    def model(self, ej, basis=None):
        levels, photons, keep, clevels = basis or self.config["basis"]
        return BudgetModel(
            self.point,
            self.candidate["links_ff"][0],
            self.candidate["ec_ghz"],
            levels,
            photons,
            keep,
            clevels,
            direct_ff=self.direct_ff,
            fixed_links_ff=self.candidate["links_ff"],
            rail_ej_ghz=ej[:4],
            resonator_inductances_nh=self.config["resonator_inductances_nh"],
            readout_window_ghz=self.config["readout_window_ghz"],
            eigenenergy_max_ghz=self.config.get("spectral_ceiling_ghz"),
        )

    def evaluate(self, ej, basis=None, readout=True):
        model = self.model(ej, basis)
        result = model.evaluate(float(ej[4]), readout=readout)
        result.update(energy_exchange(result))
        result.update(
            model.physical_exchange(
                float(ej[4]), levels=(basis or self.config["basis"])[0]
            )
        )
        if readout:
            result["pair_block_checks"] = block_checks(model)
        return result

    def nominal_roots(self, samples=31):
        """Stage 3: fixed rails, reduced A2-C-B1 roots over explicit EJ bracket.

        EC and link geometries are enumerated by the CLI, not reoptimized here.
        A sign scan can miss tangential roots; absence is not proof of no root.
        """
        ej = self.nominal_rail_ej + [float(self.bounds[4].mean())]
        model = self.model(ej)
        if not np.all(
            (np.asarray(ej[:4]) >= self.bounds[:4, 0])
            & (np.asarray(ej[:4]) <= self.bounds[:4, 1])
        ):
            raise ValueError("Supplied nominal rail EJs violate the agreed EJ bounds")

        def signed(value):
            return model.physical_exchange(value, levels=self.config["basis"][0])[
                "physical_signed_exchange_mhz"
            ]

        grid = np.linspace(*self.bounds[4], samples)
        values = [signed(value) for value in grid]
        roots = []
        for lo, hi, flo, fhi in zip(
            grid[:-1], grid[1:], values[:-1], values[1:], strict=True
        ):
            if flo * fhi > 0:
                continue
            root = (
                lo
                if flo == 0
                else hi
                if fhi == 0
                else brentq(signed, lo, hi, xtol=1e-10)
            )
            if any(abs(root - r["ej_ghz"][4]) < 1e-8 for r in roots):
                continue
            check = model.physical_exchange(root, levels=self.config["basis"][0])
            fmin, fmax = self.config["screens"]["coupler_frequency_bounds_ghz"]
            if not fmin <= check["coupler_f01_ghz"] <= fmax:
                continue
            roots.append(
                dict(
                    ej_ghz=self.nominal_rail_ej + [float(root)],
                    physical=check,
                    capacitance_matrix_ff=model.cap.tolist(),
                    direct_capacitance_ff=self.direct_ff,
                    ground_shunts_ff=model.cap.sum(axis=1).tolist(),
                    candidate=self.candidate,
                )
            )
        return roots

    def margins(self, result):
        """Named, dimensionless nonnegative feasibility margins (no chi reward)."""
        s, margins = self.config["screens"], {}

        def lower(name, value, bound, scale=1):
            margins[name] = (value - bound) / scale

        def upper(name, value, bound, scale=1):
            margins[name] = (bound - value) / scale

        for c in result["energy_mode_readout_checks"]:
            tag = f"{c['resonator']}_{c['spectator_logical']}"
            upper(tag + "_chi_ratio", c["mismatch"], s["chi_ratio_max"], 0.01)
            lower(
                tag + "_chi_magnitude",
                abs(c["mean_chi_mhz"]),
                s["abs_mean_chi_min_mhz"],
            )
            lower(
                tag + "_hybridization", c["hybridization"], s["hybridization_min"], 0.01
            )
            for name, value, bounds in (
                (
                    "logical",
                    c["logical_frequency_mhz"],
                    s["logical_frequency_bounds_mhz"],
                ),
                (
                    "operation",
                    c["operation_frequency_ghz"],
                    s["operation_frequency_bounds_ghz"],
                ),
            ):
                lower(tag + "_" + name + "_low", value, bounds[0])
                upper(tag + "_" + name + "_high", value, bounds[1])
            lower(
                tag + "_below_readout",
                c["resonator_frequency_ghz"] - c["operation_frequency_ghz"],
                0,
            )
        for i, block in enumerate(result["pair_block_checks"]):
            upper(
                f"pair{i}_dispersive",
                block["dispersive_ratio"],
                s["dispersive_max"],
                0.01,
            )
        lower(
            "readout_tracking",
            result["readout_assignment_min"],
            s["tracking_min"],
            0.01,
        )
        lower("logical_tracking", result["assignment_min"], s["tracking_min"], 0.01)
        lower(
            "readout_space",
            result["readout_subspace_sigma_min"],
            s["readout_subspace_min"],
            0.01,
        )
        lower("logical_space", result["sigma_min"], s["logical_subspace_min"], 0.01)
        for key, bound, sign in (
            ("coupler_low", s["coupler_frequency_bounds_ghz"][0], 1),
            ("coupler_high", s["coupler_frequency_bounds_ghz"][1], -1),
        ):
            margins[key] = sign * (result["coupler_f01_ghz"] - bound)
        return margins

    def optimize(self, seed, objective, output, maxiter=40):
        """Stage 4: exactly one explicitly selected objective, all caps fixed.

        SLSQP is local and can fail to find a feasible point. Report its status
        and actual constraint violations, never just its termination message.
        """
        output = Path(output)
        reference = np.asarray(seed, dtype=float)
        cache = {}

        def evaluate(x):
            key = tuple(np.asarray(x, dtype=float))
            if key not in cache:
                cache[key] = self.evaluate(np.asarray(x) * reference)
                save(
                    output / "progress.json",
                    dict(
                        evaluations=len(cache),
                        ej_ghz=(np.asarray(x) * reference).tolist(),
                        objective=objective,
                        objective_mhz=objective_mhz(cache[key], objective),
                        margins=self.margins(cache[key]),
                    ),
                )
            return cache[key]

        bounds = self.bounds / reference[:, None]
        # Scaling changes numerical conditioning only, not the physical objective.
        fit = minimize(
            lambda x: (objective_mhz(evaluate(x), objective) / 0.001) ** 2,
            np.clip(np.ones(5), bounds[:, 0], bounds[:, 1]),
            method="SLSQP",
            bounds=bounds.tolist(),
            constraints=[
                dict(
                    type="ineq",
                    fun=lambda x: np.array(list(self.margins(evaluate(x)).values())),
                )
            ],
            options=dict(maxiter=maxiter, ftol=1e-10, eps=3e-6),
        )
        ej = fit.x * reference
        result = evaluate(fit.x)
        baseline = 1000 * (self.point["chi_m_ghz"] + self.point["chi_p_ghz"])
        comparison = []
        for c in result["energy_mode_readout_checks"]:
            chi_sum = 2 * c["mean_chi_mhz"]
            comparison.append(
                dict(
                    **c,
                    chi_sum_mhz=chi_sum,
                    chi_sum_magnitude_degradation_percent=100
                    * (1 - abs(chi_sum / baseline))
                    if baseline
                    else None,
                )
            )
        record = dict(
            objective=objective,
            ej_ghz=ej.tolist(),
            optimizer_success=bool(fit.success),
            message=str(fit.message),
            nominal_constraints_pass=all(v >= 0 for v in self.margins(result).values()),
            margins=self.margins(result),
            readout_comparison=comparison,
            result=result,
            capacitance_matrix_ff=self.model(ej).cap.tolist(),
            candidate=self.candidate,
        )
        save(output / "optimized.json", record)
        return record

    def audit(self, ej):
        """Stage 5: recompute in both bases; no additional optimization."""
        from coupler_turn_off.channels import channels

        records = []
        for basis in (self.config["basis"], self.config["audit_basis"]):
            result = self.evaluate(ej, basis)
            records.append(
                dict(
                    basis=basis,
                    result=result,
                    margins=self.margins(result),
                    channels=channels(result),
                )
            )
        return records
