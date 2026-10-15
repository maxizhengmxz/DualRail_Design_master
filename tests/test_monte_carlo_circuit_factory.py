import unittest
from unittest.mock import patch

from dtc_dualrail.circuits import (
    _circuit_factory_name,
    _configured_dual_rail_circuit,
    _validate_circuit_factory_params,
    gen_dt,
    gen_dtc,
    gen_five_transmon_dual_rail,
    gen_tdt,
)


def custom_factory(required, optional=1.0):
    return required + optional


class FakeScqubitsCircuit:
    def __init__(self, yaml, *, from_file, ext_basis):
        self.yaml = yaml
        self.from_file = from_file
        self.ext_basis = ext_basis
        self.configure_kwargs = None

    def configure(self, **kwargs):
        self.configure_kwargs = kwargs


class CircuitFactoryTests(unittest.TestCase):
    def test_builtin_generators_accept_their_own_parameter_sets(self):
        dt_params = {
            "Ej1": 14,
            "Cj1": 5,
            "C1": 194,
            "Ej2": 14,
            "Cj2": 5,
            "C2": 194,
            "C12": 3,
            "C1r": 15,
            "C2r": 15,
            "Cr": 385,
            "Lr": 1.56,
        }
        dtc_params = {**dt_params, "Ej3": 0.56, "Cj3": 0.2}
        tdt_params = {
            **dt_params,
            "Ej1_2": 14,
            "Cj1_2": 5,
            "Ej2_2": 14,
            "Cj2_2": 5,
        }

        _validate_circuit_factory_params(gen_dt, [dt_params])
        _validate_circuit_factory_params(gen_dtc, [dtc_params])
        _validate_circuit_factory_params(gen_tdt, [tdt_params])

    def test_rejects_unknown_parameters_for_selected_generator(self):
        with self.assertRaisesRegex(ValueError, "unknown parameters.*Ej3"):
            _validate_circuit_factory_params(gen_dt, [{"Ej1": 14, "Ej3": 0.56}])

    def test_rejects_missing_required_parameters_for_custom_factory(self):
        with self.assertRaisesRegex(ValueError, "missing required parameters.*required"):
            _validate_circuit_factory_params(custom_factory, [{"optional": 2.0}])

    def test_circuit_factory_name_is_stable_for_dataset_attrs(self):
        self.assertEqual(_circuit_factory_name(gen_dt), "gen_dt")
        self.assertEqual(_circuit_factory_name(custom_factory), "custom_factory")

    def test_shared_configurator_applies_dual_rail_defaults(self):
        with patch("dtc_dualrail.circuits.scq.Circuit", FakeScqubitsCircuit):
            circuit = _configured_dual_rail_circuit("branches: []")

        self.assertEqual(circuit.yaml, "branches: []")
        self.assertFalse(circuit.from_file)
        self.assertEqual(circuit.ext_basis, "harmonic")
        self.assertEqual(circuit.cutoff_n_1, 20)
        self.assertEqual(circuit.cutoff_n_2, 20)
        self.assertEqual(circuit.cutoff_ext_3, 20)
        self.assertEqual(circuit.configure_kwargs["system_hierarchy"], [[1, 2], [3]])
        self.assertEqual(circuit.configure_kwargs["subsystem_trunc_dims"], [40, 18])

    def test_direct_five_transmon_factory_has_stable_fluxes_and_links(self):
        with patch("dtc_dualrail.circuits.scq.Circuit", FakeScqubitsCircuit):
            circuit = gen_five_transmon_dual_rail(
                rail_ej_pairs_ghz=[(12.0, 8.0)] * 4,
                rail_cj_pairs_ff=[(5.0, 5.0)] * 4,
                rail_charging_energy_ghz=[0.3, 0.31, 0.32, 0.33],
                coupler_ej_pair_ghz=(12.5, 11.5),
                coupler_cj_pair_ff=(0.2, 0.2),
                coupler_charging_energy_ghz=0.6,
                coupling_capacitance_ff={
                    "A1_A2": 2.9,
                    "A2_C": 3.1,
                    "C_B1": 3.2,
                    "B1_B2": 2.8,
                },
            )

        self.assertIn("EJ_A1_L=12.0GHz", circuit.yaml)
        self.assertIn("C_A2C=3.1fF", circuit.yaml)
        self.assertIn("C_CB1=3.2fF", circuit.yaml)
        self.assertEqual(
            circuit.dualrail_flux_map,
            {"A1": "Φ1", "A2": "Φ2", "C": "Φ3", "B1": "Φ4", "B2": "Φ5"},
        )
        self.assertEqual(circuit.configure_kwargs["system_hierarchy"], [[1], [2], [3], [4], [5]])
        self.assertEqual(circuit.configure_kwargs["subsystem_trunc_dims"], [3, 3, 4, 3, 3])


if __name__ == "__main__":
    unittest.main()
