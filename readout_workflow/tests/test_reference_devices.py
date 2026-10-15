import unittest
from model import Device, ReadoutModel
from reference_devices import original_preset


def reference_model(name):
    p = original_preset(name)
    return ReadoutModel(Device(**{
        k: p[k] for k in ("C1_shunt_fF", "C2_shunt_fF", "C12_fF", "C1r_fF", "C2r_fF")
    }, Cr_fF=531 if name == "tdt1" else 531*1.01,
       Lr_nH=1.33 if name == "tdt1" else 1.33*1.01))


class ReferenceDeviceTests(unittest.TestCase):
    def test_original_capacitance_loading_is_preserved(self):
        model = reference_model("tdt1")
        self.assertAlmostEqual(model.device.C1_shunt_fF, 64.71816193406532)
        self.assertAlmostEqual(model.device.C2_shunt_fF, 54.631864803362035)
        self.assertAlmostEqual(model.ec_ghz[0]*1e3, 237.6275060185721)
        self.assertAlmostEqual(model.ec_ghz[1]*1e3, 271.1344958764884)

    def test_both_devices_reproduce_sub_one_per_mille_cached_points(self):
        points = {"tdt1": (10.808938851700203, 9.645285353413106, 4.25614764501062),
                  "tdt2": (13.006442951152597, 11.38987638952526, 4.466872337993935)}
        for name, (ej1, ej2, frequency) in points.items():
            with self.subTest(device=name):
                p = reference_model(name).evaluate(ej1, ej2)
                self.assertAlmostEqual(p["operating_frequency_ghz"], frequency, delta=2e-6)
                self.assertLess(p["abs_delta_chi_over_chi_p"], 1e-3)


if __name__ == "__main__":
    unittest.main()
