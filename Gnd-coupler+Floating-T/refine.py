"""Calibrate loaded rail pairs and refine a specified five-transmon CZ off channel."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
from floating_coupler.model import build_model, port_exchange
from floating_coupler.design import ej_for_frequency
from floating_coupler.pairs import calibrate_pairs, PairModel, refine_pair_off
from floating_coupler.precision import compare,residual_metrics,VERIFY_BASES,RESIDUAL_HZ
from floating_coupler.channel_policy import selection_from_request,off_target_definition,require_off_definition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8"))
    selection=selection_from_request(spec).to_dict() if spec.get('channel_selection') or spec.get('gate_target',{}).get('channel_selection') else None
    if args.output.exists():require_off_definition(json.loads(args.output.read_text()).get('off_definition'))
    result = {"off_definition":off_target_definition("five"),"input": spec}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    with threadpool_limits(limits=1):
        config = calibrate_pairs(spec)
        if selection is not None:config['channel_selection']=selection
        model = build_model(config)
        result.update(config=config, electrostatics=model.electrostatics)
        reduced = model.subset()
        fc3 = brentq(lambda fc: port_exchange(reduced, ej_for_frequency(fc, reduced.ec[1], 35))["exchange_mhz"],
                     *spec["off_frequency_bracket_ghz"], xtol=1e-10)
        result["reduced_port_root"] = port_exchange(reduced, ej_for_frequency(fc3, reduced.ec[1], 35))
        print(json.dumps({"stage": "calibrated", "three_transmon_fc_ghz": fc3}), flush=True)
        save()
        result["refinement"] = []
        last_root = None;previous_detail=None
        result["precision"]={"residual_hz":RESIDUAL_HZ,"converged":False}
        for basis in spec.get("convergence_bases",VERIFY_BASES):
            fixed=None
            if last_root is not None:
                fixed=PairModel(model,**basis).evaluate(last_root)
                agreement=compare(residual_metrics(previous_detail),residual_metrics(fixed),"residual")
                result["precision"]["last_fixed_control_check"]={"basis":basis,**agreement}
                if agreement["converged"] and abs(fixed["physical_exchange"]["g_mhz"])*1e6<=RESIDUAL_HZ and fixed["physical_exchange"]["valid"] and fixed["logical_exchange"]["valid"]:
                    result["precision"].update(converged=True,final_basis=basis,fc_ghz=last_root,detail=fixed)
                    save();break
            bounds = spec["off_frequency_bracket_ghz"]
            half_width = spec.get("refinement_half_width_ghz", 0.15)
            bracket = bounds if last_root is None else [max(bounds[0], last_root - half_width),
                                                       min(bounds[1], last_root + half_width)]
            check = refine_pair_off(model, 'physical_0100_0010', bracket, basis, scan_points=17 if last_root is None else 3)
            if last_root is not None:
                check["previous_root_fixed_hardware_check"] = fixed
            result["refinement"].append(check)
            save()
            if len(check["roots"]) != 1:
                print(json.dumps({"stage": "ambiguous_or_missing_root", "basis": basis,
                                  "roots": len(check["roots"])}), flush=True)
                break
            previous_detail=check["roots"][0]
            last_root = previous_detail["coupler_bare_f01_ghz"]
            print(json.dumps({"stage": "root", "basis": basis, "fc_ghz": last_root,
                                  "target": check["roots"][0]["physical_exchange"],
                              "logical_zz_mhz": check["roots"][0]["logical_zz_mhz"],
                              "overlap": check["roots"][0]["minimum_subspace_overlap"]}), flush=True)


if __name__ == "__main__":
    main()
