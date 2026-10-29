"""Recalibrate dressed static CZ phase first, then logical return."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from threadpoolctl import threadpool_limits
from floating_coupler.model import build_model
from floating_coupler.calibrate_on import PhaseCalibration
from floating_coupler.precision import verify_fixed_controls,VERIFY_BASES,GATE_ERROR_ABS,correct_gate_phase


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    spec = json.loads(args.config.read_text(encoding="utf-8"))
    model = build_model(spec["five_config"])
    result = {"input": spec, "audits": []}

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    with threadpool_limits(limits=1):
        cal = PhaseCalibration(model, spec["fc_ghz"], spec["search_basis"], spec["fb_bounds_ghz"])
        result["original"] = cal.at(spec["initial_fb_ghz"], spec["initial_time_ns"])
        result["phase_only"] = cal.phase_root(spec["initial_time_ns"])
        result["time_refinement"] = cal.optimize_time(spec["time_bounds_ns"], spec.get("time_samples", 21))
        best = result["time_refinement"]["best"]
        print(json.dumps({"stage": "search", "phase_only": result["phase_only"], "best": best}), flush=True)
        save()
        def fixed_controls(basis):
            audit=PhaseCalibration(model,spec['fc_ghz'],basis,spec['fb_bounds_ghz'])
            point=audit.at(best['fb_ghz'],best['time_ns'])
            result['audits'].append(dict(basis=basis,fixed_search_endpoint=point))
            save()
            return point
        bases=[spec['search_basis']]+spec.get('audit_bases',VERIFY_BASES)
        # Compare fixed controls first; retune phase only once after convergence.
        result['precision_validation']=verify_fixed_controls(fixed_controls,bases,'gate')
        if result['precision_validation']['converged']:
            basis=result['precision_validation']['final_basis']
            final=PhaseCalibration(model,spec['fc_ghz'],basis,spec['fb_bounds_ghz'])
            result['phase_recalibrated_endpoint']=correct_gate_phase(
                lambda fb:final.at(fb,best['time_ns']),best,spec['fb_bounds_ghz'])
        result['gate_error_accuracy_absolute']=GATE_ERROR_ABS
        save()



if __name__ == "__main__":
    main()
