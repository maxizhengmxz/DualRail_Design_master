"""Controlled shunt changes and a rail-label exchange check at 1% matching."""
from dataclasses import replace
import json
from pathlib import Path
from threadpoolctl import threadpool_limits

from model import Device, ReadoutModel
from search_equal_g import annotate
from study_capacitance_robustness import CarrierSlice

ROOT=Path(__file__).resolve().parent/"data"/"equal_g"/"robustness"


def swapped_device(d):
    return replace(d,C1_shunt_fF=d.C2_shunt_fF,C2_shunt_fF=d.C1_shunt_fF,
                   C1r_fF=d.C2r_fF,C2r_fF=d.C1r_fF)


def main():
    entries=json.loads((ROOT/"tradeoff.json").read_text())
    rows=[]
    with threadpool_limits(limits=1):
        for entry in entries:
            base=entry["constrained_maximum"]
            d=Device(**base["device"])
            for parameter,error in (("nominal",0),("C1_shunt_fF",-.02),("C1_shunt_fF",.02),
                                    ("C2_shunt_fF",-.02),("C2_shunt_fF",.02)):
                changed=d if not error else replace(d,**{parameter:getattr(d,parameter)*(1+error)})
                m=ReadoutModel(changed)
                sl=CarrierSlice(m,base)
                # Restore the nominal *local* frequencies first, to separate
                # changed charge physics from subsequent detuning for matching.
                restored=m.evaluate(sl.reference["ej1_ghz"],sl.reference["ej2_ghz"])
                tuned=annotate(m,sl.constrained(.01))
                swapped_reference=ReadoutModel(swapped_device(d)).evaluate(base["ej2_ghz"],base["ej1_ghz"])
                sm=ReadoutModel(swapped_device(changed))
                sp=annotate(sm,CarrierSlice(sm,swapped_reference).constrained(.01))
                rows.append({"design":entry["label"],"parameter":parameter,"error":error,
                    "local_frequencies_restored":annotate(m,restored),"matched":tuned,
                    "matched_bare_detuning_mhz":1000*(tuned["bare_f1_ghz"]-tuned["bare_f2_ghz"]),
                    "swapped_matched":sp,"label_swap_H_difference":sp["hybridization"]-tuned["hybridization"]})
    (ROOT/"asymmetry_1pct.json").write_text(json.dumps(rows,indent=2))
    for r in rows:
        p=r["matched"]
        print(r["design"],r["parameter"],r["error"],"H",round(p["hybridization"],7),
              "detuning_MHz",round(r["matched_bare_detuning_mhz"],5),
              "swap_dH",f"{r['label_swap_H_difference']:.3g}")

if __name__=="__main__": main()
