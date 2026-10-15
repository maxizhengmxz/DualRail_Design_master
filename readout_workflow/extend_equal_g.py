"""Additional resonance and detuned equal-g searches."""
import argparse
import json
from threadpoolctl import threadpool_limits
from search_equal_g import solve, OUT

def main():
    points=[]
    parser=argparse.ArgumentParser()
    parser.add_argument("--carrier-matching",action="store_true")
    args=parser.parse_args()
    cases=[(c1,c2,6.,4.4,"chi") for c1,c2 in
           [(64,54),(80,80),(95,80),(115,80),(105,105),(115,100)]]
    cases += [(c,c,cap,f,"resonance") for c in (105,115,120)
              for cap in (6.,8.) for f in (3.2,3.6,4.0,4.4)]
    name="extended_grid.json"
    if args.carrier_matching:
        cases = [(118,c2,cap,f,"chi") for c2 in (68,74,80)
                 for cap in (4.,6.) for f in (4.4,4.6,4.8,5.)]
        name="carrier_matching_grid.json"
    with threadpool_limits(limits=1):
        for c1,c2,cap,f,mode in cases:
            p=solve(c1,c2,cap,f,mode=mode)
            points.append(p)
            print(c1,c2,cap,f,mode,round(p["abs_delta_chi_over_chi_p"],6),
                  round(p["lambda_max"],4),round(p["hybridization"],4),
                  p["passes"],flush=True)
            (OUT/name).write_text(json.dumps(points,indent=2))

if __name__ == "__main__":
    main()
