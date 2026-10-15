"""Search the actual one-percent target, rather than requiring exact matching."""
import json
from threadpoolctl import threadpool_limits
from search_equal_g import solve, OUT

def main():
    points=[]
    with threadpool_limits(limits=1):
        for c1,c2 in ((115,80),(120,78),(120,80),(118,78),(115,75),(122,78)):
            for cap in (6.,8.):
                for target in (-.008,.008):
                    p=solve(c1,c2,cap,mode="chi",chi_target=target)
                    points.append(p)
                    print(c1,c2,cap,target,round(p["abs_delta_chi_over_chi_p"],6),
                          round(p["lambda_max"],4),round(p["hybridization"],4),
                          round(p["loaded_ec1_mhz"],2),round(p["ej2_over_ec2"],2),
                          p["passes"],flush=True)
                    (OUT/"one_percent_grid.json").write_text(json.dumps(points,indent=2))

if __name__ == "__main__":
    main()
