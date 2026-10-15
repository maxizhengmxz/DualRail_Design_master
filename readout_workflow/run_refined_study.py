"""Checkpointed multi-stage study; main process owns all SQLite writes.

Only C1>C2 is computed. Final plotted pixels are on an integer-fF grid;
noninteger anchor points are audited separately and never interpolated into it.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
from threadpoolctl import threadpool_limits

from refined_pair_search import SETTINGS, search, validate, failures, quality, x_from_point, point_from_x, violations
from model import Basis, _transmon
from map_shunt_pairs import make_model, CHARGE_CONSTANT
from reference_devices import original_preset

BASE=Path(__file__).resolve().parent
OUT=BASE/"data"/"refined_shunt_map_2026_10_14"
COARSE=np.arange(44.,129.,2.)
UNIFORM=np.arange(36.,129.,1.)


def stamp(): return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self):
        OUT.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(OUT/"study.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS points(c1 REAL,c2 REAL,result TEXT,PRIMARY KEY(c1,c2))")
        self.db.execute("CREATE TABLE IF NOT EXISTS trials(id INTEGER PRIMARY KEY,stage TEXT,c1 REAL,c2 REAL,result TEXT,utc TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)")
        self.db.commit()
        self.points={(a,b):json.loads(r) for a,b,r in self.db.execute("SELECT c1,c2,result FROM points")}
    def meta(self,key,value=None):
        if value is None:
            row=self.db.execute("SELECT value FROM meta WHERE key=?",(key,)).fetchone()
            return json.loads(row[0]) if row else None
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)",(key,json.dumps(value)));self.db.commit()
    def save(self,result,stage):
        pair=(result["c1"],result["c2"])
        previous=self.points.get(pair)
        self.db.execute("INSERT INTO trials(stage,c1,c2,result,utc) VALUES(?,?,?,?,?)",
                        (stage,*pair,json.dumps(result),stamp()))
        keep=result
        if previous:
            priorpass=previous["status"] in ("screened_pass","validated_pass")
            newpass=result["status"] in ("screened_pass","validated_pass")
            if previous["status"]=="validated_pass" and result["status"]!="validated_pass": keep=previous
            elif priorpass and not newpass and result["status"]!="validation_failed": keep=previous
            elif not priorpass and not newpass and previous.get("point") and result.get("point"):
                if quality(previous["point"])>quality(result["point"]): keep=previous
        keep=dict(keep);keep["last_stage"]=stage
        self.points[pair]=keep
        self.db.execute("INSERT OR REPLACE INTO points VALUES(?,?,?)",(*pair,json.dumps(keep)))
        self.db.commit()
    def attempted(self,pair,stage):
        return self.db.execute("SELECT 1 FROM trials WHERE c1=? AND c2=? AND stage=? LIMIT 1",(*pair,stage)).fetchone() is not None
    def warm(self,pair,count=4):
        passing=[(key,r) for key,r in self.points.items() if r["status"] in ("screened_pass","validated_pass")]
        passing.sort(key=lambda item:(item[0][0]-pair[0])**2+(item[0][1]-pair[1])**2)
        seeds=[r["point"] for key,r in passing[:count]]
        prior=self.points.get(pair)
        if prior and prior.get("point") and prior["status"] not in ("screened_pass","validated_pass"):
            seeds.insert(0,prior["point"])
        return seeds
    def checkpoint(self,stage):
        counts={}
        for r in self.points.values(): counts[r["status"]]=counts.get(r["status"],0)+1
        state={"updated_utc":stamp(),"stage":stage,"points":len(self.points),"statuses":counts,
               "elapsed_hours":(time.time()-self.meta("start_epoch"))/3600}
        (OUT/"progress.json").write_text(json.dumps(state,indent=2))
        print(json.dumps(state),flush=True)


def frequency_window_seeds(ledger,pair):
    """Additional starts, never an exclusion test; exact local energy windows."""
    prior=ledger.points.get(pair,{}).get("point")
    if prior is None:
        warm=ledger.warm(pair,1)
        prior=warm[0] if warm else None
    if prior is None: return ledger.warm(pair)
    starts=[]
    with threadpool_limits(limits=1):
        for cg in (1.,4.,8.):
            m=make_model(*pair,2.,cg)
            ec=m.ec_ghz[:2]
            lo=max(4.02,*(_transmon(50.05*v,v,Basis())[0][1] for v in ec))
            hi=min(7.98,*(_transmon(24.99,v,Basis())[0][1] for v in ec))
            if lo>=hi: continue
            for fraction in (.5,.1,.9):
                f=lo+fraction*(hi-lo)
                c12=float(np.clip(.1*np.sqrt(np.prod(CHARGE_CONSTANT/ec))/f-cg*cg/m.device.Cr_fF,.1,5.))
                seed=dict(prior);seed["x"]=[float(f),0.,c12,cg]
                starts.append(seed)
    return starts+ledger.warm(pair,2)


def run_stage(ledger,pairs,stage,level,workers,deadline,retry=True,basis=None,window_starts=False):
    pending_pairs=[tuple(map(float,p)) for p in pairs if p[0]>p[1]]
    pending_pairs=list(dict.fromkeys(pending_pairs))
    pending_pairs=[p for p in pending_pairs
        if not ledger.attempted(p,stage)
        and ledger.points.get(p,{}).get("status") not in ("screened_pass","validated_pass")]
    # Grow outward from previous passing examples to make warm starts useful.
    def distance(pair):
        passing=[key for key,r in ledger.points.items() if r["status"] in ("screened_pass","validated_pass")]
        return min(((pair[0]-a)**2+(pair[1]-b)**2 for a,b in passing),default=(pair[0]-110)**2+(pair[1]-80)**2)
    pending_pairs.sort(key=distance)
    ledger.meta("stage",stage);ledger.checkpoint(stage)
    if not pending_pairs: return
    done=0;next_index=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={}
        def enqueue():
            nonlocal next_index
            while len(futures)<workers*2 and next_index<len(pending_pairs) and time.time()<deadline:
                pair=pending_pairs[next_index];next_index+=1
                task={"pair":pair,"level":level,"warm":frequency_window_seeds(ledger,pair) if window_starts else ledger.warm(pair)}
                if basis is not None: task["basis"]=basis
                futures[pool.submit(search,task)]=pair
        enqueue()
        while futures:
            future=next(as_completed(futures));pair=futures.pop(future)
            try: result=future.result()
            except Exception as exc:
                result={"c1":pair[0],"c2":pair[1],"status":"unresolved_error", "point":None,
                        "failures":[repr(exc)],"level":level,"attempts":[]}
            ledger.save(result,stage);done+=1
            if done%20==0: ledger.checkpoint(stage)
            enqueue()
    ledger.checkpoint(stage)


def passing_pairs(ledger):
    return [pair for pair,r in ledger.points.items() if r["status"] in ("screened_pass","validated_pass")]


def uniform_region(ledger,padding=4.):
    passed=passing_pairs(ledger)
    pairs=[]
    for c1 in UNIFORM:
        nearby=[b for a,b in passed if abs(a-c1)<=3.01]
        if not nearby: continue
        low=max(44.,np.floor(min(nearby)-padding));high=min(c1-1,np.ceil(max(nearby)+padding))
        pairs.extend((float(c1),float(c2)) for c2 in np.arange(low,high+1))
    return pairs


def retry_candidates(ledger,only_uniform=False):
    good=passing_pairs(ledger);pairs=[]
    for pair,r in ledger.points.items():
        if r["status"] in ("screened_pass","validated_pass"): continue
        if only_uniform and (pair[0]%1 or pair[1]%1): continue
        near=any(abs(a-pair[0])<=4.01 and abs(b-pair[1])<=4.01 for a,b in good)
        close=r.get("point") and np.max(violations(r["point"]))<.18
        if near or close: pairs.append(pair)
    return pairs


def validate_all(ledger,workers):
    tasks=[r for r in ledger.points.values() if r["status"]=="screened_pass"]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(validate,r):r for r in tasks}
        for i,future in enumerate(as_completed(futures),1):
            original=futures[future]
            try: result=future.result()
            except Exception as exc:
                result=dict(original);result["status"]="unresolved_validation_error";result["failures"]=[repr(exc)]
            ledger.save(result,"validation")
            if i%50==0: ledger.checkpoint("validation")
    ledger.checkpoint("validation")


def export(ledger):
    data=sorted(ledger.points.values(),key=lambda r:(r["c1"],r["c2"]))
    (OUT/"points.json").write_text(json.dumps(data,indent=2))
    uniforms=[r for r in data if r["c1"]%1==0 and r["c2"]%1==0]
    (OUT/"uniform_points.json").write_text(json.dumps(uniforms,indent=2))
    ledger.db.execute("PRAGMA wal_checkpoint(FULL)")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--hours",type=float,default=12.)
    parser.add_argument("--stage",choices=("all","coarse","uniform","global","validate","export"),default="all")
    args=parser.parse_args()
    ledger=Ledger()
    files=[BASE/"refined_pair_search.py",BASE/"model.py",BASE/"study_dispersive.py"]
    fingerprint=sha256(b"".join(path.read_bytes() for path in files)).hexdigest()
    if ledger.meta("fingerprint") and ledger.meta("fingerprint")!=fingerprint:
        raise RuntimeError("Numerical source fingerprint changed; use a separate ledger or an explicit audited migration")
    if ledger.meta("start_epoch") is None:
        ledger.meta("start_epoch",time.time());ledger.meta("deadline_epoch",time.time()+args.hours*3600)
        ledger.meta("fingerprint",fingerprint)
        ledger.meta("settings",dict(SETTINGS,coarse_spacing_fF=2.,final_spacing_fF=1.,
            coarse_capacitance_bounds_fF=[44.,128.],source_fingerprint=fingerprint,
            domain_note="90% gives initial ~44-116 fF; expanded high side covers physical EC>=150 and original A=118 anchor"))
    (OUT/"settings.json").write_text(json.dumps(ledger.meta("settings"),indent=2))
    deadline=ledger.meta("deadline_epoch")
    if args.stage in ("all","coarse"):
        # Revalidate old witnesses as admissible warm starts; the new search
        # and classification apply the updated constraints independently.
        legacy=BASE/"data"/"shunt_pair_map"/"validated.json"
        if legacy.exists() and not ledger.meta("bootstrap_done"):
            with threadpool_limits(limits=1):
                for row in json.loads(legacy.read_text()):
                    pair=(row["c1"],row["c2"])
                    if not pair[0]>pair[1]: continue
                    p=point_from_x(*pair,x_from_point(row["large_basis"]))
                    r={"c1":pair[0],"c2":pair[1],"status":"screened_pass" if not failures(p) else "no_witness_found",
                       "point":p,"level":"legacy_seed","failures":failures(p),"attempts":[]}
                    ledger.save(r,"bootstrap")
            ledger.meta("bootstrap_done",True)
        anchors=[(118.,78.),(115.,80.)]
        anchors += [(original_preset(name)["C1_shunt_fF"],original_preset(name)["C2_shunt_fF"]) for name in ("tdt1","tdt2")]
        run_stage(ledger,anchors,"anchors","deep",args.workers,deadline-1800)
        coarse=[(float(a),float(b)) for a in COARSE for b in COARSE if a>b]
        run_stage(ledger,coarse,"coarse_2fF","normal",args.workers,min(deadline-7200,time.time()+3*3600))
        run_stage(ledger,retry_candidates(ledger),"coarse_challenge","deep",args.workers,
                  min(deadline-6000,time.time()+2*3600))
    if args.stage in ("all","uniform"):
        requested=set()
        for round_index in range(3):
            pairs=uniform_region(ledger,4.)
            requested.update(pairs)
            run_stage(ledger,pairs,f"uniform_1fF_round_{round_index}","normal",args.workers,
                      min(deadline-3600,time.time()+2.5*3600))
            run_stage(ledger,retry_candidates(ledger,True),f"uniform_challenge_{round_index}","deep",args.workers,
                      min(deadline-3000,time.time()+1.5*3600))
            new=set(uniform_region(ledger,4.))
            if new.issubset(requested): break
        ledger.meta("uniform_requested",sorted(requested))
    if args.stage=="global":
        # With sufficient remaining budget, complete the entire strict
        # triangle at the same 1 fF spacing rather than leave coarse-only
        # exterior cells. The coarse high-EC branch touched C2=44 fF;
        # extend to 36 fF to resolve that boundary (4,278 centers).
        full=[(float(a),float(b)) for a in UNIFORM for b in UNIFORM if a>b]
        settings=ledger.meta('settings');settings['final_capacitance_bounds_fF']=[36.,128.]
        settings['domain_extension_reason']='Coarse above-resonator branch passes at initial C2=44 fF edge; final uniform grid extends to 36 fF.'
        ledger.meta('settings',settings)
        (OUT/'settings.json').write_text(json.dumps(settings,indent=2))
        run_stage(ledger,full,"uniform_global_1fF","normal",args.workers,
                  min(deadline-4500,time.time()+4*3600))
        # Give all remaining cells a second, broad search, including the
        # high-EC energy windows and possible above-resonator solutions.
        remaining=[p for p in full if ledger.points.get(p,{}).get("status") not in ("screened_pass","validated_pass")]
        ledger.meta("frequency_window_start_policy","Exact isolated-transmon f01 at EJ/EC=50.05 and EJ=24.99 GHz; Cg=1,4,8 fF; interior 10/50/90% starts. Prioritization only, not exclusion.")
        ledger.meta("global_runner_fingerprint",sha256(Path(__file__).read_bytes()).hexdigest())
        run_stage(ledger,retry_candidates(ledger,True),"global_energy_window_challenge","deep",args.workers,
                  min(deadline-3000,time.time()+2*3600),window_starts=True)
        run_stage(ledger,remaining,"global_energy_window_challenge","deep",args.workers,
                  deadline-3000,window_starts=True)
        ledger.meta("uniform_requested",full)
    if args.stage in ("all","validate","global"):
        validate_all(ledger,args.workers)
        broken=[pair for pair,r in ledger.points.items() if r["status"]=="validation_failed"]
        run_stage(ledger,broken,"large_basis_repair","normal",args.workers,deadline-900,
                  basis=SETTINGS["validation_basis"])
        validate_all(ledger,args.workers)
    export(ledger)
    ledger.meta("completed_stage",args.stage);ledger.checkpoint("numerical_stages_complete")


if __name__=="__main__": main()
