"""Durable continuation after the first search; no concurrent ledger writers."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parent
OUT=BASE/'data'/'refined_shunt_map_2026_10_14'


def state(stage,**kwargs):
    (OUT/'continuation.json').write_text(json.dumps(dict(stage=stage,**kwargs),indent=2))


def main():
    state('waiting_for_first_pipeline')
    while True:
        db=sqlite3.connect(OUT/'study.sqlite')
        row=db.execute("SELECT value FROM meta WHERE key='completed_stage'").fetchone()
        db.close()
        if row and json.loads(row[0]) in ('all','global'): break
        time.sleep(5)
    with (OUT/'continuation.log').open('a',encoding='utf-8') as log:
        if json.loads(row[0])!='global':
            state('global_uniform_search')
            subprocess.run([sys.executable,str(BASE/'run_refined_study.py'),'--stage','global','--workers','4'],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        state('final_audit')
        subprocess.run([sys.executable,str(BASE/'audit_refined_study.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
        state('figures_and_report')
        subprocess.run([sys.executable,str(BASE/'report_refined_study.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
    import pymupdf
    rendered=OUT/'rendered';rendered.mkdir(exist_ok=True)
    for filename in ('refined_feasible_maps.pdf','refined_charging_energies.pdf','refined_constraint_maps.pdf','refined_study_report.pdf'):
        with pymupdf.open(OUT/filename) as doc:
            for index,page in enumerate(doc):
                page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(rendered/f'{Path(filename).stem}_{index+1}.png')
    state('awaiting_visual_review')


if __name__=='__main__': main()
