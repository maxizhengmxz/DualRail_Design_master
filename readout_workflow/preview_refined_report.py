"""Render a clearly separated layout proof from validated live candidates."""
from pathlib import Path
import json
import pymupdf
import report_refined_study as report
from refined_pair_search import validate


def main():
    rows,meta,trials=report.load()
    screened=[r for r in rows if r['status'] in ('screened_pass','validated_pass')]
    selected={}
    for key in ('c1','c2'):
        for reverse in (False,True):
            r=sorted(screened,key=lambda r:r[key],reverse=reverse)[0]
            selected[(r['c1'],r['c2'])]=r
    for c1 in range(85,130,5):
        options=[r for r in screened if abs(r['c1']-c1)<=1 and r['c1']%1==0 and r['c2']%1==0]
        if options:
            for r in (min(options,key=lambda r:r['c2']),max(options,key=lambda r:r['c2'])):
                selected[(r['c1'],r['c2'])]=r
    checked={pair:validate(r) for pair,r in selected.items()}
    rows=[checked.get((r['c1'],r['c2']),r) for r in rows]
    meta['uniform_requested']=[(float(a),float(b)) for a in range(44,129) for b in range(44,a)]
    report.OUT=Path(__file__).resolve().parents[1]/'tmp'/'refined_layout_proof'
    report.OUT.mkdir(parents=True,exist_ok=True)
    uniform,stats,status=report.figures(rows,meta)
    result=report.summary(rows,uniform,meta,trials,stats,status)
    report.pdf_report(result,rows)
    sizes={}
    for filename in report.OUT.glob('*.pdf'):
        with pymupdf.open(filename) as doc:
            sizes[filename.name]=len(doc)
            for i,page in enumerate(doc):
                page.get_pixmap(matrix=pymupdf.Matrix(1.25,1.25)).save(report.OUT/f'{filename.stem}_{i+1}.png')
    print(json.dumps({'LAYOUT_PROOF_ONLY':True,'pages':sizes,'validated_examples':len(uniform),'basis_failures':[(p,r['failures']) for p,r in checked.items() if r['failures']]},indent=2))


if __name__=='__main__': main()
