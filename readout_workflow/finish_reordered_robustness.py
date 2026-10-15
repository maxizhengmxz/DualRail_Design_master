"""Durable numerical completion, audit, and rendered comparison report."""
import ctypes
import json
import subprocess
import sys
import time
from run_reordered_robustness import OUT,BASE


def main():
    def state(stage,**kwargs): (OUT/'continuation.json').write_text(json.dumps(dict(stage=stage,**kwargs),indent=2))
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        state('waiting_for_numerical_completion')
        while json.loads((OUT/'progress.json').read_text())['stage']!='numerical_complete': time.sleep(5)
        with (OUT/'completion.log').open('a',encoding='utf-8') as log:
            state('audit');subprocess.run([sys.executable,str(BASE/'audit_reordered_robustness.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
            state('report');subprocess.run([sys.executable,str(BASE/'report_reordered_robustness.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
        state('awaiting_visual_review')
    except Exception as exc: state('error',error=repr(exc));raise
    finally: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__': main()
