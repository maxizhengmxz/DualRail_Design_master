"""Continue from the checkpointed run through audit and report rendering."""
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
from run_selected_robustness import OUT,BASE


def main():
    def state(stage,**kwargs): (OUT/'continuation.json').write_text(json.dumps(dict(stage=stage,**kwargs),indent=2))
    state('waiting_for_numerical_completion')
    # Hold a process-scoped sleep-prevention request while computation runs.
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        while True:
            if (OUT/'progress.json').exists():
                p=json.loads((OUT/'progress.json').read_text())
                if p['stage']=='numerical_complete': break
            time.sleep(5)
        with (OUT/'completion.log').open('a',encoding='utf-8') as log:
            state('audit')
            subprocess.run([sys.executable,str(BASE/'audit_selected_robustness.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
            state('report')
            subprocess.run([sys.executable,str(BASE/'report_selected_robustness.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
        state('awaiting_visual_review')
    except Exception as exc:
        state('completion_error',error=repr(exc));raise
    finally: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__': main()
