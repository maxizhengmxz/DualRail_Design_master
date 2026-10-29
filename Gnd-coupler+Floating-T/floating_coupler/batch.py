"""Bounded independent process jobs, atomic checkpoints and exact resume keys.

No numerical approximation. Reuses the toy's BLAS=1 worker convention. Task
payloads and a source signature determine identity; process count does not.
An exclusive lock prevents duplicate writers. Failures are not success records.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import multiprocessing
import os
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def source_signature():
    root=Path(__file__).resolve().parents[2]
    files=[p for p in Path(__file__).resolve().parents[1].rglob('*.py') if 'output' not in p.parts]
    files+=list((root/'core').glob('*.py'))
    files+=[root/'transmon-based CZ/transmon_cz/channel_policy.py']
    return digest([(str(p.relative_to(root)),hashlib.sha256(p.read_bytes()).hexdigest())
                   for p in sorted(files)])


def _initialize():
    from core.runtime import configure_single_threaded_blas
    configure_single_threaded_blas()


def _write(path,value):
    temporary=path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


@contextmanager
def _lock(folder):
    path=folder/'writer.lock'
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    try:
        os.write(fd,str(os.getpid()).encode());os.close(fd)
        yield
    finally:path.unlink()


def run_tasks(worker,tasks,folder,workers=1,signature=None):
    """Resume completed independent jobs, preserving input order in the result."""
    if not isinstance(workers,int) or workers<1:raise ValueError('workers must be a positive integer')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    ids=[digest(task) for task in tasks]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate task payloads are not independent jobs')
    identity=dict(source=signature or source_signature(),task_ids=ids,
                  worker=f'{worker.__module__}.{worker.__qualname__}')
    with _lock(folder):
        manifest=folder/'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text())!=identity:
                raise ValueError('Source/inputs changed; use a fresh output directory')
        else:_write(manifest,identity)
        results={};pending=[]
        for key,task in zip(ids,tasks):
            path=folder/(key+'.json')
            if path.exists():results[key]=json.loads(path.read_text())['result']
            else:pending.append((key,task))
        if workers==1:
            _initialize()
            for key,task in pending:
                value=worker(task);_write(folder/(key+'.json'),dict(result=value));results[key]=value
        elif pending:
            with ProcessPoolExecutor(max_workers=min(workers,len(pending)),
                    mp_context=multiprocessing.get_context('spawn'),initializer=_initialize) as pool:
                futures={pool.submit(worker,task):key for key,task in pending}
                for future in as_completed(futures):
                    key=futures[future];value=future.result()
                    _write(folder/(key+'.json'),dict(result=value));results[key]=value
        return [results[key] for key in ids]
