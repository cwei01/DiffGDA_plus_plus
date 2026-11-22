"""Atomic state, OS-owned retained locks and sticky resource enforcement."""

from experiment.efficiency import measured as _eff_measured
import json
import os
import threading
import time
import uuid
from pathlib import Path
import psutil
from gdabench.graph_ops import ResourceLimit


@_eff_measured("save")
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.flush();os.fsync(stream.fileno())
    for attempt in range(10):
        try:os.replace(temporary,path);return
        except PermissionError:
            if attempt==9:raise
            time.sleep(.05)


def process_identity():
    process=psutil.Process()
    return dict(pid=process.pid,process_create_time=process.create_time())


def read(path):
    for attempt in range(30):
        try:return json.loads(Path(path).read_text(encoding='utf-8'))
        except (PermissionError,json.JSONDecodeError):
            if attempt==29:raise
            time.sleep(.01)


def l2(value):
    """Frobenius norm without NumPy's broken BLAS dot path on this host."""
    import numpy as np
    a=np.asarray(value,dtype=np.float64)
    return float(np.sqrt(np.square(a).sum(dtype=np.float64)))


def process_alive(record):
    try:
        process=psutil.Process(record['pid'])
        return abs(process.create_time()-record['process_create_time'])<.01 and process.is_running()
    except (psutil.Error,KeyError):return False


class RunLock:
    """Lock file remains on disk; the OS releases its byte lock on process exit."""
    def __init__(self,path):self.path=Path(path);self.stream=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open('a+b')
        self.stream.seek(0,2)
        if self.stream.tell()==0:self.stream.write(b'0');self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            self.stream.close();self.stream=None
            raise RuntimeError('Another process owns run lock: '+str(self.path))
        save(self.path.with_suffix('.owner.json'),dict(**process_identity(),started_unix=time.time()))
        return self
    def __exit__(self,*exc):
        if self.stream is not None:
            self.stream.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(),fcntl.LOCK_UN)
            self.stream.close();self.stream=None


class Monitor:
    def __init__(self,path):
        import torch
        self.torch=torch
        self.process=psutil.Process();self.path=Path(path);self.started=time.perf_counter()
        self._phase='starting';self.phase_started=time.perf_counter();self.stage_seconds={}
        self._steps=0;self.stage_updates={};self.peak_rss=0
        self.min_available=2**63-1;self.failure=None;self.stop=threading.Event()
        self.pid=process_identity();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    @property
    def steps(self):return self._steps
    @property
    def phase(self):return self._phase
    @staticmethod
    def category(phase):
        if phase.startswith('feedback_'):return 'feedback'
        if phase.startswith('final_coupling'):return 'coupling'
        for stage in ('encoder','flow','head','structure'):
            if phase.startswith(stage):return stage
        return 'integration' if phase=='heun' else 'inference' if phase=='full_target_prediction' else phase
    @phase.setter
    def phase(self,value):
        now=time.perf_counter();old=self.category(self._phase)
        self.stage_seconds[old]=self.stage_seconds.get(old,0.)+now-self.phase_started
        self._phase=value;self.phase_started=now
    @steps.setter
    def steps(self,value):
        self._steps=value
        for stage in ('encoder','flow','head'):
            if self.phase.startswith(stage+'_') or self.phase==stage:
                self.stage_updates[stage]=int(value);break
    def restored(self,stage,value):self.stage_updates[stage]=int(value);self._steps=int(value)
    def check(self):
        torch=self.torch
        available=psutil.virtual_memory().available
        self.min_available=min(self.min_available,available);self.peak_rss=max(self.peak_rss,self.process.memory_info().rss)
        if available<4*1024**3:self.failure='System available memory below 4 GiB'
        if torch.cuda.is_available() and torch.cuda.max_memory_allocated()>7*1024**3:self.failure='GPU allocation exceeded 7 GiB'
        if self.failure:raise ResourceLimit(self.failure)
    def state(self):
        torch=self.torch
        timing=dict(self.stage_seconds);key=self.category(self.phase)
        timing[key]=timing.get(key,0.)+time.perf_counter()-self.phase_started
        return dict(**self.pid,phase=self.phase,optimizer_updates=self.steps,stage_updates=dict(self.stage_updates),
            heartbeat_unix=time.time(),elapsed_seconds=time.perf_counter()-self.started,
            stage_seconds=timing,
            peak_process_rss_bytes=self.peak_rss,minimum_available_ram_bytes=self.min_available,
            resource_violation=self.failure,gpu_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)
    def loop(self):
        while not self.stop.is_set():
            try:self.check()
            except ResourceLimit:pass
            save(self.path,self.state());self.stop.wait(5)
    def close(self):
        self.stop.set();self.thread.join();save(self.path,self.state())
