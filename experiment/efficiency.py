"""Opt-in nested timing. No tensor operations or RNG changes when disabled."""
import functools
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_stack = []
_exclusive = {}
_origin = time.perf_counter()
_overhead = 0.0
EXCLUDED = {'evaluation', 'evaluation_prediction', 'artifact_save', 'save',
            'data', 'data_loading', 'checkpoint_replay', 'evolution_replay'}


def enabled():
    return bool(os.environ.get('PP_EFFICIENCY_EVENTS'))


def reset_clock():
    global _origin, _overhead
    if _stack:
        raise RuntimeError('Cannot reset inside a timing stage')
    _exclusive.clear()
    _origin, _overhead = time.perf_counter(), 0.


def sync():
    torch = sys.modules.get('torch')
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def emit(record):
    global _overhead
    path = os.environ.get('PP_EFFICIENCY_EVENTS')
    if not path:
        return
    start = time.perf_counter()
    record = dict(record, pid=os.getpid(), elapsed_wall_seconds=start-_origin)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
    _overhead += time.perf_counter()-start


@contextmanager
def stage(name):
    if not enabled():
        yield
        return
    sync()
    frame = [name, time.perf_counter(), 0.0, _overhead, 0.0]
    _stack.append(frame)
    try:
        yield
    finally:
        sync()
        elapsed = time.perf_counter()-frame[1]
        _stack.pop()
        internal_overhead = _overhead-frame[3]
        own = max(0., elapsed-frame[2]-(internal_overhead-frame[4]))
        _exclusive[name] = _exclusive.get(name, 0.)+own
        if _stack:
            _stack[-1][2] += elapsed
            _stack[-1][4] += internal_overhead
        emit(dict(type='stage', name=name, inclusive_seconds=elapsed,
                  exclusive_seconds=own, parent=_stack[-1][0] if _stack else None))


def measured(name, evaluation=False):
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if not enabled():
                return fn(*args, **kwargs)
            with stage(name):
                value = fn(*args, **kwargs)
            if evaluation and args:
                observer = args[0]
                record = getattr(observer, 'final', None)
                if isinstance(record, dict):
                    # Native protocol is retained; source holdout is never relabeled target.
                    emit(dict(type='evaluation', metrics=record,
                              domain=getattr(observer, 'domain', 'target'),
                              predictor_kind='native_task_classifier',
                              elapsed_method_seconds=max(0., time.perf_counter()-_origin
                                  -sum(v for k,v in _exclusive.items()
                                       if k in EXCLUDED)-_overhead)))
            return value
        return wrapped
    return decorate


def summary():
    return dict(exclusive_stage_seconds=dict(_exclusive),
                net_method_seconds=sum(v for k,v in _exclusive.items() if k not in EXCLUDED),
                unclassified_seconds=_exclusive.get('run_unclassified', 0.),
                instrumentation_write_seconds=_overhead,
                elapsed_wall_seconds=time.perf_counter()-_origin)
