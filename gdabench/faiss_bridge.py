"""File-based HNSW IPC: training Python never loads another environment's DLLs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=Path(os.environ.get('GDABENCH_FAISS_PREFIX',str(Path(os.environ.get('GDABENCH_FAISS_PYTHON',__import__('sys').executable)).parent)))
PYTHON=Path(os.environ.get('GDABENCH_FAISS_PYTHON',__import__('sys').executable))
WORKER=ROOT/'gdabench'/'faiss_worker.py'
CACHE=Path(os.environ.get('GDABENCH_FAISS_CACHE',str(ROOT/'runs'/'.gdabench_faiss_cache')))


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def knn(database,queries,k,normalize,seed=20261025,resource_check=None):
    if not PYTHON.exists():raise RuntimeError('Dedicated official Conda Faiss runtime missing')
    database=np.asarray(database,dtype=np.float32,order='C')
    queries=np.asarray(queries,dtype=np.float32,order='C')
    if database.ndim!=2 or queries.ndim!=2 or database.shape[1]!=queries.shape[1]:
        raise ValueError('HNSW requires matching feature dimensions')
    if not len(database) or not len(queries) or k<1:raise ValueError('Empty HNSW domain')
    if not np.isfinite(database).all() or not np.isfinite(queries).all():raise ValueError('Nonfinite HNSW input')
    version_files=sorted((RUNTIME/'conda-meta').glob('faiss-cpu-*.json'))
    version=json.loads(version_files[0].read_text(encoding='utf-8')) if version_files else dict(version='external',build=str(PYTHON))
    config=dict(k=min(k,len(database)),normalize=bool(normalize),seed=int(seed),
        shapes=[list(database.shape),list(queries.shape)],M=16,efConstruction=200,efSearch=128,
        threads=1,package=version['version']+'_'+version['build'],worker_sha256=file_sha(WORKER))
    h=hashlib.sha256(json.dumps(config,sort_keys=True).encode())
    h.update(memoryview(database).cast('B'));h.update(memoryview(queries).cast('B'))
    directory=CACHE/h.hexdigest()
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'response.json'
    if not marker.exists():
        if resource_check is not None:resource_check()
        # The content-addressed directory can contain arrays left by an
        # interrupted worker. Reuse those immutable inputs rather than opening
        # them for overwrite (which can fail on Windows while mappings linger).
        db_path,query_path=directory/'database.npy',directory/'queries.npy'
        if not db_path.exists():np.save(db_path,database,allow_pickle=False)
        if not query_path.exists():np.save(query_path,queries,allow_pickle=False)
        for path,expected in ((db_path,database),(query_path,queries)):
            cached=np.load(path,mmap_mode='r',allow_pickle=False)
            if cached.shape!=expected.shape or not np.array_equal(cached,expected):
                raise ValueError('Content-addressed Faiss IPC input mismatch: '+path.name)
        (directory/'request.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        env=dict(os.environ)
        env['PATH']=os.pathsep.join([str(RUNTIME),str(RUNTIME/'Library'/'bin'),str(RUNTIME/'Scripts'),env.get('PATH','')])
        env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        # Child must use its own Conda prefix, never the caller's Python packages.
        env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
        with (directory/'worker.log').open('w',encoding='utf-8') as log:
            command=[str(PYTHON),'-u',str(WORKER),'--request',str(directory/'request.json')]
            if resource_check is None:
                process=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            else:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        resource_check();time.sleep(1)
                except BaseException:
                    process.terminate();process.wait(timeout=20)
                    raise
        if process.returncode!=0:
            failure=dict(status='worker_failed',exit_code=process.returncode,
                exit_hex=hex(process.returncode&0xffffffff),log=str(directory/'worker.log'))
            (directory/'failure.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
            raise RuntimeError('Faiss isolated worker failed: '+json.dumps(failure))
    response=json.loads(marker.read_text(encoding='utf-8'))
    if response['neighbors_sha256']!=file_sha(directory/'neighbors.npy'):
        raise ValueError('Cached HNSW candidates hash mismatch')
    result=np.load(directory/'neighbors.npy',mmap_mode='r',allow_pickle=False)
    if result.shape!=(len(queries),config['k']) or (result<0).any() or (result>=len(database)).any():
        raise ValueError('Invalid HNSW candidate output')
    return result,response
