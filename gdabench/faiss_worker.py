"""CPU-only HNSW worker. Execute with the dedicated official Conda Python."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--request',type=Path,required=True)
    args=parser.parse_args()
    request=json.loads(args.request.read_text(encoding='utf-8'))
    directory=args.request.parent.resolve()
    if any(not (directory/name).resolve().is_relative_to(directory)
           for name in ('database.npy','queries.npy','neighbors.npy','response.json')):
        raise ValueError('IPC paths must stay in request directory')
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['MKL_NUM_THREADS']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    dll_handles=[]
    if os.name=='nt':
        for path in (Path(sys.prefix)/'Library'/'bin',Path(sys.prefix)/'DLLs'):
            if path.exists():dll_handles.append(os.add_dll_directory(str(path)))
    import numpy as np
    import faiss
    faiss.omp_set_num_threads(1)
    database=np.load(directory/'database.npy',mmap_mode='r',allow_pickle=False)
    queries=np.load(directory/'queries.npy',mmap_mode='r',allow_pickle=False)
    if database.ndim!=2 or queries.ndim!=2 or database.shape[1]!=queries.shape[1]:
        raise ValueError('Invalid HNSW input dimensions')
    if not np.isfinite(database).all() or not np.isfinite(queries).all():
        raise ValueError('Nonfinite HNSW input')
    k=min(int(request['k']),len(database))
    if k<1:raise ValueError('Empty index or nonpositive k')
    index=faiss.IndexHNSWFlat(database.shape[1],16)
    index.hnsw.efConstruction=200
    index.hnsw.efSearch=128
    rng=faiss.RandomGenerator(int(request['seed']))
    index.hnsw.rng=rng
    def block(array):
        value=np.array(array,dtype=np.float32,order='C',copy=True)
        if request['normalize']:faiss.normalize_L2(value)
        return value
    started=time.perf_counter()
    # Preserve fixed insertion order. Batches avoid a second full feature copy.
    for begin in range(0,len(database),8192):index.add(block(database[begin:begin+8192]))
    build_seconds=time.perf_counter()-started
    out=np.lib.format.open_memmap(directory/'neighbors.npy',mode='w+',dtype=np.int64,shape=(len(queries),k))
    started=time.perf_counter()
    for begin in range(0,len(queries),4096):
        _,neighbors=index.search(block(queries[begin:begin+4096]),k)
        if (neighbors<0).any():raise ValueError('HNSW returned missing candidates')
        out[begin:begin+len(neighbors)]=neighbors
    out.flush()
    response=dict(status='completed',faiss_version=faiss.__version__,numpy_version=np.__version__,
        python=sys.version,python_executable=sys.executable,shape=list(out.shape),
        normalize=bool(request['normalize']),seed=int(request['seed']),
        M=16,efConstruction=200,efSearch=128,threads=1,
        build_seconds=build_seconds,query_seconds=time.perf_counter()-started,
        database_sha256=sha(directory/'database.npy'),queries_sha256=sha(directory/'queries.npy'),
        neighbors_sha256=sha(directory/'neighbors.npy'))
    (directory/'response.json').write_text(json.dumps(response,indent=2),encoding='utf-8')
    print(json.dumps(response),flush=True)


if __name__=='__main__':main()
