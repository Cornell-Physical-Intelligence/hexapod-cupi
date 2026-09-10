"""Read-only interpreter/library/storage inventory; no analysis, writes or GPU imports."""
import hashlib,json,os,platform,shutil,sys
from pathlib import Path
import numpy as np

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=Path('/home/orionh/HEXAPOD_runs/mock_length_study_20260909')
print(json.dumps({'python_executable':sys.executable,'python_realpath':os.path.realpath(sys.executable),'python_sha256':digest(os.path.realpath(sys.executable)),'python_version':sys.version,'machine':platform.machine(),'numpy_version':np.__version__,'numpy_module':np.__file__,'numpy_module_sha256':digest(np.__file__),'numpy_smoke_finite':bool(np.isfinite(np.array([0.,1.])).all()),'numpy_float32_reduction':float(np.asarray([1.,2.,3.],dtype=np.float32).mean()),'no_bytecode':sys.dont_write_bytecode,'torch_or_cuda_imported':any(k=='torch' or k.startswith(('torch.','cupy','isaac')) for k in sys.modules),'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','CUDA_VISIBLE_DEVICES')},'remote_disk_usage_bytes':dict(zip(('total','used','free'),shutil.disk_usage(base))),'meminfo':{k:v.strip() for line in Path('/proc/meminfo').read_text().splitlines() for k,v in [line.split(':',1)] if k in ('MemTotal','MemAvailable')},'scope':'Read-only CPU NumPy import and environment inventory. No analyzer execution, GPU allocation, process control, installation or remote writes.'},indent=2))
