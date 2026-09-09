#!/usr/bin/env python3
"""Synthetic arithmetic/dispatch comparison; CPU by default, CUDA only by explicit flag."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"

import torch
from torch.utils._python_dispatch import TorchDispatchMode

_source_parser=argparse.ArgumentParser(add_help=False)
_default_source=next((path for path in Path(__file__).resolve().parents
                     if (path/'isaaclab/validate_mkii_fourbar.py').is_file()),Path.cwd())
_source_parser.add_argument('--source-dir',type=Path,default=_default_source)
_early,_unknown=_source_parser.parse_known_args()
ROOT = _early.source_dir.resolve(strict=True)
VALIDATOR = ROOT / "isaaclab/validate_mkii_fourbar.py"
KINEMATICS = ROOT / "configs/mkii_fourbar_v3_kinematics.json"
spec = importlib.util.spec_from_file_location("review_validator", VALIDATOR)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


def setup(num_envs, dtype, seed, device="cpu"):
    torch.manual_seed(seed)
    kinematics = json.loads(KINEMATICS.read_text())
    # Intentionally permute both lists; a proposed production implementation must
    # bind cached indices to runtime names, not rely on USD/config index order.
    bodies = list(kinematics["body_paths"])
    bodies = [bodies[i] for i in torch.randperm(len(bodies)).tolist()]
    joints = list(kinematics["tree_joint_names"])
    joints = [joints[i] for i in torch.randperm(len(joints)).tolist()]
    frames = []
    for name in kinematics["closure_joint_names"]:
        record = kinematics["joint_frames"][name]
        frames.append([(bodies.index(record[f"body{s}"]),
            torch.tensor(record[f"body{s}_from_hinge_matrix"], dtype=dtype)) for s in (0, 1)])
    relations = [(joints.index(name), joints.index(r["source_joint"]), r["multiplier"])
                 for name,r in kinematics["passive_relations"].items()]
    indices = torch.tensor([[i for i,_ in pair] for pair in frames])
    offsets = torch.stack([torch.stack([f[:3,3] for _,f in pair]) for pair in frames])
    axes = torch.stack([torch.stack([f[:3,2] for _,f in pair]) for pair in frames])
    relation_indices = torch.tensor([[a,b] for a,b,_ in relations])
    multipliers = torch.tensor([m for _,_,m in relations], dtype=dtype)
    rotations,_ = torch.linalg.qr(torch.randn(num_envs,31,3,3,dtype=dtype))
    args = (torch.randn(num_envs,31,3,dtype=dtype), rotations,
            torch.randn(num_envs,31,3,dtype=dtype), torch.randn(num_envs,31,3,dtype=dtype),
            torch.randn(num_envs,30,dtype=dtype),
            tuple(torch.randn(num_envs,1,3,dtype=dtype) for _ in range(31)))
    def transfer(value):
        if isinstance(value,torch.Tensor):
            return value.to(device)
        if isinstance(value,tuple):
            return tuple(transfer(v) for v in value)
        return value
    return transfer(args),[[(i,f.to(device)) for i,f in pair] for pair in frames],relations,transfer(
        (indices,offsets,axes,relation_indices,multipliers))


def current(args,frames,relations):
    pos,rot,lin,ang,qdot,sensor_forces = args
    pin_velocity = validator.closure_relative_point_velocities(rot,lin,ang,frames)
    passive_velocity = validator.passive_joint_velocity_residuals(qdot,relations)
    point_error,axis_error = [],[]
    # These statements match the corresponding PhysicalMetrics.capture block.
    for pair in frames:
        states = [(pos[:,i] + torch.einsum("nij,j->ni",rot[:,i],f[:3,3]),
                   torch.einsum("nij,j->ni",rot[:,i],f[:3,2])) for i,f in pair]
        point_error.append(torch.linalg.vector_norm(states[0][0]-states[1][0],dim=-1))
        axis_error.append(torch.linalg.vector_norm(states[0][1]-states[1][1],dim=-1))
    forces = torch.stack([torch.linalg.vector_norm(value,dim=-1).amax(-1) for value in sensor_forces],dim=1)
    return torch.stack(point_error,1),torch.stack(axis_error,1),pin_velocity,passive_velocity,forces


def proposed(args,cached):
    pos,rot,lin,ang,qdot,sensor_forces = args
    indices,offsets,axes,relation_indices,multipliers = cached
    flat_indices = indices.flatten()
    batch = pos.shape[0]
    r = rot.index_select(1,flat_indices).reshape(batch,6,2,3,3)
    offset_w = torch.einsum("nlbij,lbj->nlbi",r,offsets)
    axis_w = torch.einsum("nlbij,lbj->nlbi",r,axes)
    centers = pos.index_select(1,flat_indices).reshape(batch,6,2,3)+offset_w
    velocities = (lin.index_select(1,flat_indices).reshape(batch,6,2,3)
        + torch.cross(ang.index_select(1,flat_indices).reshape(batch,6,2,3),offset_w,dim=-1))
    point_error = torch.linalg.vector_norm(centers[:,:,0]-centers[:,:,1],dim=-1)
    axis_error = torch.linalg.vector_norm(axis_w[:,:,0]-axis_w[:,:,1],dim=-1)
    passive_velocity = (qdot.index_select(1,relation_indices[:,0])
                        - multipliers*qdot.index_select(1,relation_indices[:,1]))
    # Native sensor acquisition is deliberately unchanged, including all 31
    # sensor views. This only batches the downstream norm over each XYZ vector.
    forces = torch.linalg.vector_norm(torch.stack(sensor_forces,1),dim=-1).amax(-1)
    return point_error,axis_error,velocities[:,:,1]-velocities[:,:,0],passive_velocity,forces


class Dispatches(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.counts[str(func)] += 1
        return func(*args,**(kwargs or {}))


def synchronize(device):
    if torch.device(device).type=="cuda":
        torch.cuda.synchronize(device)


def time_call(function,device="cpu",repeats=7,loops=100):
    for _ in range(20):
        function()
    synchronize(device)
    values=[]
    for _ in range(repeats):
        synchronize(device)
        start=time.perf_counter()
        for _ in range(loops):
            function()
        synchronize(device)
        values.append((time.perf_counter()-start)/loops)
    return {"median_seconds":statistics.median(values),"min_seconds":min(values),
            "max_seconds":max(values),"repeats":repeats,"calls_per_repeat":loops}


def compare(left,right):
    rows=[]
    for name,a,b in zip(("point_error_m","axis_chord","pin_velocity_m_s","passive_velocity_rad_s","force_norm_n"),left,right):
        torch.testing.assert_close(a,b,rtol=16*torch.finfo(a.dtype).eps,atol=16*torch.finfo(a.dtype).eps,equal_nan=True)
        assert torch.equal(torch.isfinite(a),torch.isfinite(b))
        assert torch.equal(torch.isnan(a),torch.isnan(b))
        assert torch.equal(torch.isposinf(a),torch.isposinf(b))
        assert torch.equal(torch.isneginf(a),torch.isneginf(b))
        mask=torch.isfinite(a)&torch.isfinite(b)
        error=float((a[mask]-b[mask]).abs().max()) if mask.any() else 0.
        ulp=None
        if a.dtype==torch.float32:
            def ordered(value):
                bits=value.contiguous().view(torch.int32).to(torch.int64)
                return torch.where(bits<0,0x80000000-(bits&0x7fffffff),0x80000000+bits)
            ulp=int((ordered(a[mask])-ordered(b[mask])).abs().max()) if mask.any() else 0
        rows.append({"name":name,"finite_values_bitwise_equal":torch.equal(
                         a[mask].contiguous().view(torch.uint8),b[mask].contiguous().view(torch.uint8)),
                     "nonfinite_classification_equal":True,"max_abs_error":error,"max_float32_ulp_error":ulp})
    # These are the actual point/axis/contact comparisons; they do not turn
    # unphysical synthetic data into a physical-admission result.
    for a,b in zip(left[:2],right[:2]):
        threshold=.0001 if a is left[0] else math.radians(.1)
        assert torch.equal(a>threshold,b>threshold)
    assert torch.equal(left[4]>1.,right[4]>1.)
    return rows


def boundary_cases(device):
    inputs,frames,relations,cached=setup(32,torch.float32,seed=9001,device=device)
    pos,rot,lin,ang,qdot,forces=inputs
    ids,offsets,axes,relation_ids,multipliers=cached
    # An explicit zero-offset synthetic fixture lets positions represent the
    # gate value and its immediate float32 neighbours without CAD-offset
    # cancellation masking the boundary. Real CAD frames are tested separately.
    frames=[[(i,torch.eye(4,dtype=torch.float32,device=device)) for i,_ in pair] for pair in frames]
    offsets=torch.zeros_like(offsets)
    axes=torch.zeros_like(axes);axes[...,2]=1
    rot=torch.eye(3,dtype=torch.float32,device=device).expand(32,31,3,3).clone()
    pos=torch.zeros_like(pos)
    threshold=torch.tensor(.0001,dtype=torch.float32,device=device)
    neighbours=torch.stack((torch.nextafter(threshold,torch.full_like(threshold,float('-inf'))),threshold,
                             torch.nextafter(threshold,torch.full_like(threshold,float('inf')))))
    labels=neighbours[torch.arange(32,device=device)%3]
    pos[:,ids[:,1],0]=labels[:,None]
    forces=tuple(torch.zeros_like(value) for value in forces)
    force_threshold=torch.tensor(1.,dtype=torch.float32,device=device)
    force_values=torch.stack((torch.nextafter(force_threshold,torch.full_like(force_threshold,float('-inf'))),force_threshold,
                             torch.nextafter(force_threshold,torch.full_like(force_threshold,float('inf')))))
    for value in forces:value[:,0,0]=force_values[torch.arange(32,device=device)%3]
    args=(pos,rot,lin,ang,qdot,forces)
    cached=(ids,offsets,axes,relation_ids,multipliers)
    old,new=current(args,frames,relations),proposed(args,cached)
    result={"fixture":"synthetic zero pin offsets; float32 gate and immediate neighbours",
            "closure_boundary_values_m":neighbours.tolist(),"force_boundary_values_n":force_values.tolist(),
            "comparisons":compare(old,new),"point_gate_flags_equal":True,"force_gate_flags_equal":True}
    assert (old[0]>.0001).any() and (~(old[0]>.0001)).any()
    assert (old[4]>1.).any() and (~(old[4]>1.)).any()
    # Preserve rejection behaviour for unloaded/bad sensor records as well.
    forces[0][0,0,0]=float('nan');forces[1][1,0,0]=float('inf')
    result["nonfinite_force_comparisons"]=compare(current(args,frames,relations),proposed(args,cached))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--source-dir",type=Path,default=ROOT,help="Read-only robot source; defaults to surrounding checkout")
    p.add_argument("--device",choices=("cpu","cuda","cuda:0"),default="cpu",
                   help="CUDA launches GPU tensor work only when explicitly requested")
    args=p.parse_args()
    if args.report.exists():
        raise ValueError("Do not overwrite benchmark evidence")
    device="cuda:0" if args.device=="cuda" else args.device
    if device=="cpu":
        if torch.cuda.is_initialized():
            raise RuntimeError("CPU review cannot inherit an initialized CUDA context")
    elif not torch.cuda.is_available():
        raise RuntimeError("Explicit CUDA benchmark requested but CUDA is unavailable; no fallback")
    torch.set_num_threads(1)
    results=[]
    for dtype in (torch.float32,torch.float64):
        for count in (32,512):
            inputs,frames,relations,cached=setup(count,dtype,seed=19+count,device=device)
            old=lambda:current(inputs,frames,relations)
            new=lambda:proposed(inputs,cached)
            differences=compare(old(),new())
            rows={"num_envs":count,"dtype":str(dtype),"comparisons":differences,
                  "permuted_body_indices_6_by_2":cached[0].cpu().tolist(),
                  "permuted_joint_relation_indices_12_by_2":cached[3].cpu().tolist()}
            randomized=[]
            for seed in range(10):
                random_inputs,random_frames,random_relations,random_cached=setup(count,dtype,seed,device=device)
                randomized.append({"seed":seed,
                    "permuted_body_indices_6_by_2":random_cached[0].cpu().tolist(),
                    "permuted_joint_relation_indices_12_by_2":random_cached[3].cpu().tolist(),
                    "comparisons":compare(current(random_inputs,random_frames,random_relations),
                                          proposed(random_inputs,random_cached))})
            rows["randomized_actual_cad_frames"]=randomized
            if dtype==torch.float32:
                rows["current_timing"]=time_call(old,device)
                rows["proposed_timing"]=time_call(new,device)
                for name,fn in (("current",old),("proposed",new)):
                    mode=Dispatches()
                    with mode:fn()
                    synchronize(device)
                    rows[name+"_aten_dispatches"]={"total":sum(mode.counts.values()),"operations":dict(sorted(mode.counts.items()))}
            results.append(rows)
    report={"schema":"hexapod.metrics_batch_review.v1","pass":True,"physics_qualification":False,
        "torch":torch.__version__,"python":platform.python_version(),"machine":platform.machine(),
        "system":platform.system(),"threads":torch.get_num_threads(),"cuda_initialized":torch.cuda.is_initialized(),
        "device":device,"cuda_version":torch.version.cuda,
        "source_root":str(ROOT),
        "cuda_device_name":torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
        "float32_matmul_precision":torch.get_float32_matmul_precision(),
        "timing_sync":"CUDA synchronize before and after each timed block" if device.startswith('cuda') else "CPU synchronous calls",
        "scope":"synthetic arithmetic only; no native sensors, full PhysicalMetrics, PhysX, or training",
        "probe_path":str(Path(__file__).resolve()),"probe_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (VALIDATOR,KINEMATICS,ROOT/'packages/hexapod_env/hexapod_env/tasks/mkii_fourbar_v1/env.py')},
        "results":results,"gate_boundary_fixture":boundary_cases(device)}
    assert report["cuda_initialized"]==device.startswith('cuda')
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({"pass":True,"report":str(args.report),"randomized_case_count":40,
        "device":device,"timing_and_dispatch":[{k:v for k,v in row.items() if k in ('num_envs','dtype','current_timing','proposed_timing')}
                               for row in results]},indent=2))


if __name__=="__main__":
    try:
        main()
    except Exception as error:
        # Preserve a failed standalone comparison as evidence, including if an
        # explicitly requested CUDA run has no CUDA backend. Never overwrite.
        import sys
        argv=sys.argv[1:]
        if argv.count('--report')==1 and argv.index('--report')+1<len(argv):
            output=Path(argv[argv.index('--report')+1])
            if not output.exists() and output.parent.is_dir():
                failure={'schema':'hexapod.metrics_batch_review.v1','pass':False,'physics_qualification':False,
                    'error':f'{type(error).__name__}: {error}',
                    'device_requested':argv[argv.index('--device')+1] if '--device' in argv else 'cpu',
                    'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'torch':torch.__version__,'cuda_initialized':torch.cuda.is_initialized()}
                output.write_text(json.dumps(failure,indent=2)+'\n')
        raise
