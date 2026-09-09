#!/usr/bin/env python3
"""Complete-row synthetic comparison with installed SDK math; CPU by default."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import runpy
import sys
import time
import types
import unittest

os.environ['OMP_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name,path):
    loader=importlib.machinery.SourceFileLoader(name,str(path))
    spec=importlib.util.spec_from_loader(name,loader)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=HERE/'baseline_validate_mkii_fourbar.py.txt')
    p.add_argument('--candidate',type=Path,default=HERE/'proposed_validate_mkii_fourbar.py.txt')
    p.add_argument('--device',choices=('cpu','cuda','cuda:0'),default='cpu')
    p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    source=args.source_dir.resolve(strict=True)
    output=args.report.resolve()
    if output.exists():raise ValueError('Refusing to overwrite comparison evidence')
    if not output.parent.is_dir():raise ValueError('Report parent must already exist')
    device='cuda:0' if args.device=='cuda' else args.device
    report={'schema':'hexapod.complete_metrics_comparison.v1','pass':False,'physics_qualification':False,
            'source':str(source),'device':device,'errors':[],'native_sensor_acquisition_executed':False,
            'isaac_sim_started':False,'source_or_runtime_edited':False}
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1)
        if device=='cpu' and torch.cuda.is_initialized():
            raise ValueError('Default CPU comparison cannot inherit CUDA')
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('Explicit CUDA requested but unavailable; no fallback')
        sys.path[:0]=[str(source/'tools'),str(source/'isaaclab'),str(source/'packages/hexapod_core'),
                      str(source/'packages/hexapod_env')]
        from mkii_training_contract import identity
        import mkii_fourbar_kinematics as kin
        from audit_mkii_stance import _rotation
        contract_before=identity(source)
        report['functional_source_identity']=contract_before
        provenance=json.loads((HERE/'sdk_math_provenance.json').read_text())
        original=HERE/'installed_isaaclab_math.py.txt'
        extracted=HERE/'installed_matrix_from_quat.py'
        if digest(original)!=provenance['source_sha256'] or digest(extracted)!=provenance['extracted_module_sha256']:
            raise ValueError('Installed SDK source bytes differ from provenance')
        text=original.read_text();nodes=ast.parse(text).body
        for record in provenance['extracted_functions']:
            node=next(n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==record['name'])
            start=min([node.lineno]+[d.lineno for d in node.decorator_list])
            block='\n'.join(text.splitlines()[start-1:node.end_lineno])+'\n'
            if hashlib.sha256(block.encode()).hexdigest()!=record['sha256'] or block not in extracted.read_text():
                raise ValueError('SDK function body/decorator was changed during extraction')
        sdk=load('complete_metrics_installed_math',extracted)
        half=math.sqrt(.5)
        q=torch.tensor([[0.,0.,0.,1.],[0.,0.,half,half]],device=device)
        expected=torch.tensor([[[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],
                               [[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]],device=device)
        actual=sdk.matrix_from_quat(q)
        torch.testing.assert_close(actual,expected,rtol=0,atol=2e-7)
        assert torch.equal(actual,sdk.matrix_from_quat(-q))
        report['sdk_xyzw_known_rotation_check']={'pass':True,'max_absolute_error':float((actual-expected).abs().max()),
            'cases':['identity_xyzw_0001','positive_90_degrees_about_Z','q_and_negative_q_same_rotation'],
            'actual_scripted_function':isinstance(sdk.matrix_from_quat,torch.jit.ScriptFunction)}

        class NativeMathFakeRobot:
            """Two independent CAD-FK poses; no dependency on repository tests or USD."""
            def __init__(self,contract,tilt=False):
                self.contract=contract;self.device='cpu';self.num_envs=2
                self.names=list(reversed(contract['body_paths']))
                origins=np.array([[4.,-3.,1.2],[-2.,8.,-.4]])
                self._terrain=types.SimpleNamespace(env_origins=torch.tensor(origins,dtype=torch.float32))
                relative=kin.forward_kinematics(contract['joint_frames'],contract['default_joint_positions_rad'])
                self.poses=[]
                for index in range(2):
                    world=np.eye(4)
                    world[:3,:3]=_rotation(np.array([0.,0.,1.]),.63 if index else -.27)
                    if tilt:world[:3,:3]=world[:3,:3]@_rotation(np.array([0.,1.,0.]),-.24)@_rotation(np.array([1.,0.,0.]),.13)
                    world[:3,3]=origins[index]+np.array([0.,0.,.5 if tilt else contract['reset_root_height_m']])
                    self.poses.append({name:world@pose for name,pose in relative.items()})
                positions=torch.tensor(np.array([[poses[name][:3,3] for name in self.names] for poses in self.poses]),dtype=torch.float32)
                rotations=torch.tensor(np.array([[poses[name][:3,:3] for name in self.names] for poses in self.poses]),dtype=torch.float64)
                quaternions=sdk.quat_from_matrix(rotations).to(torch.float32)
                torch.testing.assert_close(sdk.matrix_from_quat(quaternions).to(torch.float64),rotations,rtol=0,atol=4e-7)
                data=types.SimpleNamespace(joint_pos=torch.tensor([[contract['default_joint_positions_rad'][name]
                    for name in contract['tree_joint_names']]]*2),joint_vel=torch.zeros(2,30),
                    body_link_pos_w=positions,body_link_quat_w=quaternions,
                    root_pos_w=positions[:,self.names.index('body')],body_com_pos_w=positions+10.,
                    body_com_quat_w=torch.tensor([0.,0.,0.,1.]).expand_as(quaternions),
                    body_link_lin_vel_w=torch.zeros_like(positions),body_link_ang_vel_w=torch.zeros_like(positions),
                    body_com_lin_vel_w=torch.full_like(positions,99.))
                self._robot=types.SimpleNamespace(body_names=self.names,joint_names=list(contract['tree_joint_names']),data=data)

        rows={'row_comparisons':0,'max_absolute_difference_by_field':[0.]*18}
        settings={'source_dir':source,'baseline':args.baseline.resolve(strict=True),
            'candidate':args.candidate.resolve(strict=True),'device':device,'sdk_xyzw':True,
            'fake_robot_class':NativeMathFakeRobot,'matrix_from_quat':sdk.matrix_from_quat,'comparison_summary':rows}
        tests=runpy.run_path(str(HERE/'test_complete_metrics.py'),init_globals={'_COMPARISON_SETTINGS':settings},run_name='complete_metrics_selected_backend')
        baseline=tests['baseline']
        report.update(physics_dt_s=baseline.PHYSICS_DT_S,decimation=baseline.DECIMATION,
            quaternion_order='XYZW',torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
            cuda_version=torch.version.cuda,cuda_device_name=torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            baseline_sha256=digest(args.baseline),candidate_sha256=digest(args.candidate),
            trainer_sha256=digest(source/'isaaclab/train_mkii_fourbar.py'),
            comparator_sha256=digest(Path(__file__)),test_source_sha256=digest(HERE/'test_complete_metrics.py'),
            sdk_provenance=provenance)
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests['CompleteMetricsTests'])
        log=io.StringIO()
        if device.startswith('cuda'):torch.cuda.synchronize(device)
        started=time.perf_counter()
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
        if device.startswith('cuda'):torch.cuda.synchronize(device)
        report.update(elapsed_seconds=time.perf_counter()-started,tests_run=result.testsRun,
            failures=[{'test':str(test),'traceback':error} for test,error in result.failures],
            test_errors=[{'test':str(test),'traceback':error} for test,error in result.errors],
            test_log=log.getvalue(),complete_row_comparison=rows,cuda_initialized=torch.cuda.is_initialized(),
            source_unchanged=identity(source)==contract_before,
            pass_criteria='every complete18-field row byte-equal; strict0 tolerance; equal complete drained windows and guard outcomes')
        report['pass']=result.wasSuccessful() and report['source_unchanged']
        report['limits']=['Synthetic tensors and counted property reads, not native sensor retrieval or PhysX.',
            'Complete capture consumes exact installed scripted XYZW matrix function without the Isaac package initializer.',
            'No performance or physical admission follows from this comparison.']
    except Exception as error:
        import traceback
        report['errors'].append(f'{type(error).__name__}: {error}')
        report['traceback']=traceback.format_exc()
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'pass':report['pass'],'report':str(output),'device':device,
                      'tests_run':report.get('tests_run'),'errors':report['errors']},indent=2))
    return 0 if report['pass'] else 1


if __name__=='__main__':raise SystemExit(main())
