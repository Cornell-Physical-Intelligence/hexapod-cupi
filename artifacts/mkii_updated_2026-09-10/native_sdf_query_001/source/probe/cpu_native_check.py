"""CPU-only integration check: fake query backend, real source USD and actual003 poses."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch
import numpy as np
from pxr import Usd
import native_probe as probe

HERE=Path(__file__).resolve().parent

def check(asset):
    fixture=json.loads((HERE/'fixture/fixture.json').read_text())
    prior=json.loads((HERE/'PARENT_NATIVE_SNAPSHOT.json').read_text())['snapshot']
    points=np.asarray(fixture['points'],dtype=np.float32)
    values=np.c_[fixture['distance_m'],fixture['gradient']].astype(np.float32)
    lookup={tuple(p):value for p,value in zip(points,values)}
    class SdfView:
        count=1;max_num_points=len(points)
        def __init__(self,path):self.object_paths=[path];self.buffer=np.zeros((1,len(points),4),dtype=np.float32)
        def check(self):return True
        def get_sdf_and_gradients(self,data):
            self.buffer[:]=[[lookup[tuple(p)]for p in data[0]]]
            return self.buffer
    class Physics:
        device='cpu'
        def create_sdf_shape_view(self,path,count):
            assert isinstance(path,str)and count==len(points)
            return SdfView(path)
    class Sim:
        stage=Usd.Stage.Open(str(asset/'robot.usda'));physics_sim_view=Physics()
        def get_physics_step_count(self):return prior['explicit_counter']
    class View:
        shared_metatype=types.SimpleNamespace(link_names=prior['link_names'])
        def get_link_transforms(self):return np.asarray(prior['link_pose'])
        def get_link_velocities(self):return np.asarray(prior['link_com_velocity'])
        def get_dof_positions(self):return np.asarray(prior['joint_position'])
        def get_dof_velocities(self):return np.asarray(prior['joint_velocity'])
    identity={'CPU_FAKE_BACKEND':True}
    fake_warp=types.SimpleNamespace(float32=np.float32,array=lambda values,**kwargs:np.asarray(values,dtype=np.float32))
    with tempfile.TemporaryDirectory()as temp:
        with patch.dict(sys.modules,{'warp':fake_warp}),patch.object(probe,'API_SHA',probe.sha(__file__)),patch.object(probe.inspect,'getsourcefile',return_value=__file__),patch.object(probe,'verify_inputs',return_value=(fixture,HERE/'fixture/mesh.npz',identity)):
            state=probe.run_probe(Sim(),View(),HERE/'fixture/fixture.json',asset,Path(temp)/'result')
        if state['status']!='completed':raise AssertionError(state)
        report=json.loads((Path(temp)/'result/report.json').read_text())
        assert report['declared_semantics_supported']and report['geometry_accuracy_within_proposed_bounds']
        assert state['queries_completed']==12 and report['no_observed_state_change']
        return {'scope':'CPU fake backend integration, not actual native SDF evidence','state':state,'report':report,
                'native_backend_executed':False,'warp_or_simulator_imported':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--asset-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier result')
    args.output.write_text(json.dumps(check(args.asset_root.resolve()),indent=2,allow_nan=False)+'\n')
