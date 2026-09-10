"""Explicit synthetic CPU fixtures; these do not qualify physics."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).parent/'reference'))
from batch_wave import BatchWave005
from test_batch_wave import pack,Fixture,WaveContactReference
from reference_residual_oracle import ReferenceResidualTarget,ResidualConfig
from observation import ObservationBuilder

class Rig:
    def __init__(self,n=1,names=None):
        self.fs=[Fixture(names) for _ in range(n)];self.rs=[WaveContactReference(f.names) for f in self.fs]
        self.names=self.fs[0].names;self.n=n;self.steps=torch.zeros(n,dtype=torch.int64);self.episodes=torch.zeros_like(self.steps)
        self.wave=BatchWave005(self.names,n)
        for r,f in zip(self.rs,self.fs):r.reset(f.snapshot())
        out=self.wave.reset(pack([f.snapshot() for f in self.fs]),torch.ones(n,dtype=torch.bool),self.episodes)
        limits=self.fs[0].snapshot()['soft_joint_pos_limits_rad'][0]
        self.controller=ReferenceResidualTarget(self.names,dict(zip(self.names,limits[:,0])),dict(zip(self.names,limits[:,1])),n,ResidualConfig('formal_004',.02,.25,2.,8.))
        self.controller.reset(out['q_ref'],out['q_ref'])
        self.c=self.controller.step(out['q_ref'],torch.zeros(n,18,dtype=torch.float64),reference_valid=out['valid'])
        self.builder=ObservationBuilder(self.names,n);self.builder.reset(torch.ones(n,dtype=torch.bool),self.episodes,torch.zeros(n,dtype=torch.float64))
    def packet(self,requested=None):
        m=pack([f.snapshot() for f in self.fs]);m['contact_point_world_m']=torch.nan_to_num(m['contact_point_world_m'])
        m.update(joint_velocity_rad_s=self.c['target_velocity_rad_s'].clone(),projected_gravity_body=-(torch.tensor([0.,0.,1.],dtype=torch.float64)[None,None,:]@m['rotation_world_from_body']).squeeze(1),
            root_link_velocity_body_mps=m['velocity_body_mps'].clone(),contact_valid=torch.ones(self.n,6,dtype=torch.bool),contact_age_s=torch.zeros(self.n,6,dtype=torch.float64),measurement_valid=torch.ones(self.n,dtype=torch.bool))
        return dict(measurement_source='synthetic_fixture',joint_names_runtime=self.names,measurement=m,reference=self.wave.output(),controller=deepcopy(self.c),
            requested_twist=torch.zeros(self.n,3,dtype=torch.float64) if requested is None else torch.tensor(requested,dtype=torch.float64),world_up=torch.tensor([[0.,0.,1.]]*self.n,dtype=torch.float64),
            episode_ids=self.episodes.clone(),step_indices=self.steps.clone(),critic_simulator_reported_twist=torch.zeros(self.n,3,dtype=torch.float64))
    def advance(self,commands):
        snaps=[f.snapshot() for f in self.fs]
        olds=[r.step(s,c) for r,s,c in zip(self.rs,snaps,commands)]
        out=self.wave.step(pack(snaps),torch.tensor(commands,dtype=torch.float64))
        if not out['valid'].all():raise AssertionError(out['failure_code'])
        self.c=self.controller.step(out['q_ref'],torch.zeros(self.n,18,dtype=torch.float64),reference_valid=out['valid'])
        for f,old in zip(self.fs,olds):f.advance(old)
        self.steps=self.steps+1
        return self.packet(commands)

def fresh_encode(packet):
    b=ObservationBuilder(packet['joint_names_runtime'],len(packet['episode_ids']))
    packet=deepcopy(packet);packet['step_indices'].zero_()
    b.reset(torch.ones(b.n,dtype=torch.bool),packet['episode_ids'],packet['measurement']['time_s'])
    return b.build(packet)
