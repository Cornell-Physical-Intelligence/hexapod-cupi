"""Position/velocity stepping prior plus learned bounded joint corrections.

No base pose, base velocity, contact forces or foot positions are imposed on
physics. All reference targets go through the existing capped RS05 actuator.
"""

# Allow direct invocation from any working directory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from tools.c_study_runtime import bootstrap_c_study_runtime
bootstrap_c_study_runtime()

import json
from pathlib import Path
import torch
from hexapod_rl.env import HexapodEnv


def interpolate_periodic(table,phase):
    position=torch.remainder(phase*table.shape[0]-.5,table.shape[0])
    lower=position.floor().long();upper=(lower+1)%table.shape[0]
    fraction=(position-lower).unsqueeze(-1)
    return table[lower]*(1-fraction)+table[upper]*fraction


class ReferenceGaitEnv(HexapodEnv):
    def __init__(self,*args,reference_path:Path,reference_enabled=True,warmup_s=2.,**kwargs):
        super().__init__(*args,**kwargs)
        data=json.loads(reference_path.read_text())
        lookup={name:i for i,name in enumerate(data["joint_names"])}
        if set(lookup)!=set(self._robot.joint_names):raise RuntimeError("Reference joint names mismatch")
        order=[lookup[name] for name in self._robot.joint_names]
        self._reference_positions=torch.tensor(data["positions_rad"],device=self.device)[:,order]
        self._reference_derivatives=torch.tensor(data["derivative_per_cycle"],device=self.device)[:,order]
        self._reference_enabled=reference_enabled
        self._reference_age=torch.zeros(self.num_envs,device=self.device)
        self._reference_velocity=torch.zeros_like(self._actions)
        self._reference_warmup=warmup_s

    def _reset_idx(self,env_ids):
        super()._reset_idx(env_ids)
        if hasattr(self,"_reference_age"):
            self._reference_age[env_ids]=0
            self._reference_velocity[env_ids]=0

    def _pre_physics_step(self,actions):
        super()._pre_physics_step(actions)
        if not self._reference_enabled:return
        moving=self._commands[:,0]>.05
        self._reference_age=torch.where(moving,self._reference_age+self.step_dt,torch.zeros_like(self._reference_age))
        u=(self._reference_age/self._reference_warmup).clamp(0,1)
        blend=(u*u*(3-2*u)).unsqueeze(-1)
        blend_rate=(6*u*(1-u)/self._reference_warmup).unsqueeze(-1)
        q=interpolate_periodic(self._reference_positions,self._gait_phase)
        derivative=interpolate_periodic(self._reference_derivatives,self._gait_phase)
        frequency=(self._commands[:,0]*self.cfg.gait_cycles_per_meter).clamp(self.cfg.gait_min_frequency_hz,self.cfg.gait_max_frequency_hz)
        frequency=frequency*moving
        default=self._robot.data.default_joint_pos.torch
        target=default+blend*(q-default)+self.cfg.action_scale*self._actions
        limits=self._robot.data.soft_joint_pos_limits.torch
        self._processed_actions=target.clamp(limits[:,:,0],limits[:,:,1])
        self._reference_velocity=blend*derivative*frequency.unsqueeze(-1)+blend_rate*(q-default)
        self._reference_velocity=torch.where((target>=limits[:,:,0])&(target<=limits[:,:,1]),self._reference_velocity,torch.zeros_like(target))

    def _apply_action(self):
        super()._apply_action()
        if self._reference_enabled:
            self._robot.set_joint_velocity_target_index(target=self._reference_velocity)
