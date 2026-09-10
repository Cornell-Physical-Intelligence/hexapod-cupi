"""Source-bound device sensor-clock evidence, after ordinary lazy data access.

No calls to sensor.update/reset or simulation stepping occur here. Observed
zero age is read from the installed sensor cache, never fabricated by the caller.
"""
from pathlib import Path
import hashlib
import importlib
import json
import torch

HERE=Path(__file__).parent

def verify_installed_sources():
    contract=json.loads((HERE/'sensor_source_contract.json').read_text())
    result={}
    for module,entry in contract['installed_modules'].items():
        loaded=importlib.import_module(module);path=Path(loaded.__file__).resolve()
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if actual!=entry['sha256']:raise ValueError('Installed sensor source differs: '+module)
        result[module]={'path':str(path),'sha256':actual}
    return result

class ContactFreshness:
    """Read exact14 sensor clocks; require normal8×2.5ms clock advancement.

    `as_tensor` is warp.to_torch in Isaac, injected only for isolated CPU tests.
    Sensor clocks use repeated float32 addition in the installed Warp kernel;
    expected advancement preserves that recurrence, not a rounded dt*8 sum.
    """
    def __init__(self,env,as_tensor):
        self.sensors=(*env._feet_contact_sensors,env._coxa_contact_sensor,*env._femur_contact_sensors,env._base_contact_sensor)
        if len(self.sensors)!=14:raise ValueError('Exact six feet/six femurs/coxa/base sensors required')
        self.n=env.num_envs;self.device=torch.device(env.device);self.as_tensor=as_tensor
        for s in self.sensors:
            if s.cfg.update_period!=.0025 or not s._is_initialized:raise ValueError('Initialized2.5ms sensor update required')
        current,last,outdated=self._read()
        with torch.inference_mode(False):
            self.previous=current.clone();self.ready=torch.ones(self.n,device=self.device,dtype=torch.bool)
        self.initial=current.clone()
        self.ready &= self._finite(current,last,outdated)
    def _tensor(self,value,dtype):
        result=self.as_tensor(value).detach().clone()
        if result.device!=self.device or result.dtype!=dtype or result.shape!=(self.n,):raise ValueError('Wrong sensor clock tensor contract')
        return result
    def _read(self):
        # Ordinary .data access performs the source-verified lazy buffer update.
        for sensor in self.sensors:_=sensor.data
        current=torch.stack([self._tensor(s._timestamp,torch.float32) for s in self.sensors],-1)
        last=torch.stack([self._tensor(s._timestamp_last_update,torch.float32) for s in self.sensors],-1)
        outdated=torch.stack([self._tensor(s._is_outdated,torch.bool) for s in self.sensors],-1)
        return current,last,outdated
    def _finite(self,current,last,outdated):
        return (torch.isfinite(current)&torch.isfinite(last)&(current>=0)&(last>=0)&(current==last)&~outdated).all(-1)
    def read_after_normal_updates(self,physics_steps=8):
        if type(physics_steps) is not int or physics_steps not in (0,8):raise ValueError('Only duplicate read or exact8 physics updates admitted')
        current,last,outdated=self._read();expected=self.previous.clone()
        for _ in range(physics_steps):expected=expected+.0025
        valid=self.ready&self._finite(current,last,outdated)&(current==expected).all(-1)
        age=(current-last).to(torch.float64)
        self.previous.copy_(current);self.ready &= valid
        return dict(contact_valid=valid[:,None].expand(-1,6).clone(),contact_age_s=age[:,:6].clone(),all_sensors_valid=valid,
            sensor_timestamp_s=current,sensor_last_update_s=last,sensor_outdated=outdated,sensor_age_s=age,
            expected_timestamp_s=expected,physics_updates_since_previous_read=physics_steps,
            freshness_semantics='observed_source_bound_lazy_cache_clock_age_not_fabricated_zero')
