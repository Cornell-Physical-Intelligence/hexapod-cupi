from types import SimpleNamespace as NS
import unittest
import torch
from sensor_freshness import ContactFreshness
from pre_reset_device import capture_before_reset,require_one

class Sensor:
    def __init__(self,n):
        self.cfg=NS(update_period=.0025);self._is_initialized=True
        self._timestamp=torch.zeros(n,dtype=torch.float32);self._timestamp_last_update=torch.zeros_like(self._timestamp)
        self._is_outdated=torch.ones(n,dtype=torch.bool);self.skip_lazy_update=False
    @property
    def data(self):
        if not self.skip_lazy_update:
            self._timestamp_last_update[self._is_outdated]=self._timestamp[self._is_outdated];self._is_outdated.fill_(False)
        return NS()
    def tick(self):self._timestamp+=.0025;self._is_outdated.fill_(True)

def fixture():
    sensors=[Sensor(3) for _ in range(14)]
    return NS(_feet_contact_sensors=sensors[:6],_coxa_contact_sensor=sensors[6],_femur_contact_sensors=sensors[7:13],_base_contact_sensor=sensors[13],num_envs=3,device='cpu'),sensors

class BridgeTests(unittest.TestCase):
    def test_exact_clock_recurrence_lazy_update_and_duplicate_read(self):
        env,sensors=fixture();r=ContactFreshness(env,lambda x:x)
        for _ in range(400):
            for __ in range(8):
                for s in sensors:s.tick()
            result=r.read_after_normal_updates();self.assertTrue(result['all_sensors_valid'].all())
            self.assertTrue((result['contact_age_s']==0).all())
            self.assertTrue(r.read_after_normal_updates(0)['all_sensors_valid'].all())
        self.assertNotEqual(float(sensors[0]._timestamp[0]),8.) # Preserve float32 recurrence, not invented scalar time.
    def test_missing_update_stale_cache_and_clock_reset_fail_affected_rows(self):
        for kind in ('missing','stale','reset'):
            env,sensors=fixture();r=ContactFreshness(env,lambda x:x)
            for _ in range(8):
                for s in sensors:s.tick()
            if kind=='missing':sensors[1]._timestamp[1]-=.0025
            elif kind=='stale':sensors[1].skip_lazy_update=True
            else:sensors[1]._timestamp[1]=0
            result=r.read_after_normal_updates();self.assertFalse(result['all_sensors_valid'][1])
            if kind!='stale':self.assertEqual(result['all_sensors_valid'].tolist(),[True,False,True])
            self.assertFalse(r.read_after_normal_updates(0)['all_sensors_valid'][1])
    def test_wrong_dtype_period_uninitialized_or_nonfinite_rejected(self):
        env,sensors=fixture();sensors[0].cfg.update_period=.02
        with self.assertRaises(ValueError):ContactFreshness(env,lambda x:x)
        env,sensors=fixture();sensors[0]._timestamp=sensors[0]._timestamp.double();sensors[0]._timestamp_last_update=sensors[0]._timestamp_last_update.double()
        with self.assertRaises(ValueError):ContactFreshness(env,lambda x:x)
        env,sensors=fixture();r=ContactFreshness(env,lambda x:x);sensors[0]._timestamp[2]=torch.nan
        self.assertFalse(r.read_after_normal_updates()['all_sensors_valid'][2])
    def test_hook_observes_original_done_once_before_reset_and_restores(self):
        class Env:
            def __init__(self):self.calls=0;self.position=torch.tensor([1.,2.])
            def _get_dones(self):self.calls+=1;return torch.tensor([False,True]),torch.tensor([False,False])
            def step(self):
                t,u=self._get_dones();self.position[t]=99.;return t,u
        e=Env();original=e._get_dones
        def capture(t,u):return {'measurement':{'terminated':t,'truncated':u,'q':e.position.clone()}}
        with capture_before_reset(e,capture) as samples:
            t,u=e.step();p=require_one(samples,0,t,u);self.assertEqual(p['measurement']['q'].tolist(),[1.,2.])
        self.assertEqual(e._get_dones,original);self.assertEqual(e.calls,1)
        with self.assertRaises(RuntimeError):require_one(samples,1,t,u)
        with self.assertRaises(ValueError):
            with capture_before_reset(e,lambda *_:(_ for _ in ()).throw(ValueError('capturefailed'))):e.step()
        self.assertEqual(e._get_dones,original)

if __name__=='__main__':unittest.main()
