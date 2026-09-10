from copy import deepcopy
import math,unittest
import torch
from moving_reward import phase,tracking,scoped_terms,RAW_KEYS,Normalization

def fixture(commands,quiet=False):
    c=torch.tensor(commands,dtype=torch.float64);n=len(c);b=lambda v:torch.full((n,),v,dtype=torch.bool);i=lambda v:torch.full((n,),v,dtype=torch.int64)
    s={'ready':b(True),'quiet_valid':b(quiet),'sw_active':b(not quiet),'land_active':b(False),
       'failure':i(0),'mode':i(5 if quiet else 1),'current_leg':i(-1 if quiet else 0),
       'requested':c.clone(),'command':c.clone(),'command_target':c.clone(),'rate':torch.zeros_like(c),'time':torch.ones(n,dtype=torch.float64),
       'quiet_time':torch.ones(n,dtype=torch.float64),'factor':torch.ones(n,dtype=torch.float64)}
    return c,s,torch.ones((n,6),dtype=torch.bool),b(True)

def classify(args):return phase(*args)
class ScopedRewardTests(unittest.TestCase):
    def test_all_bearings_yaw_signs_arcs_and_tiny_nonzero_are_motion(self):
        c=[[0,0,0],[.005,0,0],[-.005,0,0],[0,.005,0],[0,-.005,0],[0,0,.015],[0,0,-.015],
           [.003,-.004,-.01],[1e-12,0,0],[0,0,-1e-12]]
        args=fixture(c);scope=classify(args)
        self.assertEqual(scope['commanded_motion'].tolist(),[False]+[True]*9)
        self.assertEqual(scope['finite_supported_stop'].tolist(),[True]+[False]*9)
        self.assertFalse(scope['settled_quiet'].any())
        q=classify(fixture([[0,0,0]],quiet=True));self.assertTrue(q['settled_quiet'].all())
        self.assertFalse(q['policy_training_allowed'])
    def test_move_supported_landing_stop_then_quiet_mask_truth_table(self):
        for label,args in [('moving',fixture([[.005,0,0]])),('stopping',fixture([[0,0,0]])),('quiet',fixture([[0,0,0]],True))]:
            c,s,contacts,valid=args
            if label=='stopping':s.update(command=torch.tensor([[.002,0,0]],dtype=torch.float64),mode=torch.tensor([6]),land_active=torch.tensor([True]));contacts[0,0]=False
            scope=classify(args);raw={k:torch.tensor([float(j+1)],dtype=torch.float64) for j,k in enumerate(RAW_KEYS)}
            terms=scoped_terms(torch.zeros(1,2,dtype=torch.float64),torch.zeros(1,dtype=torch.float64),scope,raw)
            for j,k in enumerate(RAW_KEYS[:-1]):self.assertEqual(float(terms[k]),float(j+1) if label=='quiet' else 0.)
            self.assertEqual(float(terms['airtime']),5. if label=='moving' else 0.)
            if label=='stopping':
                self.assertEqual(float(scope['tracking_target'][0,0]),.002)
                self.assertEqual(float(terms['linear_progress']),0.)
                self.assertTrue(scope['finite_supported_stop'].all())
    def test_exact_zero_preserves_old_tracking_and_normalized_extremes(self):
        zero=classify(fixture([[0,0,0]],True));v=torch.tensor([[.01,-.02]],dtype=torch.float64);w=torch.tensor([.03],dtype=torch.float64)
        t=tracking(v,w,zero)
        self.assertAlmostEqual(float(t['linear_tracking']),math.exp(-(.01**2+.02**2)/.04**2),14)
        self.assertAlmostEqual(float(t['yaw_tracking']),math.exp(-.03**2/.1**2),14)
        for multiplier,lin,prog in [(0,math.exp(-1),0),(1,1,1),(-1,math.exp(-4),-1),(2,math.exp(-1),1)]:
            c,s,co,valid=fixture([[.005,0,.015]]);scope=phase(c,s,co,valid);t=tracking(c[:,:2]*multiplier,c[:,2]*multiplier,scope)
            for k in ('linear_tracking','yaw_tracking'):self.assertAlmostEqual(float(t[k]),lin,13)
            for k in ('linear_progress','yaw_progress'):self.assertEqual(float(t[k]),prog)
    def test_rotation_symmetry_and_continuous_tracking_progress_zero_limit(self):
        values=[]
        for angle in (0.,.7,math.pi,4.1):
            rot=torch.tensor([[math.cos(angle),-math.sin(angle)],[math.sin(angle),math.cos(angle)]],dtype=torch.float64)
            c,s,co,valid=fixture([[.003,.004,-.01]]);c[:,:2]=c[:,:2]@rot.T;s['requested']=c.clone();s['command']=c.clone();s['command_target']=c.clone()
            values.append(tracking(c[:,:2]*.4,c[:,2]*.4,phase(c,s,co,valid)))
        for key in values[0]:
            for v in values[1:]:torch.testing.assert_close(v[key],values[0][key],atol=1e-14,rtol=0)
        base=tracking(torch.tensor([[.001,-.002]],dtype=torch.float64),torch.tensor([.003],dtype=torch.float64),classify(fixture([[0,0,0]],True)))
        for sign in (-1,1):
            errors=[]
            for eps in (1e-5,1e-7,1e-9,1e-11):
                scope=classify(fixture([[sign*eps,0,sign*eps]]));v=torch.tensor([[.001,-.002]],dtype=torch.float64);w=torch.tensor([.003],dtype=torch.float64);t=tracking(v,w,scope)
                errors.append(max(abs(float(t[k]-base[k])) for k in ('linear_tracking','yaw_tracking')))
            self.assertTrue(all(a>b for a,b in zip(errors,errors[1:])))
            self.assertLess(errors[-1],1e-9)
        # Mode-dependent mask changes are deliberately discrete, not falsely
        # presented as globally continuous full rewards for fixed noisy motion.
        self.assertTrue(classify(fixture([[1e-12,0,0]]))['commanded_motion'][0])
        self.assertTrue(classify(fixture([[0,0,0]],True))['settled_quiet'][0])
    def test_stale_queued_invalid_support_or_inconsistent_quiet_rejected(self):
        changes=[lambda a:a[1]['requested'].fill_(.001),lambda a:a[3].fill_(False),lambda a:a[1]['ready'].fill_(False),
                 lambda a:a[1]['failure'].fill_(15),lambda a:a[2].fill_(False),lambda a:a[1]['command'].fill_(float('nan')),
                 lambda a:a[1]['factor'].fill_(.5),lambda a:a[1]['mode'].fill_(9),lambda a:a[1].pop('rate')]
        for mutate in changes:
            a=fixture([[.005,0,0]]);mutate(a)
            with self.assertRaises((ValueError,KeyError)):classify(a)
        for key in ('sw_active','land_active'):
            a=fixture([[0,0,0]],True);a[1][key].fill_(True)
            with self.assertRaises(ValueError):classify(a)
        a=fixture([[0,0,0]],True);a[2][0,0]=False
        with self.assertRaises(ValueError):classify(a)
        for value in (0.,-1.,float('inf'),float('nan'),True):
            with self.assertRaises(ValueError):Normalization(translation_mps=value)
    def test_tiny_resume_stale_mode_and_contact_pause_are_not_false_quiet(self):
        # Real wave005 leaves mode=5 until a new liftoff but clears quiet_valid
        # for every nonzero request, including requests below its liftoff trigger.
        c,s,co,valid=fixture([[1e-12,0,0]],quiet=True)
        s['quiet_valid'].fill_(False);s['command']*=1e-3
        scope=phase(c,s,co,valid)
        self.assertTrue(scope['commanded_motion'][0]);self.assertFalse(scope['settled_quiet'][0])
        c,s,co,valid=fixture([[.005,0,0]])
        s['mode'].fill_(2);s['factor'].zero_();s['command_target'].zero_();s['command']*=.4;co[0,0]=False
        scope=phase(c,s,co,valid)
        self.assertTrue(scope['commanded_motion'][0]);self.assertEqual(float(scope['tracking_target'][0,0]),.005)
        self.assertEqual(float(scope['reference_governor_factor'][0]),0.)
    def test_missing_or_nonfinite_unmasked_terms_rejected(self):
        scope=classify(fixture([[0,0,0]],True));raw={k:torch.zeros(1,dtype=torch.float64) for k in RAW_KEYS}
        for bad in ({},dict(raw,stand_posture=torch.tensor([float('nan')],dtype=torch.float64)),dict(raw,stand_joint_velocity=torch.tensor([-1.],dtype=torch.float64))):
            with self.assertRaises(ValueError):scoped_terms(torch.zeros(1,2,dtype=torch.float64),torch.zeros(1,dtype=torch.float64),scope,bad)
if __name__=='__main__':unittest.main()
