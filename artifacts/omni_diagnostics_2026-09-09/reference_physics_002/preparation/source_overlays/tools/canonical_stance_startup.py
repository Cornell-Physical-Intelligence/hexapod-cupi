"""Bounded target-only C2 return from the actual randomized reset to nominal.

The physical reset, joint noise, body pose and motor dynamics are untouched.
No target clipping or retiming is allowed. The residual controller independently
checks the emitted discrete knots; continuous derivatives are separate telemetry.
"""
import math
import numpy as np


class CanonicalStanceStartup:
    def __init__(self, start, nominal, lower, upper, *, duration_s=2., dt=.02,
                 reference_velocity_rad_s=1.75, reference_acceleration_rad_s2=6.,
                 residual_margin_rad=.02):
        self.start = np.array(start, dtype=np.float64, copy=True)
        self.nominal = np.array(nominal, dtype=np.float64, copy=True)
        self.lower = np.array(lower, dtype=np.float64, copy=True)
        self.upper = np.array(upper, dtype=np.float64, copy=True)
        if (self.start.ndim != 2 or self.start.shape[1] != 18
                or any(x.shape != self.start.shape for x in (self.nominal,self.lower,self.upper))
                or any(not np.isfinite(x).all() for x in (self.start,self.nominal,self.lower,self.upper))):
            raise ValueError('Finite matching [environment,18] named arrays required')
        if (not all(math.isfinite(x) and x > 0 for x in
                    (duration_s,dt,reference_velocity_rad_s,reference_acceleration_rad_s2,residual_margin_rad))
                or not math.isclose(duration_s/dt,round(duration_s/dt),abs_tol=1e-10)):
            raise ValueError('Positive finite duration/rates and whole control intervals required')
        self.duration_s,self.dt=duration_s,dt
        self.velocity_limit,self.acceleration_limit=reference_velocity_rad_s,reference_acceleration_rad_s2
        self.margin=residual_margin_rad
        self.steps=int(round(duration_s/dt))
        if any(((q < self.lower+self.margin) | (q > self.upper-self.margin)).any()
               for q in (self.start,self.nominal)):
            raise ValueError('Both endpoints require actual soft-limit residual margin')
        # Check every startup knot plus both first hold derivatives before any
        # simulator target is emitted, including v(0)=0 and held endpoint knots.
        for step in range(1,self.steps+3):
            self.sample(step)

    def position(self, step):
        if type(step) is not int or step < 0:
            raise ValueError('Nonnegative integer control-knot index required')
        if step == 0:
            return self.start.copy()
        if step >= self.steps:
            return self.nominal.copy()
        u=step/self.steps
        s=u*u*u*(10.+u*(-15.+6.*u))
        return self.start+(self.nominal-self.start)*s

    def sample(self, step):
        if type(step) is not int or step < 0:
            raise ValueError('Nonnegative integer control-knot index required')
        q=self.position(step)
        previous=self.position(max(0,step-1))
        previous_previous=self.position(max(0,step-2))
        v=(q-previous)/self.dt
        previous_v=(previous-previous_previous)/self.dt
        a=(v-previous_v)/self.dt
        u=min(1.,step/self.steps)
        delta=self.nominal-self.start
        analytic_v=delta*(30*u*u*(1-u)*(1-u))/self.duration_s
        analytic_a=delta*(60*u-180*u*u+120*u*u*u)/(self.duration_s**2)
        if (np.abs(v).max() > self.velocity_limit+1e-10
                or np.abs(a).max() > self.acceleration_limit+1e-10
                or ((q < self.lower+self.margin) | (q > self.upper-self.margin)).any()):
            raise ValueError('Startup reference exceeds named position/rate/acceleration budget')
        return {'q_ref':q,'v_ref':v,'a_ref':a,
                'analytic_velocity_rad_s':analytic_v,'analytic_acceleration_rad_s2':analytic_a,
                'valid':np.ones(len(q),dtype=bool),'time_s':step*self.dt,
                'phase':'transition' if step < self.steps else 'canonical_hold'}

    def contract(self):
        samples=[self.sample(step) for step in range(self.steps+3)]
        return {'kind':'target_only_quintic_C2_random_reset_to_named_canonical_stance',
                'duration_s':self.duration_s,'control_dt_s':self.dt,
                'physical_reset_preserved':True,'randomized_start_target_rad':self.start.tolist(),
                'nominal_target_rad':self.nominal.tolist(),'reference_residual_margin_rad':self.margin,
                'max_discrete_velocity_rad_s':max(float(np.abs(x['v_ref']).max()) for x in samples),
                'max_discrete_acceleration_rad_s2':max(float(np.abs(x['a_ref']).max()) for x in samples),
                'analytic_endpoint_velocity_and_acceleration_zero':True,
                'discrete_hold_derivatives_checked_through_step':self.steps+2}
