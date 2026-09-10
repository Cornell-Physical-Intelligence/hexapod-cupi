"""Offline scalar-oracle fixture construction; excluded from kernel timings.

This intentionally uses CPU dictionaries/NumPy. It is NOT a runtime adapter or
a claim that the scalar state machine is batched. All data are synthetic.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'oracle'))


def oracle(binding):
    key = {'wave002_5mm':'wave002', 'wave003_7mm':'wave003'}[binding]
    module_key = 'tensor_kernel_oracle_'+key
    if module_key in sys.modules:
        return sys.modules[module_key]
    spec = importlib.util.spec_from_file_location(module_key, HERE/'oracle'/f'{key}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_key] = module
    spec.loader.exec_module(module)
    return module


def rows(binding, names=None):
    o = oracle(binding)
    # This fixture imports the exact scalar class only to access its helpers;
    # use the explicitly selected class/config for every controller instance.
    previous = sys.modules.get('wave_reference')
    sys.modules['wave_reference'] = o
    try:
        spec = importlib.util.spec_from_file_location('kernel_fixture', HERE/'oracle/test_wave_reference.py')
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        fixture = m.Fixture(names=names)
    finally:
        if previous is None:
            sys.modules.pop('wave_reference', None)
        else:
            sys.modules['wave_reference'] = previous
    ref = o.WaveContactReference(fixture.names)
    ref.reset(fixture.snapshot())
    result = []
    for i in range(165):
        before = deepcopy(fixture.snapshot())
        q, v = ref.q.copy(), ref.v.copy()
        out = ref.step(before, [0.,0.,0.] if i == 0 else [.005,0.,0.])
        if not out['valid'][0]:
            raise ValueError(out['failure_reason'])
        fixture.advance(out)
        if i in (0,1,20,40,60,80,100,101,105,115,130,155):
            result.append(dict(before=before, after=deepcopy(fixture.snapshot()), output=deepcopy(out),
                               q_previous=q, v_previous=v, mode=out['state']['mode']))
    if not {'reference_quiet_hold', 'swing', 'landing_blend'} <= {r['mode'] for r in result}:
        raise ValueError('Expected mixed phase oracle fixtures')
    return ref, result


def pack_state(rows, g):
    from state_geometry import SHAPES
    result = {key:[] for key in SHAPES}
    result['trajectory_active'] = []
    for row in rows:
        measured, out = row['after'], row['output']
        state = out['state']
        direct = dict(position_world_m=measured['position_world_m'][0], rotation_world_from_body=measured['rotation_world_from_body'][0],
            time_s=out['target_time_s'], target_position_rad=out['q_ref'][0], target_velocity_rad_s=out['v_ref'][0],
            reference_position_rad=out['q_ref'][0], reference_velocity_rad_s=out['v_ref'][0],
            residual_position_rad=np.zeros(18), residual_velocity_rad_s=np.zeros(18))
        for key in SHAPES:
            if key in direct: value = direct[key]
            elif key.startswith('trajectory_'): continue
            else: value = state[key]
            result[key].append(value)
        trajectories = [state['swing'], state['landing']]
        result['trajectory_active'].append([t is not None for t in trajectories])
        for key, shape, source in [('trajectory_coefficients',(6,3),'coefficients'),
                ('trajectory_endpoints_world_m',(3,),'endpoint_world_m'), ('trajectory_start_s',(),'start_s'),
                ('trajectory_duration_s',(),'duration_s'), ('trajectory_lift_m',(),'lift_m')]:
            result[key].append([np.zeros(shape) if t is None else t[source] for t in trajectories])
    return {key:torch.as_tensor(np.asarray(value), dtype=torch.bool if key=='trajectory_active' else g.dtype, device=g.device)
            for key,value in result.items()}
