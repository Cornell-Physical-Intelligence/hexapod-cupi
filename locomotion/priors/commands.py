"""Bind demonstration coverage to the maintained training command bank."""
import numpy as np


def motion_cases():
    from locomotion.task import TaskConfig, command_bank
    return [{'case_id': f'amp:command_{i:02d}', 'profile': 'omni_static',
             'command': command, 'controls': 1000}
            for i, command in enumerate(command_bank(TaskConfig()))]


def command_index(command):
    value = np.asarray(command)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError('Expected one finite forward/left/yaw command')
    matches = [i for i, case in enumerate(motion_cases())
               if np.allclose(value, case['command'], atol=1e-7, rtol=0)]
    if len(matches) != 1:
        raise ValueError('Command does not match one demonstration coverage case')
    return matches[0]
