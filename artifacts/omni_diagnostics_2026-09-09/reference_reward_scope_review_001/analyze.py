"""Read-only reward-scope audit; does not alter or evaluate a learned policy."""
from pathlib import Path
import argparse, ast, hashlib, json
import numpy as np
import torch
import yaml

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def function(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sm = a.source / 'campaign_source_hashes.json'
    assert sha(sm) == '04942c62de1f557d8f117f648f990b6dac69443d61c546529d05a4e2a7f0e01e'
    source_map = json.loads(sm.read_text())
    files = [a.source / 'tools' / x for x in ('omni_flat_math.py', 'omni_flat_env.py')]
    for f in files: assert sha(f) == source_map[str(f.relative_to(a.source))]
    track = function(files[0], 'tracking_terms')
    quiet = function(files[1], 'quiet_stand_terms')
    cases = [('stand', [0., 0., 0.]), ('forward', [.005, 0., 0.]),
             ('reverse', [-.005, 0., 0.]), ('left_strafe', [0., .005, 0.]),
             ('left_turn', [0., 0., .015]),
             ('forward_right_arc', [.005 / 2**.5, -.005 / 2**.5, -.01])]
    rows = []
    for name, command in cases:
        c = torch.tensor([command], dtype=torch.float64)
        stopped = track(torch.zeros((1, 2)), torch.zeros(1), c)
        perfect = track(c[:, :2], c[:, 2], c)
        # Unit velocities make the exact frozen quiet mask observable.
        q = quiet(c, torch.ones((1, 18)), torch.full((1, 18), .02), torch.zeros((1, 18)), .02)
        rows.append(dict(case=name, requested_twist=command,
            frozen_quiet_penalties_enabled=bool(q['stand_joint_velocity'][0]),
            frozen_linear_progress_at_perfect_tracking=float(perfect['linear_progress'][0]),
            frozen_yaw_progress_at_perfect_tracking=float(perfect['yaw_progress'][0]),
            stopped_linear_tracking=float(stopped['linear_tracking'][0]),
            stopped_yaw_tracking=float(stopped['yaw_tracking'][0]),
            perfect_tracking_gain_weighted_per_second=float(4 * (perfect['linear_tracking']-stopped['linear_tracking'])[0] + 2 * (perfect['yaw_tracking']-stopped['yaw_tracking'])[0])))
    base = a.repository / 'artifacts/omni_diagnostics_2026-09-09/reference_physics_009'
    traces = list(base.rglob('wave/trace.npz')); assert len(traces) == 1
    environments = list(base.rglob('wave/environment.yaml')); assert len(environments) == 1
    trace, environment = traces[0], environments[0]
    lines = environment.read_text().splitlines(); start = lines.index('omni_reward_weights:')
    end = next(i for i in range(start+1, len(lines)) if lines[i] and not lines[i].startswith(' '))
    weights = yaml.safe_load('\n'.join(lines[start:end]))['omni_reward_weights']
    with np.load(trace) as f:
        c = torch.as_tensor(f['requested_command'][:, 0], dtype=torch.float64)
        active = c.norm(dim=-1) > 0
        position = torch.as_tensor(f['applied_position_target_rad'][:, 0])
        previous = torch.cat((position[:1], position[:-1]))
        joint_velocity = torch.as_tensor(f['joint_velocity_rad_s'][:, 0])
        terms = quiet(c, joint_velocity, position, previous, .02)
        component_replay = {key: {'weight': weights[key],
            'mean_weighted_per_second_on_requested_motion': float((value[active] * weights[key]).mean())}
            for key, value in terms.items()}
    report = dict(scope='frozen reward mask and isolated components; not exact full reward replay',
        source_manifest_sha256=sha(sm), source_files={str(f.relative_to(a.source)):sha(f) for f in files},
        actual_trace={'path':str(trace.relative_to(a.repository)), 'sha256':sha(trace)},
        environment={'path':str(environment.relative_to(a.repository)), 'sha256':sha(environment)},
        cases=rows, actual_requested_motion_rows=int(active.sum()), component_replay=component_replay,
        limitations=['Requested command is replayed; internal reward command slew timing is not reconstructed.',
            'Reported SDK joint-rate bias remains; no finite-difference substitution.',
            'This audit does not show a trained reward exploit or quantify all reward terms.',
            'Current two-update zero-command integration is unchanged.'],
        requires_before_moving_PPO=['Explicit moving/finite-stop/settled-stand reward states.',
            'Direction-invariant tracking/progress at the actually admitted speed envelope.',
            'Raw component logging and matched quiet/forward retention with unchanged acceptance gates.'],
        gpu_launches=0, Stage2_complete=False)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'cases':len(rows), 'all_low_speed_cases_receive_stand_penalties':all(r['frozen_quiet_penalties_enabled'] for r in rows), 'motion_rows':int(active.sum()), 'components':component_replay}))

if __name__ == '__main__': main()
