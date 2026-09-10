"""Read-only actual-smoke replay; prints JSON, never edits input or starts Isaac.

Usage: python3 -B review.py --raw <published-smoke002/raw>
Dependencies: NumPy, SciPy. Only snapshot source copies are imported.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
SOURCE_SHA = '64144b2c2466f17c809554ef53f82b736b4199136cc821451d7ac2788752bd2e'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def scalar_yaml(path, key):
    matches = re.findall(r'^\s*' + re.escape(key) + r': ([0-9.e+-]+)$', path.read_text(), re.M)
    assert len(matches) == 1, (path, key, matches)
    return float(matches[0])

def main(raw):
    raw_map = json.loads((HERE / 'RAW_SHA256.json').read_text())
    for name, row in raw_map.items():
        path = raw / name
        assert digest(path) == row['sha256'], name
        assert path.stat().st_size == row['bytes'], name
    source_map = json.loads((HERE / 'source/campaign_source_hashes.json').read_text())
    assert digest(HERE / 'source/campaign_source_hashes.json') == SOURCE_SHA
    bindings = json.loads((HERE / 'SOURCE_BINDINGS.json').read_text())
    for name, row in bindings.items():
        assert digest(HERE / 'source' / name) == row['sha256'] == source_map[row['source_path']]
    assert digest(HERE / 'NATIVE_FREEZE_SHA256.json') == '20fbcd40a869b39d8e3143a6c9715ec62ada5cc6211eaf473f60f8b7167678cb'
    native_map = json.loads((HERE / 'NATIVE_FREEZE_SHA256.json').read_text())
    assert digest(HERE / 'source/build_source.py') == native_map['build_source.py']
    sdk_map = json.loads((HERE / 'INSTALLED_SNAPSHOT_FREEZE.json').read_text())['files']
    assert digest(HERE / 'source/installed_base_articulation_data.py') == sdk_map['installed_primary/base_articulation_data.py']
    entry = (HERE / 'source/train_length_study.py').read_text()
    assert 'cfg.episode_length_s = 20.0 if args.mode=="train" else (90.0 if omni else 25.0)' in entry
    base = (HERE / 'source/hexapod_env.py').read_text()
    assert 'time_out = self.episode_length_buf >= self.max_episode_length - 1' in base
    omni_ast = ast.parse((HERE / 'source/omni_flat_env.py').read_text())
    omni_cls = next(n for n in omni_ast.body if isinstance(n, ast.ClassDef) and n.name == 'OmniFlatEnv')
    assert not any(isinstance(n, ast.FunctionDef) and n.name == '_get_dones' for n in omni_cls.body)
    stop_src = (HERE / 'source/direct_stop_evaluation.py').read_text()
    assert 'env.episode_length_buf.zero_()' in stop_src
    installed = (HERE / 'source/installed_base_articulation_data.py').read_text()
    assert 'Root link orientation (x, y, z, w)' in installed and 'return self.root_link_quat_w' in installed

    run = raw / 'run'
    with np.load(run / 'final_stop/stop_trace.npz', allow_pickle=False) as f:
        stop = {k: f[k] for k in f.files}
    with np.load(run / 'train/training_trace.npz', allow_pickle=False) as f:
        train = {k: f[k] for k in f.files}
    with np.load(run / 'train/training_joint_trace.npz', allow_pickle=False) as f:
        joints = {k: f[k] for k in f.files}
    assert stop['command'].shape == (1600, 48, 3)
    assert train['target_delta_rms_rad'].shape == (48, 32)
    assert np.array_equal(stop['joint_names'], train['joint_names'])
    assert np.array_equal(joints['joint_names'], train['joint_names'])

    qxyzw, qwxyz = stop['quaternion_world_xyzw'], stop['quaternion_world_wxyz']
    assert np.array_equal(qwxyz, qxyzw[..., [3, 0, 1, 2]])
    rot = Rotation.from_quat(qxyzw.reshape(-1, 4))
    gravity = rot.inv().apply([0., 0., -1.]).reshape(1600, 48, 3)
    forward = rot.apply([0., -1., 0.]).reshape(1600, 48, 3)
    independent_heading = np.unwrap(np.arctan2(forward[..., 1], forward[..., 0]), axis=0)
    w, x, y, z = qwxyz.astype(np.float64).transpose(2, 0, 1)
    scorer_heading = np.unwrap(np.arctan2(-1 + 2*(x*x + z*z), 2*(w*z - x*y)), axis=0)
    gravity_error = float(np.abs(gravity - stop['projected_gravity']).max())
    heading_error = float(np.abs(independent_heading - scorer_heading).max())
    assert gravity_error < 1e-6 and heading_error < 1e-6
    # Independent nontrivial yaw/roll examples, using installed XYZW convention.
    synthetic = []
    for axis in ('z', 'x'):
        raw_q = Rotation.from_euler(axis, np.array([[0.], [10.]]), degrees=True).as_quat()
        w, x, y, z = raw_q[:, [3, 0, 1, 2]].T
        h = np.unwrap(np.arctan2(-1 + 2*(x*x + z*z), 2*(w*z - x*y)))
        change = float(np.degrees(h[1] - h[0]))
        assert abs(change - (10. if axis == 'z' else 0.)) < 1e-10
        synthetic.append({'axis': axis, 'rotation_deg': 10., 'heading_change_deg': change})

    command = stop['command']
    assert not np.any(command[1100:])
    assert not np.any(stop['target_command'][800:])
    last_nonzero = [800 + int(np.flatnonzero(np.any(command[800:, i] != 0, axis=1))[-1])
                    if np.any(command[800:, i]) else 799 for i in range(48)]
    last = max(last_nonzero)
    assert not np.any(command[last + 1:])
    receipt = json.loads((run / 'train/training_receipt.json').read_text())
    updates = receipt['optimizer_updates']
    losses = [{'update': u['completed_update'], 'learning_rate_at_update_end': u['learning_rate'],
               **{k: u['losses'][k] for k in ('caps_temporal', 'caps_spatial', 'caps_weighted', 'caps_valid_pair_fraction')},
               'temporal_to_spatial_loss_ratio': u['losses']['caps_temporal'] / u['losses']['caps_spatial']}
              for u in updates]
    assert all(r['caps_temporal'] > 0 and r['caps_spatial'] > 0 for r in losses)
    delta = train['target_delta_rms_rad']
    observed_delta = np.sqrt(np.mean(np.diff(joints['joint_target_rad'], axis=0)**2, axis=-1))
    # These exact eight rows had no reset. Do not compare across a reset discontinuity.
    ids = joints['env_ids']
    assert not np.any(train['terminated'][:, ids]) and not np.any(train['truncated'][:, ids])
    delta_error = float(np.abs(observed_delta - delta[1:, ids]).max())
    assert delta_error < 1e-8 and np.all(delta > 0)

    spec = importlib.util.spec_from_file_location('frozen_quiet', HERE / 'source/direct_quiet_metrics.py')
    quiet_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quiet_module)
    quiet = [quiet_module.quiet_metrics(stop, i, 1100, stop['joint_names'].tolist(), .02) for i in range(48)]
    saved = json.loads((run / 'final_stop/stop_diagnostics.json').read_text())
    saved_rows = [r for s in saved['scenarios'] for r in s['replicas']]
    assert [r['pass'] for r in quiet] == [r['quiet']['pass'] for r in saved_rows]
    return {
        'scope': 'CPU review of actual native003 direct smoke002 only; no new physical admission, source edits or GPU work',
        'verified_raw_files': len(raw_map), 'verified_source_copies': len(bindings), 'source_manifest_sha256': SOURCE_SHA,
        'episode': {'training_config_s': scalar_yaml(run/'train/environment.yaml', 'episode_length_s'),
                    'stop_config_s': scalar_yaml(run/'final_stop/environment.yaml', 'episode_length_s'),
                    'stop_controls': 1600, 'stop_duration_s': 32.,
                    'stop_terminations': int(stop['terminated'].sum()), 'stop_truncations': int(stop['truncated'].sum()),
                    'quiet_terminations': int(stop['terminated'][1100:].sum()), 'quiet_truncations': int(stop['truncated'][1100:].sum()),
                    'stop_termination_indices': np.argwhere(stop['terminated']).tolist(),
                    'train_terminations': int(train['terminated'].sum()), 'train_truncation_indices': np.argwhere(train['truncated']).tolist()},
        'quaternion': {'raw_order': 'XYZW', 'scorer_order': 'WXYZ', 'bitwise_reorder_equal': True,
                       'norm_max_error': float(np.abs(np.linalg.norm(qxyzw.astype(float), axis=-1)-1).max()),
                       'first_W_min': float(qwxyz[0,:,0].min()), 'first_W_max': float(qwxyz[0,:,0].max()),
                       'independent_projected_gravity_max_error': gravity_error, 'independent_heading_max_error_rad': heading_error,
                       'synthetic_10deg_regression': synthetic},
        'commands': {'quiet_first_row': 1100, 'quiet_first_endpoint_s': float(stop['time_s'][1100]),
                     'quiet_last_endpoint_s': float(stop['time_s'][-1]), 'quiet_samples': 500,
                     'scorer_duration_s': 10., 'endpoint_span_s': float(stop['time_s'][-1]-stop['time_s'][1100]),
                     'quiet_actual_command_abs_max': float(np.abs(command[1100:]).max()),
                     'stop_target_abs_max': float(np.abs(stop['target_command'][800:]).max()),
                     'all_replicas_permanently_zero_first_row': last+1,
                     'all_replicas_permanently_zero_first_endpoint_s': float(stop['time_s'][last+1]),
                     'per_replica_last_nonzero_after_stop_row': last_nonzero},
        'learning': {'nominal_config_learning_rate': scalar_yaml(run/'train/agent.yaml', 'learning_rate'),
                     'updates': losses, 'intraupdate_learning_rate_or_gradient_share_not_recorded': True,
                     'target_delta_nonzero_rows': int(np.count_nonzero(delta)), 'target_delta_total_rows': int(delta.size),
                     'target_delta_rms_min_rad': float(delta.min()), 'target_delta_rms_max_rad': float(delta.max()),
                     'consecutive_target_replay_env_ids': ids.tolist(), 'consecutive_target_replay_controls': 47,
                     'consecutive_target_rms_max_error_rad': delta_error},
        'quiet': {'pass_count': sum(r['pass'] for r in quiet), 'replicas': 48, 'saved_verdict_parity': True,
                  'failed_bounds_per_replica': [r['failed_bounds'] for r in quiet]},
        'Stage2_complete': False,
    }

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--raw', type=Path, required=True)
    a = p.parse_args(); print(json.dumps(main(a.raw.resolve()), indent=2, allow_nan=False))
