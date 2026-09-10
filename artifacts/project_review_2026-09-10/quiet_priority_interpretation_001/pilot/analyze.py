"""Read-only independent calculation from the actual smoke receipt and raw maps."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def summarize(receipt):
    assert receipt['complete'] is True and receipt['updates_completed'] == 50
    rows_all = []
    updates = []
    gradients = []
    previous_lr = 5e-5
    for uid, update in enumerate(receipt['optimizer_updates'], 1):
        assert uid == update['completed_update']
        rows = update['minibatches']
        assert len(rows) == 20
        counts = dict.fromkeys(('valid_pairs', 'quiet_pairs', 'moving_pairs'), 0)
        for mid, row in enumerate(rows, 1):
            assert row['update'] == uid and row['minibatch'] == mid
            kl = row['kl_mean']
            before, after = row['learning_rate_before'], row['learning_rate_after']
            assert all(math.isfinite(v) for v in (kl, before, after))
            assert math.isclose(before, previous_lr, rel_tol=1e-12, abs_tol=0)
            expected = max(1e-5, before / 1.5) if kl > .02 else min(1e-2, before * 1.5) if 0 < kl < .005 else before
            assert math.isclose(after, expected, rel_tol=1e-12, abs_tol=0), (uid, mid, after, expected)
            previous_lr = after
            pc = row['pair_counts']
            assert pc['valid_pairs'] == pc['quiet_pairs'] + pc['moving_pairs']
            assert 0 <= pc['valid_pairs'] <= 6144
            for key in counts:
                counts[key] += pc[key]
            g = row['gradient']
            assert (g is not None) == (uid in (1, 10, 25, 50) and mid in (1, 20))
            if g:
                norms = g['norms']
                assert all(math.isfinite(x) and x >= 0 for x in norms.values())
                assert math.isclose(g['component_sum_norm'], g['combined_actor_before_clip'], rel_tol=1e-6)
                assert math.isclose(g['combined_actor_after_clip'], 1., rel_tol=1e-6)
                factor = g['combined_actor_after_clip'] / g['combined_actor_before_clip']
                q = norms['quiet_temporal']
                known_dot = q * q + q * norms['ppo_actor'] * g['ppo_quiet_cosine']
                unknown_dot_bound = q * (norms['moving_temporal'] + norms['spatial'])
                gradients.append({
                    'update': uid, 'minibatch': mid, 'raw_gradient': g,
                    'quiet_norm_over_ppo_actor_norm': norms['quiet_temporal'] / norms['ppo_actor'],
                    'quiet_norm_over_spatial_norm': norms['quiet_temporal'] / norms['spatial'],
                    'quiet_norm_over_moving_temporal_norm': norms['quiet_temporal'] / norms['moving_temporal'],
                    'common_actor_clip_scale': factor,
                    'quiet_component_norm_after_common_clip': norms['quiet_temporal'] * factor,
                    'quiet_dot_total_gradient_interval': [known_dot - unknown_dot_bound, known_dot + unknown_dot_bound],
                    'dot_bound_method': 'Q dot (P+Q) from measured norms/cosine; absolute Q dot (M+S) bounded by ||Q|| (||M||+||S||). Applies before Adam, not its parameter update.',
                    'interpretation': 'Weighted loss-gradient components, not measured Adam parameter-update shares',
                })
        kls = [r['kl_mean'] for r in rows]
        updates.append({
            'update': uid, 'losses': update['losses'], 'pair_presentations': counts,
            'quiet_fraction_of_valid_pair_presentations': counts['quiet_pairs'] / counts['valid_pairs'],
            'kl_min': min(kls), 'kl_max': max(kls), 'kl_mean': statistics.mean(kls),
            'kl_above_adaptation_threshold_0_02': sum(k > .02 for k in kls),
            'lr_after_exact_literal_floor': sum(r['learning_rate_after'] == 1e-5 for r in rows),
            'lr_after_floor_with_1e_minus12_relative_tolerance': sum(math.isclose(r['learning_rate_after'], 1e-5, rel_tol=1e-12, abs_tol=0) for r in rows),
            'minibatches': rows,
        })
        rows_all.extend(rows)
    first, last = receipt['optimizer_updates'][0]['losses'], receipt['optimizer_updates'][-1]['losses']
    keys = ('caps_quiet_temporal_mean', 'caps_moving_temporal_mean', 'caps_temporal', 'caps_spatial', 'caps_weighted')
    return {
        'updates': updates, 'sparse_gradients': gradients,
        'loss_change_percent': {k: (last[k] / first[k] - 1) * 100 for k in keys},
        'kl_mean_all1000': statistics.mean(r['kl_mean'] for r in rows_all),
        'kl_above_0_02_all1000': sum(r['kl_mean'] > .02 for r in rows_all),
        'all1000_adaptive_learning_rate_transitions_replayed': True,
        'final_checkpoint_sha256': receipt['final_checkpoint_sha256'],
        'strict_reload_reported': receipt['reload']['passed'],
        'checkpoint_reload_independently_reexecuted_here': False,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists(), 'Fresh output required'
    root = a.repo / 'tmp/direct_omni_quiet_pilot_training_observation_001'
    receipt_path = root / 'training_receipt.json'
    assert sha(receipt_path) == 'c41560dbe34d29bbce40d1060bd1c8b0155cdcc27173dc2316806f8368ad9b11'
    receipt = json.loads(receipt_path.read_text())
    observation = json.loads((root / 'remote_observation.json').read_text())
    assert observation['training_receipt_text'].encode() == receipt_path.read_bytes()
    assert observation['accepted_train']['receipt_sha256'] == sha(receipt_path)
    assert observation['accepted_train']['checkpoint_sha256'] == receipt['final_checkpoint_sha256']
    assert observation['accepted_train']['source_manifest_sha256'] == 'ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62'
    result = summarize(receipt)
    windows = {}
    for name, updates in [('first5', receipt['optimizer_updates'][:5]), ('last5', receipt['optimizer_updates'][-5:])]:
        windows[name] = {'update_ids': [u['completed_update'] for u in updates],
                         'loss_means': {key: statistics.mean(u['losses'][key] for u in updates) for key in updates[0]['losses']}}
    result['disjoint_window_summary'] = windows
    result['last5_over_first5_change_percent'] = {k: (windows['last5']['loss_means'][k] / windows['first5']['loss_means'][k] - 1) * 100 for k in ('caps_quiet_temporal_mean', 'caps_moving_temporal_mean', 'caps_temporal', 'caps_weighted', 'caps_spatial')}
    all_rows = [row for u in receipt['optimizer_updates'] for row in u['minibatches']]
    result['lr_summary'] = {
        'min': min(r['learning_rate_after'] for r in all_rows),
        'max': max(r['learning_rate_after'] for r in all_rows),
        'floor_exact_count': sum(r['learning_rate_after'] == 1e-5 for r in all_rows),
        'floor_roundoff_count': sum(math.isclose(r['learning_rate_after'], 1e-5, rel_tol=1e-12, abs_tol=0) for r in all_rows),
        'final': all_rows[-1]['learning_rate_after'],
    }
    for update in result['updates']:
        loss = update['losses']
        q = (loss['caps_weighted'] - .1 * loss['caps_temporal'] - .1 * loss['caps_spatial']) / .9
        update['recovered_mean_weighted_loss_components'] = {'quiet_temporal': q, 'moving_temporal': .1 * (loss['caps_temporal'] - q), 'spatial': .1 * loss['caps_spatial']}
    result.update({
        'schema': 'direct_quiet_pilot50_training_only_review_v1',
        'training_receipt_sha256': sha(receipt_path),
        'source_manifest_sha256': observation['accepted_train']['source_manifest_sha256'],
        'checkpoint_sha256_verified_as_receipt_binding_only': receipt['final_checkpoint_sha256'],
        'checkpoint_bytes_locally_verified': False,
        'campaign_status_at_supplied_observation': observation['campaign_status'],
        'accepted_train_receipt': observation['accepted_train'],
        'full_terminal_source_cleanup_and_physical_scores_reviewed_here': False,
        'scope': 'Actual pilot training receipt and retained optimizer diagnostics only; no physical qualification',
        'Stage2_complete': False,
        'inputs_sha256': {p.relative_to(a.repo).as_posix(): sha(p) for p in root.iterdir() if p.is_file()},
    })
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('last5_over_first5_change_percent', 'lr_summary', 'kl_mean_all1000', 'kl_above_0_02_all1000')}))

if __name__ == '__main__':
    main()
