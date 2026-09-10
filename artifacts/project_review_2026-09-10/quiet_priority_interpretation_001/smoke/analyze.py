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
    assert receipt['complete'] is True and receipt['updates_completed'] == 2
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
            assert 0 <= pc['valid_pairs'] <= 192
            for key in counts:
                counts[key] += pc[key]
            g = row['gradient']
            assert (g is not None) == (uid == 1 and mid in (1, 20))
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
        loss = update['losses']
        quiet_shared = (loss['caps_weighted'] - .1 * loss['caps_temporal'] - .1 * loss['caps_spatial']) / .9
        updates.append({
            'update': uid, 'losses': update['losses'], 'pair_presentations': counts,
            'quiet_fraction_of_valid_pair_presentations': counts['quiet_pairs'] / counts['valid_pairs'],
            'kl_min': min(kls), 'kl_max': max(kls), 'kl_mean': statistics.mean(kls),
            'kl_above_adaptation_threshold_0_02': sum(k > .02 for k in kls),
            'lr_after_exact_literal_floor': sum(r['learning_rate_after'] == 1e-5 for r in rows),
            'lr_after_floor_with_1e_minus12_relative_tolerance': sum(math.isclose(r['learning_rate_after'], 1e-5, rel_tol=1e-12, abs_tol=0) for r in rows),
            'minibatches': rows,
            'recovered_mean_weighted_loss_components': {'quiet_temporal': quiet_shared, 'moving_temporal': .1 * (loss['caps_temporal'] - quiet_shared), 'spatial': .1 * loss['caps_spatial']},
        })
        rows_all.extend(rows)
    first, last = (u['losses'] for u in receipt['optimizer_updates'])
    keys = ('caps_quiet_temporal_mean', 'caps_moving_temporal_mean', 'caps_temporal', 'caps_spatial', 'caps_weighted')
    return {
        'updates': updates, 'sparse_gradients': gradients,
        'loss_change_percent': {k: (last[k] / first[k] - 1) * 100 for k in keys},
        'kl_mean_all40': statistics.mean(r['kl_mean'] for r in rows_all),
        'kl_above_0_02_all40': sum(r['kl_mean'] > .02 for r in rows_all),
        'all40_adaptive_learning_rate_transitions_replayed': True,
        'final_checkpoint_sha256': receipt['final_checkpoint_sha256'],
        'strict_reload_reported': receipt['reload']['passed'],
        'checkpoint_reload_independently_reexecuted_here': False,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists(), 'Fresh output required; preserve previous evidence'
    terminal = a.repo / 'tmp/direct_omni_train_smoke003_terminal_001'
    native = a.repo / 'tmp/direct_omni_recovery_001/native_004'
    analysis = a.repo / 'tmp/direct_omni_train_smoke003_analysis_001'
    inputs = {}
    def bind(p):
        inputs[p.relative_to(a.repo).as_posix()] = sha(p)
        return json.loads(p.read_text())
    audit = bind(terminal / 'remote_terminal_audit.json')
    for name, item in audit['inventory'].items():
        p = terminal / name
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], name
        inputs[p.relative_to(a.repo).as_posix()] = sha(p)
    freeze = bind(native / 'FREEZE_SHA256.json')
    assert sha(native / 'FREEZE_SHA256.json') == '1ab678beeba6bf74978c210c4e38f13655e25728fe66e3007e3b1db5e19b6b20'
    for name, digest in freeze.items():
        assert sha(native / name) == digest, name
        inputs[(native / name).relative_to(a.repo).as_posix()] = digest
    analyzer_inputs = bind(analysis / 'INPUTS_SHA256.json')
    for name, digest in analyzer_inputs.items():
        p = Path(name)
        assert sha(p) == digest, name
        inputs[p.relative_to(a.repo).as_posix()] = digest
    analyzer = bind(analysis / 'report.json')
    inputs[(analysis / 'REPORT.md').relative_to(a.repo).as_posix()] = sha(analysis / 'REPORT.md')
    campaign = bind(terminal / 'run/campaign.json')
    receipt = bind(terminal / 'run/train/training_receipt.json')
    result = summarize(receipt)
    assert campaign['identity']['source_manifest_sha256'] == analyzer['source_manifest_sha256'] == 'ef962070bd770d03553eb824fb6caccada7d6a06ac26d2f180bad068f0a9cc62'
    assert sha(terminal / 'run/train/policy/final.pt') == result['final_checkpoint_sha256']
    assert sha(terminal / 'run/train/training_receipt.json') == campaign['accepted_phases']['train']['receipt_sha256']
    for expected, actual in zip(analyzer['optimizer_diagnostics']['updates'], result['updates']):
        assert expected['losses'] == actual['losses'] and expected['pair_presentations'] == actual['pair_presentations']
    result.update({
        'schema': 'direct_quiet_smoke003_independent_diagnostics_v1',
        'source_manifest_sha256': campaign['identity']['source_manifest_sha256'],
        'plan_sha256': campaign['identity']['plan_sha256'],
        'original_checkpoint_sha256': campaign['identity']['checkpoint_sha256'],
        'campaign_sha256': sha(terminal / 'run/campaign.json'),
        'training_receipt_sha256': sha(terminal / 'run/train/training_receipt.json'),
        'native_freeze_sha256': sha(native / 'FREEZE_SHA256.json'),
        'raw_local_payloads_verified_against_remote_audit': len(audit['inventory']),
        'native_owner_payloads_verified': len(freeze),
        'source_scope': 'Local native owner hashes verified; executed source identity is bound by actual campaign/phase receipts. No new remote source audit.',
        'Stage2_complete': False, 'source_mutated': False, 'GPU_used': False,
        'inputs_sha256': inputs,
    })
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('raw_local_payloads_verified_against_remote_audit','native_owner_payloads_verified','kl_above_0_02_all40','final_checkpoint_sha256')}))

if __name__ == '__main__':
    main()
