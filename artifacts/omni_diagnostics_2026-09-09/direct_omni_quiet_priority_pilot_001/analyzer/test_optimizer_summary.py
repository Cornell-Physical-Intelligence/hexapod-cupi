"""NumPy/stdlib only: exact recorded trends, malformed receipts and scorer preservation."""
import ast
import copy
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze
from optimizer_summary import LOSS_KEYS, NORM_KEYS, human, summarize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NATIVE = ROOT/'tmp/direct_omni_recovery_001/native_004'
PARENT = ROOT/'tmp/direct_omni_train_analysis_002/analyze.py'
SCORERS = ('Inputs', 'require', 'close', 'cadence', 'named_rows', 'trace_rates',
           'constant', 'compare_constant', 'recompute_quiet', 'stop', 'compare_stop',
           'parse_timings', 'training')


def fixture(updates=2):
    receipt = {'optimizer_diagnostics_schema': 'direct315_actor_gradients_v1',
               'complete': True, 'updates_completed': updates,
               'selection': {'updates': updates, 'replicas': 32, 'branch': 'quiet_priority'},
               'optimizer_updates': []}
    for u in range(1, updates+1):
        rows = []
        for m in range(1, 21):
            # First/last LR both floor while an interior interval rises: endpoint
            # only analysis must not incorrectly describe an all-floor run.
            before = 1.5e-5 if m == 10 else 1e-5
            after = 1.5e-5 if m == 9 else 1e-5
            g = None
            if u in (1, 10, 25, 50) and m in (1, 20):
                g = {'norms': dict(zip(NORM_KEYS, (.2, .3, .1, .01))),
                     'ppo_quiet_cosine': -.25, 'component_sum_norm': .4,
                     'combined_actor_before_clip': .4, 'combined_actor_after_clip': .4}
            rows.append({'update': u, 'minibatch': m, 'kl_mean': .001*m,
                         'learning_rate_before': before, 'learning_rate_after': after,
                         'pair_counts': {'valid_pairs': 192, 'quiet_pairs': 144, 'moving_pairs': 48},
                         'gradient': g})
        receipt['optimizer_updates'].append({'completed_update': u, 'learning_rate': 1e-5,
            'optimizer_wall_seconds': .05, 'losses': {k: float(u)/10 for k in LOSS_KEYS},
            'minibatches': rows})
    return receipt


class OptimizerSummaryTests(unittest.TestCase):
    def test_exact_scorer_and_gate_ast_preserved(self):
        def trees(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        old, new = trees(PARENT), trees(HERE/'analyze.py')
        for name in SCORERS: self.assertEqual(old[name], new[name], name)
        def assignment(path, name):
            return next(ast.dump(n.value, include_attributes=False) for n in ast.parse(path.read_text()).body
                        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
        for name in ('QUIET_GATES', 'CONSTANT_METRICS', 'ORIGINAL'):
            self.assertEqual(assignment(PARENT, name), assignment(HERE/'analyze.py', name))

    def test_all_minibatch_lr_and_sparse_components_not_endpoint_inference(self):
        receipt = fixture(); before = copy.deepcopy(receipt)
        result = summarize(receipt)
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['update_end_learning_rates'], [1e-5, 1e-5])
        self.assertEqual(result['learning_rate_after']['max'], 1.5e-5)
        self.assertEqual(result['learning_rate_at_existing_floor_count'], 38)
        self.assertEqual(result['minibatches_retained'], 40)
        self.assertEqual(result['sparse_gradient_rows_retained'], 2)
        self.assertEqual(result['quiet_valid_pair_fraction'], .75)
        self.assertEqual(result['loss_windows']['initial_update_ids'], [1])
        self.assertEqual(result['loss_windows']['final_update_ids'], [2])
        self.assertEqual(result['sparse_gradients'][0]['ppo_quiet_cosine'], -.25)
        self.assertEqual(receipt, before)  # Analysis never mutates receipt data.
        self.assertIn('Aggregate temporal losses are not quiet target-step p95', '\n'.join(human(result)))
        self.assertFalse(result['admission'])

    def test_fifty_updates_sparse_inventory_and_native_contract_agree(self):
        # The frozen validator is stdlib-only; use its AST to avoid adding any
        # runtime dependency to this analyzer or importing robot/training code.
        tree = ast.parse((NATIVE/'direct_contract.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('finite', 'validate_optimizer_diagnostics')]
        scope = {'math': math}; exec(compile(ast.Module(body=functions, type_ignores=[]), '<frozen contract>', 'exec'), scope)
        receipt = fixture(50)
        expected = scope['validate_optimizer_diagnostics'](receipt, receipt['selection'])
        result = summarize(receipt)
        self.assertEqual(result['errors'], [])
        self.assertEqual(result['sparse_gradient_rows_retained'], expected['sparse_actor_gradient_rows'])
        self.assertEqual(result['minibatches_retained'], expected['minibatches'])
        self.assertEqual(result['loss_windows']['initial_update_ids'], list(range(1, 11)))
        self.assertEqual(result['loss_windows']['final_update_ids'], list(range(41, 51)))

    def test_missing_nonfinite_partial_and_zero_norm_diagnostics_preserved(self):
        for receipt in (None, {}, {'complete': False, 'updates_completed': 0, 'optimizer_updates': None}):
            result = summarize(receipt); self.assertTrue(result['errors'])
            self.assertFalse(result['diagnostic_inventory_complete']); self.assertFalse(result['admission'])
            self.assertIn('unavailable', '\n'.join(human(result)))
            json.dumps(result, allow_nan=False)
        receipt = fixture(); receipt['complete'] = False
        receipt['optimizer_updates'][1]['minibatches'].pop()
        receipt['optimizer_updates'][0]['losses']['caps_quiet_temporal_mean'] = float('nan')
        receipt['optimizer_updates'][0]['minibatches'][0]['gradient']['norms']['quiet_temporal'] = 0.
        receipt['optimizer_updates'][0]['minibatches'][0]['gradient']['ppo_quiet_cosine'] = None
        receipt['optimizer_updates'][1]['minibatches'][5]['kl_mean'] = float('inf')
        result = summarize(receipt)
        self.assertTrue(result['errors']); self.assertEqual(result['minibatches_retained'], 39)
        self.assertEqual(result['updates_retained'], 2)
        self.assertIsNone(result['updates'][0]['losses']['caps_quiet_temporal_mean'])
        self.assertIsNone(result['sparse_gradients'][0]['ppo_quiet_cosine'])
        json.dumps(result, allow_nan=False); human(result)

    def test_missing_campaign_and_receipt_write_forensic_files(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            result = analyze.analyze(base/'absent_campaign', base/'absent_cold', base/'review')
            self.assertTrue(result['errors']); self.assertFalse(result['evidence_verified'])
            self.assertFalse(result['Stage2_complete']); self.assertFalse(result['automatic_continuation'])
            for name in ('report.json', 'REPORT.md', 'INPUTS_SHA256.json'): self.assertTrue((base/'review'/name).is_file())
            self.assertIn('No usable training receipt', (base/'review/REPORT.md').read_text())

    def test_completed_claim_missing_reload_and_raw_is_never_admitted(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d); (base/'run/train').mkdir(parents=True)
            receipt = fixture(); receipt.update(reload=None, audit=None)
            (base/'run/train/training_receipt.json').write_text(json.dumps(receipt))
            (base/'run/campaign.json').write_text(json.dumps({'status': 'completed', 'identity': None, 'note': float('nan')}))
            result = analyze.analyze(base/'run', base/'cold', base/'review')
            self.assertEqual(result['training_receipt_forensics']['updates_completed'], 2)
            self.assertIsNone(result['training_receipt_forensics']['reload_passed'])
            self.assertFalse(result['evidence_verified']); self.assertFalse(result['Stage2_complete'])
            self.assertTrue(any('Nonfinite report field' in e for e in result['errors']))
            self.assertIsNone(result['campaign']['note'])
            self.assertNotIn('training', result)
            json.dumps(result, allow_nan=False)
            self.assertIn('producer reports 2 updates', (base/'review/REPORT.md').read_text())

    def test_exact_new_campaign_binding_and_old_source_rejection(self):
        # This tests the orchestration/admission seam only. Physical scorers have
        # separate raw-fixture tests and exact parent AST equality above.
        with tempfile.TemporaryDirectory() as d:
            base = Path(d); run = base/'run'; cold = base/'cold'; cold.mkdir()
            phases = ['standing', 'train', 'final_constant', 'final_stop']
            selected = {'schema': analyze.SCHEMA, 'allocation': 'smoke', 'branch': 'quiet_priority',
                        'replicas': 32, 'controls_per_update': 24, 'updates': 2,
                        'caps': {'temporal_weight': .1, 'spatial_weight': .1, 'quiet_temporal_weight': 1., 'noise_scale': 1., 'noise_seed': 1157}}
            identity = {'schema': analyze.SCHEMA, 'source_manifest_sha256': analyze.SOURCE,
                        'plan_sha256': analyze.PLAN, 'checkpoint_sha256': analyze.ORIGINAL,
                        'actor_width': 315, 'critic_width': 318, 'selection': selected}
            campaign = {'identity': identity, 'allocation': 'smoke', 'branch': 'quiet_priority',
                        'status': 'completed', 'terminal_inputs_unchanged': True,
                        'planned_phases': phases, 'accepted_phases': {}}
            (run/'jobs').mkdir(parents=True)
            for phase in phases:
                (run/phase).mkdir()
                state = {'status': 'completed', 'plan_sha256': analyze.PLAN,
                         'urdf_sha256': 'e916b1ebbb2720c90f23974bbffa5da120b32d2e5409fe419fb1492b8556746c',
                         'runtime_binding': {'source_manifest_sha256': analyze.SOURCE,
                             'runtime_tree_sha256': 'abe4e3542c7af7093f4a203b0db3631706a0b981f77befb40b243559b7a69280'}}
                (run/phase/'state.json').write_text(json.dumps(state))
                (run/'jobs'/f'{phase}.json').write_text(json.dumps({'status': 'completed', 'cleanup_checked': True, 'exit_code': 0}))
                campaign['accepted_phases'][phase] = {'state_sha256': analyze.sha(run/phase/'state.json')}
            receipt = fixture(); receipt['selection'] = selected
            (run/'train/training_receipt.json').write_text(json.dumps(receipt))
            for name in ('final_constant/diagnostics.json', 'final_stop/stop_diagnostics.json'):
                (run/name).write_text('{}')
            (cold/'campaign.json').write_text(json.dumps({'identity': {'source_manifest_sha256': '4671eab012aaf79ce49769cfdc1667142237a4b4a45966fd0399a9caa900c51b'}, 'status': 'completed', 'terminal_inputs_unchanged': True}))
            result_training = {'complete': True, 'checkpoint_sha256': analyze.ORIGINAL, 'updates_completed': 2, 'controls': 48,
                'replicas': 32, 'transitions': 1536, 'training_wrapper_wall_seconds': 2., 'wrapper_transitions_per_second': 768.,
                'strict_reload': {'passed': True}, 'per_row': [{'terminations': 0, 'truncations': 0, 'nonfoot_env_steps': 0,
                                                           'requested_peak_nm': 1., 'applied_peak_nm': 1.}]}
            with patch.object(analyze, 'constant', return_value={'checkpoint_sha256': analyze.ORIGINAL}), \
                 patch.object(analyze, 'compare_constant', return_value=[]), \
                 patch.object(analyze, 'stop', return_value={'checkpoint_sha256': analyze.ORIGINAL, 'quiet_passed_replicas': 0, 'cases': []}), \
                 patch.object(analyze, 'training', return_value=result_training):
                (run/'campaign.json').write_text(json.dumps(campaign))
                result = analyze.analyze(run, cold, base/'bound')
                self.assertEqual(result['errors'], []); self.assertTrue(result['evidence_verified'])
                self.assertFalse(result['Stage2_complete']); self.assertFalse(result['automatic_continuation'])
                campaign['identity']['source_manifest_sha256'] = '64144b2c2466f17c809554ef53f82b736b4199136cc821451d7ac2788752bd2e'
                (run/'campaign.json').write_text(json.dumps(campaign))
                result = analyze.analyze(run, cold, base/'old_source')
                self.assertFalse(result['evidence_verified'])
                self.assertTrue(any('Unbound native source/plan' in e for e in result['errors']))

    def test_runtime_imports_no_torch_rsl_or_cuda(self):
        program = ('import sys; sys.path.insert(0,'+repr(str(HERE))+'); import analyze,optimizer_summary; '
                   'assert not any(k == "torch" or k.startswith("torch.") or k == "rsl_rl" or k.startswith("rsl_rl.") for k in sys.modules)')
        subprocess.run([sys.executable, '-B', '-c', program], check=True, capture_output=True, text=True)


if __name__ == '__main__': unittest.main()
