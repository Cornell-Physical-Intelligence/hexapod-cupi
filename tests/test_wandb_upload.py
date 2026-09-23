"""Check the host W&B upload without contacting W&B or importing wandb."""
import importlib.util, json, tempfile, unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1]/'tools/wandb_upload.py'
spec = importlib.util.spec_from_file_location('wandb_upload', PATH)
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)


class FakeRun:
    def __init__(self):
        self.logged, self.summary, self.finished = [], {}, False

    def log(self, values, step=None):
        self.logged.append((step, values))

    def finish(self):
        self.finished = True


class FakeWandb:
    def __init__(self):
        self.kwargs, self.run = None, FakeRun()

    def init(self, **kwargs):
        self.kwargs = kwargs
        return self.run

    def Video(self, path):
        return ('video', path)


class WandbUploadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run_dir = Path(self.directory.name)/'train_seed7'
        self.run_dir.mkdir()
        self.write('state.json', {'mode': 'train', 'status': 'completed', 'updates': 2,
                                  'identity': {'seed': 7, 'ppo_config': {'logger': 'tensorboard'}}})
        rows = [{'update': 1, 'loss': {'surrogate': .5}, 'learning_rate': 1e-3,
                 'task': {'task_reward_mean': .2, 'diagnostic_scope': 'text', 'moving_signed_command_direction_speed_mps': None}},
                {'update': 2, 'loss': {'surrogate': .4}, 'learning_rate': 1e-3,
                 'task': {'task_reward_mean': .3, 'zero_hold_joint_rate_rms_rad_s': [.01, .02]}}]
        (self.run_dir/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        self.write('force_metrics.json', {'status': 'available', 'windows': {'full': {'mean_n': 73.2}}})
        (self.run_dir/'evaluation').mkdir()
        (self.run_dir/'evaluation/rollout.mp4').write_bytes(b'mp4')

    def write(self, name, value):
        (self.run_dir/name).write_text(json.dumps(value)+'\n')

    def test_flatten_keeps_finite_numbers_and_drops_text(self):
        self.assertEqual(tool.flatten({'a': {'b': 1, 'c': 'x', 'd': None, 'e': [2., float('nan')], 'f': True}}),
                         {'a/b': 1, 'a/e/0': 2., 'a/f': 1})

    def test_records_bind_files_and_order_updates(self):
        config, steps, summary, videos = tool.records(self.run_dir)
        self.assertEqual([s for s, _ in steps], [1, 2])
        self.assertEqual(steps[0][1], {'loss/surrogate': .5, 'learning_rate': 1e-3, 'task/task_reward_mean': .2})
        self.assertEqual(steps[1][1]['task/zero_hold_joint_rate_rms_rad_s/1'], .02)
        self.assertEqual(config['files_sha256']['metrics.jsonl'], tool.sha(self.run_dir/'metrics.jsonl'))
        self.assertNotIn('ppo_config.json', config['files_sha256'])
        self.assertEqual(summary, {'updates': 2, 'force_metrics/windows/full/mean_n': 73.2})
        self.assertEqual(videos, [self.run_dir/'evaluation/rollout.mp4'])

    def test_unfinished_runs_and_repeated_updates_are_rejected(self):
        self.write('state.json', {'mode': 'train', 'status': 'running'})
        with self.assertRaises(ValueError):
            tool.records(self.run_dir)
        self.write('state.json', {'mode': 'train', 'status': 'failed'})
        (self.run_dir/'metrics.jsonl').write_text('{"update": 2}\n{"update": 2}\n')
        with self.assertRaises(ValueError):
            tool.records(self.run_dir)

    def test_upload_keeps_files_in_the_run_and_logs_each_update(self):
        fake = FakeWandb()
        result = tool.upload(self.run_dir, 'hexapod-amp', mode='offline', wandb=fake)
        self.assertEqual(fake.kwargs['dir'], str(self.run_dir.resolve()))
        self.assertEqual((fake.kwargs['project'], fake.kwargs['name'], fake.kwargs['mode'], fake.kwargs['job_type']),
                         ('hexapod-amp', 'train_seed7', 'offline', 'train'))
        self.assertEqual([step for step, _ in fake.run.logged], [1, 2])
        self.assertEqual(fake.run.summary['force_metrics/windows/full/mean_n'], 73.2)
        self.assertTrue(fake.run.finished)
        self.assertEqual((result['steps'], result['videos']), (2, 0))
        videos = FakeWandb()
        self.assertEqual(tool.upload(self.run_dir, 'hexapod-amp', videos=True, wandb=videos)['videos'], 1)
        self.assertEqual(videos.run.logged[-1], (None, {'video/evaluation/rollout.mp4':
                                                       ('video', str((self.run_dir/'evaluation/rollout.mp4').resolve()))}))

    def test_invalid_project_and_mode_are_rejected(self):
        for project, mode in (('bad name', 'online'), ('hexapod', 'disabled')):
            with self.assertRaises(ValueError):
                tool.upload(self.run_dir, project, mode=mode, wandb=FakeWandb())


if __name__ == '__main__':
    unittest.main()
