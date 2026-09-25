"""The policy reward viewer embeds the scorer's per-control series beside the rollout."""
import json
from pathlib import Path
import re
import shutil
import tempfile
import unittest

import numpy as np

from locomotion import reward_scorer, reward_viewer
from locomotion.task import TaskConfig, measured_reward

ROOT = Path(__file__).resolve().parents[2]
TRACE = ROOT/'locomotion/tests/fixtures/control_trace.npz'
NOMINAL_HEIGHT = 0.09780231400684256
REWARDS = {'current': measured_reward}


def embedded(html):
    return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S).group(1))


class RewardViewerTests(unittest.TestCase):
    def test_series_match_the_scorer_row_for_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            view = reward_viewer.trace_view(TRACE, REWARDS, NOMINAL_HEIGHT, Path(temporary)/'page.html')
        trace = reward_scorer.load_trace(TRACE)
        reward, _ = reward_scorer.evaluate(trace, measured_reward, TaskConfig(), NOMINAL_HEIGHT,
                                           reward_scorer.root_com_local())
        rows = len(trace.data['command']) - 1
        self.assertEqual(len(view['time_s']), rows)
        np.testing.assert_allclose(view['rewards']['current']['recorded'], reward.numpy(), rtol=1e-4, atol=1e-6)
        self.assertEqual(len(view['contact']), 6)
        self.assertTrue(all(len(foot) == rows for foot in view['contact']))
        self.assertIsNone(view['video'])

    def test_video_is_linked_relative_to_the_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            evaluation = Path(temporary)/'run/evaluation_00'
            evaluation.mkdir(parents=True)
            shutil.copy(TRACE, evaluation/'control_trace.npz')
            (evaluation/'rollout.mp4').write_bytes(b'')
            view = reward_viewer.trace_view(evaluation, REWARDS, NOMINAL_HEIGHT, Path(temporary)/'pages/page.html')
        self.assertEqual(view['video'], '../run/evaluation_00/rollout.mp4')

    def test_video_frames_trail_control_time_by_one_frame(self):
        # evaluate.py appends frame k after control 2k+1, whose end time is (2k+2)*0.02 s.
        for frame in (0, 1, 250):
            self.assertAlmostEqual(frame/reward_viewer.VIDEO_FPS + reward_viewer.VIDEO_OFFSET_S,
                                   (2*frame + 2)*reward_viewer.CONTROL_DT_S)

    def test_replica_outside_the_trace_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, 'replica'):
            reward_viewer.trace_view(TRACE, REWARDS, NOMINAL_HEIGHT, Path(temporary)/'page.html', replica=1)

    def test_page_embeds_every_trace_and_escapes_markup(self):
        with tempfile.TemporaryDirectory() as temporary:
            evaluation = Path(temporary)/'<'/'script>'
            evaluation.mkdir(parents=True)
            self.assertIn('</script>', str(evaluation))
            shutil.copy(TRACE, evaluation/'control_trace.npz')
            html = reward_viewer.build_page([TRACE, evaluation], REWARDS, NOMINAL_HEIGHT, Path(temporary)/'page.html')
        self.assertEqual(html.count('</script>'), 2)
        payload = embedded(html)
        self.assertEqual(len(payload['traces']), 2)
        self.assertEqual(payload['legs'], ['lf', 'lm', 'lr', 'rf', 'rm', 'rr'])
        self.assertEqual(payload['rewards'], ['current'])

    def test_role_prefix_labels_a_rollout_for_the_scoreboard(self):
        self.assertEqual(reward_viewer.parse_evaluation('walking=/runs/a'), ('walking', Path('/runs/a')))
        self.assertEqual(reward_viewer.parse_evaluation('/runs/x=y'), (None, Path('/runs/x=y')))
        self.assertEqual(reward_viewer.parse_evaluation(TRACE), (None, TRACE))
        with tempfile.TemporaryDirectory() as temporary:
            html = reward_viewer.build_page([f'walking={TRACE}', TRACE], REWARDS, NOMINAL_HEIGHT, Path(temporary)/'page.html')
        self.assertEqual([t['role'] for t in embedded(html)['traces']], ['walking', None])
        self.assertIn('id="scoreboard"', html)

    def test_command_line_writes_the_page(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'viewer.html'
            reward_viewer.main([str(TRACE), '--nominal-height', str(NOMINAL_HEIGHT), '--output', str(output)])
            payload = embedded(output.read_text())
        self.assertEqual(payload['nominal_height_m'], NOMINAL_HEIGHT)
        self.assertEqual(len(payload['declared_gaps']), len(reward_scorer.DECLARED_GAPS))


if __name__ == '__main__':
    unittest.main()
