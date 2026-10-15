"""Check source locations and revision links in the generated developer map."""
import hashlib
from pathlib import Path
import tempfile
import unittest

from tools import code_map

ROOT = Path(__file__).resolve().parents[1]


class CodeMapTests(unittest.TestCase):
    def test_map_contains_nested_definitions_without_retired_wrappers(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = code_map.build(ROOT, tmp)
            rows = {row['path']: row for row in value['modules']}
            native = rows['locomotion/gait_clock/env.py']
            method = next(row for row in native['symbols'] if row['name'] == 'LocomotionEnv.step')
            source = ROOT/native['path']
            self.assertIn('def step(', source.read_text().splitlines()[method['line']-1])
            self.assertIn('/'+value['revision']+'/', method['url'])
            self.assertEqual(native['sha256'], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(native['area'], 'gait_clock')
            self.assertNotIn('locomotion/env.py', rows)
            self.assertIn('locomotion/gait_clock/control.py', native['depends_on'])
            self.assertTrue(all(target in rows for row in rows.values() for target in row['depends_on']))
            self.assertEqual(rows['locomotion/tests/test_native_ranges.py']['area'], 'tests')
            self.assertFalse(any('/fixtures/' in path for path in rows))
            self.assertTrue((Path(tmp)/'code-map.json').is_file())
            html = (Path(tmp)/'index.html').read_text()
            self.assertNotIn('__DATA__', html)
            self.assertNotIn('audit.html', html)
            self.assertNotIn('id="compatibility"', html)
            self.assertIn('aria-label="Search code map"', html)


if __name__ == '__main__':
    unittest.main()
