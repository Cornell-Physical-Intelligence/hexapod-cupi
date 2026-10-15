"""The GPU lock files must stay readable by every account whatever the creator's umask."""
import os
from pathlib import Path
import stat
import tempfile
import unittest

import locomotion.tools.launch as launch


class LockPermissionTests(unittest.TestCase):
    def test_created_lock_is_world_readable_under_a_strict_umask(self):
        previous = os.umask(0o027)
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'gpu.lock'
                os.close(launch.open_lock(path))
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)
                # An existing file opens without O_CREAT and keeps its mode.
                os.chmod(path, 0o640)
                os.close(launch.open_lock(path))
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)
        finally:
            os.umask(previous)


if __name__ == '__main__':
    unittest.main()
