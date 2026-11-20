"""Native status and loading compare the complete runtime identity."""

import unittest

from contracts.release import runtime_changes


class RuntimeIdentityTests(unittest.TestCase):
    def test_full_file_union_and_metadata_are_compared(self):
        before = {"schema": "v1", "files": {"same.py": "a", "gone.py": "b", "edit.py": "c"}}
        after = {"schema": "v2", "files": {"same.py": "a", "new.py": "d", "edit.py": "e"}}
        self.assertEqual(
            runtime_changes(before, after),
            {
                "added": ["new.py"],
                "removed": ["gone.py"],
                "modified": ["edit.py"],
                "metadata": ["schema"],
            },
        )
        self.assertFalse(any(runtime_changes(before, before).values()))

    def test_missing_manifest_is_not_equal_to_an_empty_manifest(self):
        self.assertTrue(any(runtime_changes(None, {"files": {}}).values()))
        self.assertTrue(any(runtime_changes({}, {"files": {}}).values()))


if __name__ == "__main__":
    unittest.main()
