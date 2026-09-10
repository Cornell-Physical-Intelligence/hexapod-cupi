"""The successor changes identity constants only, never launch behavior."""
import ast,hashlib,json,unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'direct_omni_train_host_001'
OLD='332fad29e0a0f88ba07012700f2a21c3a4f0453f3d71224da84fcc477e48394f'

class PinDeltaTests(unittest.TestCase):
    def test_exact_frozen_parent_inventory(self):
        sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        self.assertEqual(sha(PARENT/'FREEZE_SHA256.json'),OLD)
        manifest=json.loads((PARENT/'FREEZE_SHA256.json').read_text())
        self.assertEqual({p.relative_to(PARENT).as_posix():sha(p) for p in PARENT.rglob('*') if p.is_file() and p.name!='FREEZE_SHA256.json'},manifest)

    def test_only_two_identity_constants_change(self):
        a=ast.parse((PARENT/'launch_train_spark.py').read_text());b=ast.parse((HERE/'launch_train_spark.py').read_text())
        allowed={'SOURCE_MAP','CONTRACT_FREEZE'}
        def normalize(tree):
            found=[]
            for node in tree.body:
                if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in allowed:
                    found.append(node.targets[0].id);node.value=ast.Constant('VERSION_PIN')
            self.assertEqual(set(found),allowed)
            return ast.dump(tree)
        self.assertEqual(normalize(a),normalize(b))

    def test_entry_and_legacy_map_are_byte_identical(self):
        for name in ('run_train_entry.py','LEGACY_RUNTIME_SHA256.json'):
            self.assertEqual((PARENT/name).read_bytes(),(HERE/name).read_bytes())

if __name__=='__main__':unittest.main()
