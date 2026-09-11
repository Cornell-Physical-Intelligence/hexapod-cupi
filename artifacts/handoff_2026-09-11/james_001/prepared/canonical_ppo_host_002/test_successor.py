import ast
import hashlib
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
class SuccessorTests(unittest.TestCase):
    def test_runtime_exact_parent_after_three_declared_pin_changes(self):
        old=(HERE/'inputs/PARENT_HOST.py').read_text()
        new=(HERE/'launch_ppo_spark.py').read_text()
        replacements={
            '1e173946fe2548a82207791528f503ac6d12766fd5b50f25746e6edf92704617':'cc62938666b7b394e119fe05664767b2a265928ca46c8e15afd97312611f5d63',
            'e922ff1312b5106f957b384874ca5eb6ed4a9ed6566c0ef9a0602247a49670f0':'acb589708fcba8be2f6ef64171795889eca8b6583fdbb4bc2fc25d61717b0467',
            '649ccda1bdef20bc967ae78f68a992cc8001f4ed7009f78cab5133cc67b29d8f':'35af5100e39980f55a3f5e19654e4550b3c3fb6da114b64fbe0d143817242cf3'}
        for current,parent in replacements.items():
            self.assertEqual(new.count(current),1);new=new.replace(current,parent)
        self.assertEqual(new,old)
        self.assertEqual(hashlib.sha256(old.encode()).hexdigest(),'f5ed526382b52fc0677b52613cd79d5273a6122b5d8c1af7f50aa188a42dfdbf')
    def test_every_runtime_function_ast_unchanged(self):
        functions=lambda p:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        self.assertEqual(functions(HERE/'inputs/PARENT_HOST.py'),functions(HERE/'launch_ppo_spark.py'))
if __name__=='__main__':unittest.main()
