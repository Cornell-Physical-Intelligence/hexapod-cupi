import ast,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
class NvidiaCLI(unittest.TestCase):
 def test_literal_native_query_argv_and_correct_parent(self):
  def argv(path):
   tree=ast.parse(path.read_text())
   return [ast.literal_eval(n.args[0]) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='call' and n.args and isinstance(n.args[0],ast.List) and n.args[0].elts and isinstance(n.args[0].elts[0],ast.Constant) and n.args[0].elts[0].value=='nvidia-smi']
  expected=[['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits']]
  self.assertEqual(argv(HERE/'launch_guarded_remote.py'),expected)
  self.assertEqual(argv(HERE/'inputs/guard_parent.py'),expected)
  self.assertNotEqual(argv(HERE/'inputs/failed_guard001.py'),expected)
if __name__=='__main__':unittest.main()
