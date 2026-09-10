import unittest
import run_inspection as entry
class Binding(unittest.TestCase):
 def test_actual_string_only_backend_contract(self):
  class StringOnly:
   def create_sdf_shape_view(self,pattern,sample_point_counts):
    if not isinstance(pattern,str):raise TypeError('pattern: str, sample_point_counts: int=[]')
    self.call=(pattern,sample_point_counts);return 'view'
  backend=StringOnly()
  with self.assertRaises(TypeError):backend.create_sdf_shape_view(['/Robot/body/collisions/part_0000'],1)
  self.assertEqual(entry.create_sdf_view(backend),'view')
  self.assertEqual(backend.call,('/Robot/*/collisions/part_*',1))
if __name__=='__main__':unittest.main()
