import gzip,hashlib,importlib.util,io,json,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('streaming_raw_test',HERE/'stream_raw.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class StreamTests(unittest.TestCase):
 def fixture(self,root,content):
  compressed=gzip.compress(content,mtime=0);parts=[]
  for i,start in enumerate(range(0,len(compressed),7)):
   p=root/f'part_{i:03d}';p.write_bytes(compressed[start:start+7]);parts.append({'path':p.name})
  return {'files':{'raw':{'encoding':'gzip_concatenated_parts','parts':parts}}}
 def test_cross_header_and_payload_boundaries_bounded_reads(self):
  content=(b'abcd\n'*100001)+b'partial last row'
  with tempfile.TemporaryDirectory()as t:
   root=Path(t);mapping=self.fixture(root,content)
   with m.open_raw(root,'raw',mapping)as stream:
    digest,size=m.hash_stream(stream)
   self.assertEqual(digest,hashlib.sha256(content).hexdigest());self.assertEqual(size,len(content))
   self.assertEqual({p.name for p in root.iterdir()},{p['path']for p in mapping['files']['raw']['parts']})
 def test_missing_reordered_and_corrupt_parts_fail(self):
  for mode in ('missing','reversed','corrupt'):
   with tempfile.TemporaryDirectory()as t:
    root=Path(t);mapping=self.fixture(root,b'testing data\n'*10000);parts=mapping['files']['raw']['parts']
    if mode=='missing':(root/parts[1]['path']).unlink()
    elif mode=='reversed':parts.reverse()
    else:(root/parts[-1]['path']).write_bytes(b'bad')
    with self.assertRaises(Exception):
     with m.open_raw(root,'raw',mapping)as stream:m.hash_stream(stream)
 def test_unsafe_storage_path_rejects(self):
  with tempfile.TemporaryDirectory()as t:
   with self.assertRaises(ValueError):m.safe(Path(t),'../outside')
if __name__=='__main__':unittest.main()
