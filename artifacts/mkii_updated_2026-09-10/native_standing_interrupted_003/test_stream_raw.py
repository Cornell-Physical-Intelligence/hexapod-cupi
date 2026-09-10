"""Local I/O counterexamples for the publication reader; no native dependencies."""
from pathlib import Path
import gzip,hashlib,json,tempfile,unittest
from stream_raw import open_raw,verify_raw

class StreamTests(unittest.TestCase):
 def fixture(self,root,encoding='gzip'):
  data=(b'exact recorded prefix\n'*1000)+b'partial tail';name='run/contacts.jsonl';path=root/'terminal/run/contacts.jsonl.gz';path.parent.mkdir(parents=True)
  stored=gzip.compress(data,mtime=0)if encoding=='gzip'else data;path.write_bytes(stored)
  raw={name:{'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data)}}
  enc={name:{**raw[name],'storage_path':'run/contacts.jsonl.gz','encoding':encoding}}
  storage={'original_raw_files':1,'original_raw_bytes':len(data),'stored_raw_bytes':len(stored),'files':{name:{**enc[name],'storage_path':'terminal/run/contacts.jsonl.gz','stored_sha256':hashlib.sha256(stored).hexdigest(),'stored_size_bytes':len(stored)}}}
  for f,value in [('terminal/RAW_SHA256.json',raw),('terminal/RAW_ENCODING.json',enc),('terminal/audit.json',{'raw_inventory':raw}),('RAW_STORAGE.json',storage)]:
   (root/f).write_text(json.dumps(value))
  return name,data,storage,path
 def test_identity_and_gzip_exact_partial_tail_preserved(self):
  for encoding in ['identity','gzip']:
   with tempfile.TemporaryDirectory()as d:
    root=Path(d);name,data,_,_=self.fixture(root,encoding)
    with open_raw(root,name)as f:self.assertEqual(f.read(),data)
    result=verify_raw(root);self.assertEqual(result['original_raw_bytes_verified'],len(data));self.assertEqual(result['temporary_decompressed_files'],0)
 def test_changed_stored_content_refused(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);_,_,_,path=self.fixture(root);path.write_bytes(path.read_bytes()+b'extra')
   with self.assertRaisesRegex(AssertionError,'stored bytes'):verify_raw(root)
 def test_truncated_gzip_and_path_escape_refused(self):
  with tempfile.TemporaryDirectory()as d:
   root=Path(d);name,_,storage,path=self.fixture(root);path.write_bytes(path.read_bytes()[:-8])
   with self.assertRaises(EOFError):
    with open_raw(root,name)as f:f.read()
   storage['files'][name]['storage_path']='../foreign'
   with self.assertRaisesRegex(ValueError,'Unsafe'):
    with open_raw(root,name,storage):pass

if __name__=='__main__':unittest.main()
