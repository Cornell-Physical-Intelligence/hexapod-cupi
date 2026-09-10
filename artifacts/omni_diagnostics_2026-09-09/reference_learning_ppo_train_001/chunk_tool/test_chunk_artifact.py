import hashlib,json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from chunk_artifact import split,consume,reconstruct,MAX_CHUNK_BYTES
class ChunkTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
 def make(self,data=b'abcdefghijklmnopqrstuvwxyz'):
  source=self.root/'original.npz';source.write_bytes(data);chunks=self.root/'chunks';split(source,chunks,'smoke/trace.npz',8);return source,chunks
 def test_roundtrip_empty_exact_boundary_and_multiple_chunks(self):
  for n in (0,1,8,9,32,37):
   with self.subTest(n=n),tempfile.TemporaryDirectory(dir=self.root) as td:
    p=Path(td);data=bytes(range(n));src=p/'source';src.write_bytes(data);m=split(src,p/'chunks','smoke/trace.npz',8);reconstruct(p/'chunks',p/'copy')
    self.assertEqual((p/'copy').read_bytes(),data);self.assertEqual(src.read_bytes(),data);self.assertEqual(m['original_sha256'],hashlib.sha256(data).hexdigest());self.assertTrue(all(r['bytes']<=8 for r in m['chunks']))
 def test_changed_chunk_or_whole_hash_rejects(self):
  _,c=self.make();part=c/'part-000000.bin';original=part.read_bytes();part.write_bytes(b'X'+original[1:])
  with self.assertRaises(ValueError):consume(c)
  part.write_bytes(original);m=json.loads((c/'manifest.json').read_text());m['original_sha256']='0'*64;(c/'manifest.json').write_text(json.dumps(m))
  with self.assertRaises(ValueError):consume(c)
 def test_missing_reordered_traversal_and_extra_files_reject(self):
  for kind in ('missing','reordered','traversal','extra'):
   with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir=self.root) as td:
    p=Path(td);s=p/'s';s.write_bytes(b'0123456789');c=p/'c';split(s,c,'a',4);m=json.loads((c/'manifest.json').read_text())
    if kind=='missing':(c/'part-000000.bin').unlink()
    elif kind=='reordered':m['chunks'].reverse();(c/'manifest.json').write_text(json.dumps(m))
    elif kind=='traversal':m['chunks'][0]['path']='../s';(c/'manifest.json').write_text(json.dumps(m))
    else:(c/'extra').write_bytes(b'x')
    with self.assertRaises(ValueError):consume(c)
 def test_existing_output_or_chunk_dir_never_overwritten(self):
  src,c=self.make();output=self.root/'existing';output.write_bytes(b'keep')
  with self.assertRaises(FileExistsError):reconstruct(c,output)
  with self.assertRaises(FileExistsError):split(src,c,'a')
  self.assertEqual(output.read_bytes(),b'keep')
 def test_corrupt_reconstruction_leaves_no_output_or_partial(self):
  _,c=self.make();(c/'part-000001.bin').write_bytes(b'xxxxxxxx');out=self.root/'result'
  with self.assertRaises(ValueError):reconstruct(c,out)
  self.assertFalse(out.exists());self.assertFalse(list(self.root.glob('*.partial-*')))
 def test_size_limits_and_boolean_rejected(self):
  s=self.root/'s';s.write_bytes(b'123')
  for size in (0,-1,MAX_CHUNK_BYTES+1,True):
   with self.assertRaises(ValueError):split(s,self.root/'c','a',size)
 def test_truncated_during_read_rejects_without_looping(self):
  source=self.root/'source';source.write_bytes(b'12345678');original_open=Path.open
  def open_mutating(path,mode='r',*args,**kwargs):
   if path==source and mode=='rb':
    with original_open(source,'wb'):pass
   return original_open(path,mode,*args,**kwargs)
  with patch.object(Path,'open',open_mutating):
   with self.assertRaisesRegex(ValueError,'truncated'):split(source,self.root/'c','a',4)
  self.assertFalse((self.root/'c/manifest.json').exists())
 def test_symlink_or_inside_output_rejected(self):
  _,c=self.make();original=c/'part-000000.bin';saved=self.root/'saved';original.rename(saved);original.symlink_to(saved)
  with self.assertRaises(ValueError):consume(c)
  with self.assertRaises(ValueError):reconstruct(c,c/'result')
if __name__=='__main__':unittest.main()
