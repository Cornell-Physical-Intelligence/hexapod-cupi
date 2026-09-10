"""Bounded-memory access to exact logical raw evidence; never invokes native code."""
from pathlib import Path
from contextlib import contextmanager
import gzip,hashlib,json

BLOCK=1<<20

def safe(root,name):
 root=Path(root).resolve();name=Path(name);p=root/name
 if name.is_absolute()or '..'in name.parts or p.is_symlink()or not p.resolve().is_relative_to(root):raise ValueError('Unsafe stored path')
 return p

@contextmanager
def open_raw(root,name,storage=None):
 root=Path(root);storage=storage or json.loads((root/'RAW_STORAGE.json').read_text());item=storage['files'][name];path=safe(root,item['storage_path'])
 if item['encoding']=='identity':
  with path.open('rb')as stream:yield stream
 elif item['encoding']=='gzip':
  with gzip.open(path,'rb')as stream:yield stream
 else:raise ValueError('Unknown raw encoding')

def hash_stream(stream):
 digest=hashlib.sha256();size=0
 for block in iter(lambda:stream.read(BLOCK),b''):digest.update(block);size+=len(block)
 return digest.hexdigest(),size

def verify_raw(root):
 root=Path(root);storage=json.loads((root/'RAW_STORAGE.json').read_text());original=json.loads((root/'terminal/RAW_SHA256.json').read_text());encoding=json.loads((root/'terminal/RAW_ENCODING.json').read_text());audit=json.loads((root/'terminal/audit.json').read_text())
 assert set(original)==set(storage['files'])==set(encoding)and original==audit['raw_inventory'],'Changed raw inventory'
 assert len(original)==storage['original_raw_files']
 total=stored=0
 for name,item in storage['files'].items():
  assert {k:item[k]for k in ('sha256','size_bytes')}==original[name]
  assert item['storage_path']=='terminal/'+encoding[name]['storage_path']and item['encoding']==encoding[name]['encoding']
  path=safe(root,item['storage_path'])
  with path.open('rb')as f:digest,size=hash_stream(f)
  assert digest==item['stored_sha256']and size==item['stored_size_bytes'],'Changed stored bytes:'+name
  stored+=size
  with open_raw(root,name,storage)as f:digest,size=hash_stream(f)
  assert digest==item['sha256']and size==item['size_bytes'],'Changed logical raw bytes:'+name
  total+=size
 assert total==storage['original_raw_bytes']and stored==storage['stored_raw_bytes']
 return {'logical_raw_files':len(original),'original_raw_bytes_verified':total,'stored_raw_bytes_verified':stored,'temporary_decompressed_files':0,'bounded_read_bytes':BLOCK}

if __name__=='__main__':
 import argparse,shutil,sys
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('raw_path');args=parser.parse_args()
 with open_raw(Path(__file__).resolve().parent,args.raw_path)as stream:shutil.copyfileobj(stream,sys.stdout.buffer,BLOCK)
