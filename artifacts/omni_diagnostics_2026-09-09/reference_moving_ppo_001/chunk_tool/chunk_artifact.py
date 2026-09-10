"""Lossless bounded chunks for immutable artifact publication. Standard library only."""
from pathlib import Path
import argparse,hashlib,json,math,os,re,uuid
MAX_CHUNK_BYTES=64*1024*1024
BLOCK=1024*1024

def stream_hash(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(BLOCK),b''):h.update(block)
 return h.hexdigest()
def validate_manifest(root):
 root=Path(root)
 if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):raise ValueError('Symlink chunk substitution')
 meta=json.loads((root/'manifest.json').read_text())
 if set(meta)!={'schema','logical_path','original_bytes','original_sha256','chunk_size_bytes','chunks'} or meta['schema']!='lossless_binary_chunks_v1':raise ValueError('Unknown descriptor')
 n,c=meta['original_bytes'],meta['chunk_size_bytes']
 if type(n) is not int or n<0 or type(c) is not int or not 1<=c<=MAX_CHUNK_BYTES:raise ValueError('Invalid sizes')
 logical=meta['logical_path']
 if not isinstance(logical,str) or not logical or Path(logical).is_absolute() or '..' in Path(logical).parts:raise ValueError('Logical name must be relative metadata')
 if not isinstance(meta['original_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',meta['original_sha256']):raise ValueError('Invalid whole-file hash')
 rows=meta['chunks']
 if not isinstance(rows,list) or len(rows)!=(n+c-1)//c:raise ValueError('Wrong chunk count')
 inventory={'manifest.json'}
 for index,row in enumerate(rows):
  if set(row)!={'path','bytes','sha256'} or row['path']!=f'part-{index:06d}.bin':raise ValueError('Chunk order/path differs')
  expected=min(c,n-index*c)
  if type(row['bytes']) is not int or row['bytes']!=expected or not re.fullmatch('[0-9a-f]{64}',str(row['sha256'])):raise ValueError('Invalid chunk entry')
  p=root/row['path']
  if not p.is_file() or p.stat().st_size!=expected:raise ValueError('Missing/truncated chunk')
  inventory.add(row['path'])
 if {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}!=inventory:raise ValueError('Unexpected or missing payload')
 return meta

def consume(root,output=None):
 root=Path(root);meta=validate_manifest(root);whole=hashlib.sha256();total=0
 for row in meta['chunks']:
  local=hashlib.sha256();size=0
  with (root/row['path']).open('rb') as source:
   for block in iter(lambda:source.read(BLOCK),b''):
    local.update(block);whole.update(block);size+=len(block);total+=len(block)
    if output is not None:output.write(block)
  if size!=row['bytes'] or local.hexdigest()!=row['sha256']:raise ValueError('Changed chunk '+row['path'])
 if total!=meta['original_bytes'] or whole.hexdigest()!=meta['original_sha256']:raise ValueError('Whole-file hash or length differs')
 return meta

def split(source,root,logical_path,chunk_size=MAX_CHUNK_BYTES):
 source,root=Path(source),Path(root)
 if type(chunk_size) is not int or not 1<=chunk_size<=MAX_CHUNK_BYTES:raise ValueError('Chunk size must be1..64MiB')
 if source.is_symlink() or not source.is_file():raise ValueError('Regular original file required')
 if not isinstance(logical_path,str) or not logical_path or Path(logical_path).is_absolute() or '..' in Path(logical_path).parts:raise ValueError('Relative logical path required')
 if root.exists():raise FileExistsError(root)
 before=source.stat();root.mkdir(parents=True,exist_ok=False);rows=[];whole=hashlib.sha256();total=0
 with source.open('rb') as f:
  while total<before.st_size:
   name=f'part-{len(rows):06d}.bin';local=hashlib.sha256();size=0
   with (root/name).open('xb') as out:
    while size<chunk_size:
     block=f.read(min(BLOCK,chunk_size-size))
     if not block:break
     out.write(block);local.update(block);whole.update(block);size+=len(block);total+=len(block)
   if size==0:raise ValueError('Original truncated during split')
   rows.append({'path':name,'bytes':size,'sha256':local.hexdigest()})
  if f.read(1):raise ValueError('Original grew during split')
 after=source.stat()
 if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('Original changed during split')
 meta={'schema':'lossless_binary_chunks_v1','logical_path':logical_path,'original_bytes':total,'original_sha256':whole.hexdigest(),'chunk_size_bytes':chunk_size,'chunks':rows}
 (root/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');consume(root);return meta

def reconstruct(root,output):
 root,output=Path(root),Path(output)
 if output.exists():raise FileExistsError(output)
 if root.resolve()==output.resolve() or root.resolve() in output.resolve().parents:raise ValueError('Output must be outside immutable chunk tree')
 meta=validate_manifest(root);temp=output.with_name(output.name+'.partial-'+uuid.uuid4().hex)
 try:
  with temp.open('xb') as target:consume(root,target);target.flush();os.fsync(target.fileno())
  # Atomic installation without replacement, even if another writer raced us.
  os.link(temp,output)
 finally:
  if temp.exists():temp.unlink()
 return meta

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
 a=sub.add_parser('split');a.add_argument('--input',type=Path,required=True);a.add_argument('--chunks',type=Path,required=True);a.add_argument('--logical-path',required=True);a.add_argument('--chunk-size',type=int,default=MAX_CHUNK_BYTES)
 a=sub.add_parser('verify');a.add_argument('--chunks',type=Path,required=True)
 a=sub.add_parser('reconstruct');a.add_argument('--chunks',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
 args=p.parse_args()
 if args.mode=='split':r=split(args.input,args.chunks,args.logical_path,args.chunk_size)
 elif args.mode=='verify':r=consume(args.chunks)
 else:r=reconstruct(args.chunks,args.output)
 print(json.dumps({'passed':True,'mode':args.mode,'original_bytes':r['original_bytes'],'original_sha256':r['original_sha256'],'chunks':len(r['chunks']),'maximum_chunk_bytes':MAX_CHUNK_BYTES,'compression_or_reencoding':False},indent=2))
if __name__=='__main__':main()
