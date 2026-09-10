#!/usr/bin/env python3
"""Bounded, hash-pinned extraction; never imports or executes supplied CAD files.

The source ZIP omitted its UTF-8 name flag. Decode CP437-observed filenames back
through UTF-8 and require exact agreement with the frozen extraction manifest.
Use --verify-only to validate all bytes without creating an expanded source copy.
"""
import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

EXPECTED_SHA256='31f33044285c3de2dbd85180e7c4886c31d80da4270dfd6b94bbd50d150a4cf7'
MAX_ENTRIES=128
MAX_TOTAL_BYTES=64*1024*1024
MAX_MEMBER_BYTES=8*1024*1024
MAX_COMPRESSION_RATIO=100

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def safe_name(name):
 if not isinstance(name,str) or '\x00' in name or '\\' in name or ':' in name:
  raise ValueError('Unsafe member name')
 p=PurePosixPath(name)
 if p.is_absolute() or any(x in ('','.','..') for x in name.rstrip('/').split('/')):
  raise ValueError('Unsafe member path')
 return p

def decoded_name(info):
 if info.flag_bits&0x800:return info.filename
 try:return info.filename.encode('cp437').decode('utf-8')
 except (UnicodeEncodeError,UnicodeDecodeError):return info.filename

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--archive',type=Path,required=True)
 p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('extraction_manifest.json'))
 p.add_argument('--out',type=Path)
 p.add_argument('--verify-only',action='store_true')
 p.add_argument('--receipt',type=Path)
 a=p.parse_args()
 if a.verify_only==bool(a.out):p.error('Choose exactly one of --verify-only or --out')
 if digest(a.archive)!=EXPECTED_SHA256:raise ValueError('Archive SHA256 mismatch')
 records=json.loads(a.manifest.read_text());by_zip={r['zip_name']:r for r in records}
 if len(records)!=121 or len(by_zip)!=len(records):raise ValueError('Expected121 unique file manifest entries')
 paths=set()
 for r in records:
  safe_name(r['zip_name']);safe_name(r['decoded_name'])
  if r['decoded_name'] in paths:raise ValueError('Duplicate decoded destination')
  paths.add(r['decoded_name'])
 if a.out:
  if a.out.is_symlink() or a.out.exists() and (not a.out.is_dir() or any(a.out.iterdir())):
   raise ValueError('Output must be a new or empty non-symlink directory')
  a.out.mkdir(parents=True,exist_ok=True);dest=a.out.resolve()
 else:dest=None
 total=0;verified=[]
 with zipfile.ZipFile(a.archive) as archive:
  entries=archive.infolist()
  if len(entries)>MAX_ENTRIES or sum(i.file_size for i in entries)>MAX_TOTAL_BYTES:
   raise ValueError('Archive exceeds entry/expanded-byte bound')
  if len({i.filename for i in entries})!=len(entries):raise ValueError('Duplicate ZIP member')
  for i in entries:
   safe_name(i.filename);name=decoded_name(i);safe_name(name)
   mode=i.external_attr>>16
   if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0,stat.S_IFREG,stat.S_IFDIR):raise ValueError('Special files prohibited')
   if i.flag_bits&1:raise ValueError('Encrypted ZIP members prohibited')
   if i.is_dir():
    if name!='assets/':raise ValueError('Unexpected directory')
    continue
   if i.file_size>MAX_MEMBER_BYTES or i.file_size/max(i.compress_size,1)>MAX_COMPRESSION_RATIO:
    raise ValueError('Member exceeds size/compression bound')
   r=by_zip.get(i.filename)
   if r is None or r['decoded_name']!=name:raise ValueError('Name decoding differs from frozen source manifest')
   with archive.open(i) as f:data=f.read(MAX_MEMBER_BYTES+1)
   if len(data)!=i.file_size or len(data)>MAX_MEMBER_BYTES:raise ValueError('Expanded size mismatch')
   h=hashlib.sha256(data).hexdigest()
   if h!=r['sha256']:raise ValueError('Member SHA256 mismatch')
   total+=len(data)
   if total>MAX_TOTAL_BYTES:raise ValueError('Expanded total exceeds bound')
   if dest:
    target=dest/Path(name)
    if not target.resolve().is_relative_to(dest):raise ValueError('Destination escapes root')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write(data)
   verified.append(name)
 if set(verified)!=paths:raise ValueError('Archive/manifest completeness mismatch')
 if dest:
  with (dest/'extraction_manifest.json').open('xb') as f:f.write(a.manifest.read_bytes())
 receipt={'source_zip_sha256':EXPECTED_SHA256,'files_verified':len(verified),'expanded_file_bytes':total,
  'verify_only':a.verify_only,'extracted_source_files_unchanged':True,
  'filename_decoding':'CP437-observed ZIP names decoded through UTF-8 and checked against frozen mapping',
  'pickle_loaded_or_executed':False,'bounds':{'entries':MAX_ENTRIES,'member_bytes':MAX_MEMBER_BYTES,'total_bytes':MAX_TOTAL_BYTES,'compression_ratio':MAX_COMPRESSION_RATIO}}
 if a.receipt:a.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
