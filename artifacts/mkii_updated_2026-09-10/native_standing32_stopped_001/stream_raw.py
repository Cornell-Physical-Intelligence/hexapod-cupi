"""Lossless bounded-memory access to logical raw files; no temporary expansion."""
from pathlib import Path
from contextlib import contextmanager, ExitStack
import gzip, hashlib, io, json

class PartsReader(io.RawIOBase):
    def __init__(self, paths):
        super().__init__(); self.paths=iter(paths); self.current=None
    def readable(self):return True
    def readinto(self, buffer):
        while True:
            if self.current is None:
                try:self.current=next(self.paths).open('rb')
                except StopIteration:return 0
            count=self.current.readinto(buffer)
            if count:return count
            self.current.close();self.current=None
    def close(self):
        if self.current is not None:self.current.close();self.current=None
        super().close()

def safe(root,name):
    p=root/name
    if p.is_symlink()or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Unsafe stored path')
    return p

@contextmanager
def open_raw(root,name,storage=None):
    root=Path(root);storage=storage or json.loads((root/'RAW_STORAGE.json').read_text());item=storage['files'][name]
    with ExitStack()as stack:
        if item['encoding']=='identity':stream=stack.enter_context(safe(root,item['storage_path']).open('rb'))
        elif item['encoding']=='gzip_concatenated_parts':
            source=stack.enter_context(io.BufferedReader(PartsReader([safe(root,p['path'])for p in item['parts']]),buffer_size=1<<20))
            stream=stack.enter_context(gzip.GzipFile(fileobj=source,mode='rb'))
        else:raise ValueError('Unknown public raw encoding')
        yield stream

def hash_stream(stream):
    h=hashlib.sha256();size=0
    for block in iter(lambda:stream.read(1<<20),b''):h.update(block);size+=len(block)
    return h.hexdigest(),size

def verify_raw(root):
    root=Path(root);storage=json.loads((root/'RAW_STORAGE.json').read_text());original=json.loads((root/'terminal/RAW_SHA256.json').read_text())
    assert set(original)==set(storage['files'])and len(original)==27
    assert sum(v['size_bytes']for v in original.values())==storage['original_raw_bytes']==1836888673
    total=0
    for name,item in storage['files'].items():
        assert {k:item[k]for k in ('sha256','size_bytes')}==original[name]
        if item['encoding']=='gzip_concatenated_parts':
            assert 0<item['max_part_bytes']<=48<<20
            compressed=hashlib.sha256();compressed_size=0
            for i,part in enumerate(item['parts']):
                path=safe(root,part['path']);assert path.name==f'part_{i:03d}'
                assert 0<path.stat().st_size==part['size_bytes']<=item['max_part_bytes']
                part_hash=hashlib.sha256()
                with path.open('rb')as stream:
                    for block in iter(lambda:stream.read(1<<20),b''):compressed.update(block);part_hash.update(block);compressed_size+=len(block)
                assert part_hash.hexdigest()==part['sha256']
            assert compressed.hexdigest()==item['compressed_sha256']and compressed_size==item['compressed_size_bytes']
        with open_raw(root,name,storage)as stream:digest,size=hash_stream(stream)
        assert digest==item['sha256']and size==item['size_bytes'],name
        total+=size
    return {'logical_raw_files':len(original),'original_raw_bytes_verified':total,'temporary_decompressed_files':0,'bounded_read_bytes':1<<20}

if __name__=='__main__':
    import argparse, shutil, sys
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('raw_path');args=parser.parse_args()
    with open_raw(Path(__file__).resolve().parent,args.raw_path)as stream:shutil.copyfileobj(stream,sys.stdout.buffer,1<<20)
