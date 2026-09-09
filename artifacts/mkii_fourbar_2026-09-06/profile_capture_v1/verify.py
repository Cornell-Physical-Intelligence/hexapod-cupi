#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
m=json.loads((root/'retrieval_metadata.json').read_text())
for name,row in m['captured']['files'].items():
 data=(root/name).read_bytes()
 assert len(data)==row['size_bytes'],name
 assert hashlib.sha256(data).hexdigest()==row['sha256']==row['local_sha256'],name
 assert row['match'] is True,name
 print(name+': exact remote/local bytes verified')
for name,row in m['captured']['remote_only_traces'].items():
 assert row['copied'] is False and not (root/name).exists(),name
 assert len(row['sha256'])==64 and row['size_bytes']>0,name
 print(name+': remote hash/reference only; trace not present or locally verified')
print('PASS')
