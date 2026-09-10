"""Portable exact-inventory verification; no simulator, source tree or network."""
import hashlib
import json
from pathlib import Path
import numpy as np


def verify(root=None):
    root=Path(root or Path(__file__).resolve().parent)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    freeze=json.loads((root/'FREEZE_SHA256.json').read_text())
    actual={p.relative_to(root).as_posix()for p in root.rglob('*')if p.is_file()and '__pycache__'not in p.parts and p!=root/'FREEZE_SHA256.json'}
    if actual!=set(freeze):raise ValueError('Payload inventory changed')
    for name,digest in freeze.items():
        if sha(root/name)!=digest:raise ValueError('Payload changed: '+name)
    fixture=json.loads((root/'fixture/fixture.json').read_text())
    if fixture['mesh_npz_sha256']!=sha(root/'fixture/mesh.npz'):raise ValueError('Mesh identity differs')
    with np.load(root/'fixture/mesh.npz',allow_pickle=False)as z:
        if hashlib.sha256(z['vertices'].tobytes()).hexdigest()!=fixture['mesh_manifest']['vertex_array_sha256']:raise ValueError('Source vertices differ')
        if hashlib.sha256(z['faces'].tobytes()).hexdigest()!=fixture['mesh_manifest']['triangle_index_sha256']:raise ValueError('Source triangles differ')
    return {'files':len(freeze),'bytes':sum((root/name).stat().st_size for name in freeze),
            'freeze_sha256':sha(root/'FREEZE_SHA256.json'),'verified':True}


if __name__=='__main__':print(json.dumps(verify(),indent=2))
