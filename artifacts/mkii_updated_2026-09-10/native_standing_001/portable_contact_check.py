"""Read the recorded little-endian NumPy arrays with the Python standard library."""
from pathlib import Path
import ast, math, struct, zipfile

def arrays(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            raw = archive.read(name)
            assert raw[:6] == b'\x93NUMPY'
            size = 2 if raw[6] == 1 else 4
            length = int.from_bytes(raw[8:8 + size], 'little')
            offset = 8 + size + length
            header = ast.literal_eval(raw[8 + size:offset].decode())
            assert header['fortran_order'] is False
            fmt = {'<f4': '<f', '<u4': '<I'}[header['descr']]
            values = [item[0] for item in struct.iter_unpack(fmt, raw[offset:])]
            assert len(values) == math.prod(header['shape'])
            result[name[:-4]] = (header['shape'], values)
    return result

def check(path):
    data = arrays(path)
    assert {k: v[0] for k, v in data.items()} == {
        'force': (1024, 1), 'point': (1024, 3), 'normal': (1024, 3),
        'separation': (1024, 1), 'counts': (19, 1), 'starts': (19, 1)}
    f, p, n, s, counts, starts = [data[k][1] for k in ('force', 'point', 'normal', 'separation', 'counts', 'starts')]
    used = []
    for count, start in zip(counts, starts):
        assert start + count <= 1024
        used.extend(range(start, start + count))
    assert len(used) == len(set(used)) == 153
    bad, forceful, normal_error = [], [], []
    for i in used:
        xyz, normal = p[3*i:3*i+3], n[3*i:3*i+3]
        assert all(math.isfinite(v) for v in [f[i], s[i], *xyz, *normal])
        error = abs(math.sqrt(sum(v*v for v in normal)) - 1)
        if error > 1e-3:
            assert f[i] == 0 and normal == [0, 0, 0] and s[i] == 0
            bad.append(i)
        if f[i] != 0:
            assert error <= 1e-3
            forceful.append(i)
            normal_error.append(error)
    assert bad == list(range(17, 144)) and len(forceful) == 7
    assert len({tuple(p[3*i:3*i+3]) for i in bad}) == 1
    return {'counted_slots': len(used), 'invalid_normal_slots': len(bad),
            'all_invalid_exact_zero_force_normal_separation': True,
            'nonzero_force_slots': len(forceful), 'nonzero_invalid_normals': 0,
            'nonzero_normal_max_abs_unit_error': max(normal_error)}

if __name__ == '__main__':
    import json
    print(json.dumps(check(Path(__file__).resolve().parent/'terminal/run/standing/failed_contact_buffer.npz'), sort_keys=True))
