"""Small standard-library reader for fixed numeric NumPy evidence, never objects."""
import array
import ast
import math
import struct
import sys
import zipfile


def numeric(path,key,shape,kind='float'):
    """Require exact C-order shape, finite numeric values and a complete payload."""
    with zipfile.ZipFile(path) as archive:
        name=key+'.npy'
        if archive.namelist().count(name)!=1:raise ValueError('Missing/duplicate numeric evidence: '+key)
        member=archive.getinfo(name)
        if member.file_size>32*1024*1024:raise ValueError('Unexpected numeric evidence allocation')
        data=archive.read(name)
    if data[:6]!=b'\x93NUMPY':raise ValueError('Wrong NPY magic')
    version=data[6:8]
    if version==b'\x01\x00':offset=10;length=struct.unpack('<H',data[8:10])[0]
    elif version==b'\x02\x00':offset=12;length=struct.unpack('<I',data[8:12])[0]
    else:raise ValueError('Unsupported NPY version')
    if length>65536:raise ValueError('Unexpected numeric header')
    header=ast.literal_eval(data[offset:offset+length].decode('latin1'))
    if header.get('fortran_order') is not False or tuple(header.get('shape',()))!=tuple(shape):
        raise ValueError('Numeric evidence shape/order mismatch: '+key)
    allowed={'float':{'<f4':'f','<f8':'d'},'int':{'<i4':'i','<i8':'q'},'bool':{'|b1':'B'}}[kind]
    dtype=header.get('descr')
    if dtype not in allowed:raise ValueError('Numeric evidence dtype mismatch: '+key)
    value=array.array(allowed[dtype]);payload=data[offset+length:]
    if len(payload)!=math.prod(shape)*value.itemsize:raise ValueError('Incomplete numeric payload: '+key)
    value.frombytes(payload)
    if sys.byteorder!='little' and value.itemsize>1:value.byteswap()
    if any(not math.isfinite(v) for v in value):raise ValueError('Nonfinite numeric evidence: '+key)
    if kind=='bool' and any(v not in (0,1) for v in value):raise ValueError('Invalid boolean payload')
    return value
