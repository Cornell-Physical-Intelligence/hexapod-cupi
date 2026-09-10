"""Capture one device snapshot with original done flags, before automatic reset."""
from contextlib import contextmanager
from types import MethodType
import torch

def tensor(value):
    value=value.torch if hasattr(value,'torch') else value
    if not isinstance(value,torch.Tensor) or value.dtype!=torch.bool:raise ValueError('Original done flags must be device bool tensors')
    return value.detach().clone()

@contextmanager
def capture_before_reset(env,callback):
    original=env._get_dones;samples=[]
    def wrapped(instance):
        terminated,truncated=original()
        samples.append(callback(tensor(terminated),tensor(truncated)))
        return terminated,truncated
    env._get_dones=MethodType(wrapped,env)
    try:yield samples
    finally:env._get_dones=original

def require_one(samples,previous_count,terminated,truncated):
    if len(samples)!=previous_count+1:raise RuntimeError('Expected exactlyone pre-reset device snapshot')
    sample=samples[-1];m=sample['measurement']
    if not torch.equal(m['terminated'],tensor(terminated)) or not torch.equal(m['truncated'],tensor(truncated)):
        raise RuntimeError('Pre-reset/returned done flags disagree')
    return sample
