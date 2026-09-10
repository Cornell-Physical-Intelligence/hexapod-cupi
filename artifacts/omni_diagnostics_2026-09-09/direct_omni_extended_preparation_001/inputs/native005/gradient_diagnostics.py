"""Read-only actor-gradient summaries on an already built graph; no model forward."""
import torch

def gradient_vector(loss,parameters):
    parameters=tuple(parameters)
    if not parameters:raise ValueError('Actor parameters required')
    grads=(torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
           if loss.requires_grad else (None,)*len(parameters))
    return torch.cat([(torch.zeros_like(p) if g is None else g.detach()).reshape(-1)
                      for p,g in zip(parameters,grads)])

def gradient_decomposition(ppo_actor,regularizer_terms,parameters):
    parameters=tuple(parameters)
    terms={'ppo_actor':ppo_actor}
    terms.update({k:regularizer_terms[k] for k in ('quiet_temporal','moving_temporal','spatial')})
    vectors={k:gradient_vector(v,parameters) for k,v in terms.items()}
    norms={k:torch.linalg.vector_norm(v) for k,v in vectors.items()}
    a,b=vectors['ppo_actor'],vectors['quiet_temporal']
    denom=norms['ppo_actor']*norms['quiet_temporal']
    cosine=None if float(denom)==0. else float(torch.dot(a,b)/denom)
    result={'norms':{k:float(v) for k,v in norms.items()},'ppo_quiet_cosine':cosine,
            'component_sum_norm':float(torch.linalg.vector_norm(sum(vectors.values())))}
    if any(not torch.isfinite(v).all() for v in vectors.values()):raise ValueError('Nonfinite diagnostic gradients')
    return result

def assigned_gradient_norm(parameters):
    values=[p.grad.detach().reshape(-1) for p in parameters if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(values))) if values else 0.
