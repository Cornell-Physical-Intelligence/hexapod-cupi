"""Narrow source-bound reward callback; every inherited non-scope term retained."""
import ast
from contextlib import contextmanager
import hashlib
from pathlib import Path
from types import MethodType
import torch
from moving_reward import phase, scoped_terms

OMNI_SHA='f6ef7b74143c6c746b13e1f2c2bc8718c8350ff393f7a7fe9528ebdca3168eee'


def build_reward_callable(module):
    path=Path(module.__file__)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=OMNI_SHA:
        raise ValueError('Moving reward adapter requires exact source009 OmniFlatEnv')
    tree=ast.parse(path.read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='OmniFlatEnv')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_get_rewards')
    matches=[i for i,x in enumerate(fn.body) if isinstance(x,ast.Assign)
             and len(x.targets)==1 and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='reward']
    if len(matches)!=1:raise ValueError('Expected one exact original weighted reward assignment')
    expected='reward = sum((terms[k] * weight for k, weight in weights.items())) * self.step_dt'
    if ast.unparse(fn.body[matches[0]])!=expected:
        raise ValueError('Original reward formula changed')
    fn.body[matches[0]]=ast.parse("reward = self._moving_score(terms, v, w, air, first_contact, d)").body[0]
    code=compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(path)+'[moving_reward_callback]','exec')
    namespace=dict(module.__dict__);exec(code,namespace)
    return namespace['_get_rewards']


@contextmanager
def install_reward(env, session):
    import omni_flat_env
    call=build_reward_callable(omni_flat_env)
    original=env._get_rewards
    def score(instance, terms, v, w, air, first_contact, d):
        good=session.scorable
        result={k:value.clone() for k,value in terms.items()}
        if good.any():
            state={k:value[good] for k,value in session.reference['state'].items()}
            m=session.last['measurement']
            scope=phase(session.commands[good],state,m['distal_contact'][good],m['measurement_valid'][good])
            unmasked={
                'stand_joint_velocity':d.joint_vel.torch.square().mean(-1)[good].double(),
                'stand_target_velocity':((instance._processed_actions-instance.omni_previous_target)/.02).square().mean(-1)[good].double(),
                'stand_posture':(d.joint_pos.torch-d.default_joint_pos.torch).square().mean(-1)[good].double(),
                'stand_raw_action':instance.omni_raw_policy_action.square().mean(-1)[good].double(),
                'airtime':((air-.20).clamp(-.20,.25)*first_contact).sum(-1)[good].double()}
            changed=scoped_terms(v[good,:2].double(),w[good,2].double(),scope,unmasked)
            for key,value in changed.items():result[key][good]=value.to(result[key].dtype)
            session.reward_scope={k:value.detach().clone() for k,value in scope.items() if isinstance(value,torch.Tensor)}
        result={k:torch.where(good,value,0.) for k,value in result.items()}
        # The original function continues into capture_step and omni_sums.
        # Mutate its local dictionary so those paths describe the very same
        # effective ordinary reward, including zero recovery contributions.
        terms.clear();terms.update(result)
        session.reward_components={k:value.detach().clone() for k,value in terms.items()}
        total=sum(terms[k]*weight for k,weight in instance.cfg.omni_reward_weights.items())*.02
        # Failure events are applied once by the row-outcome layer; recovery
        # and failed measurement rows have no ordinary tracking reward.
        return torch.where(good,total,0.)
    def wrapped(instance):
        d=instance._robot.data
        if not torch.isfinite(d.applied_torque.torch).all() or not torch.isfinite(d.computed_torque.torch).all():
            raise RuntimeError('Nonfinite physical torque')
        reward=call(instance)
        if not torch.isfinite(reward).all():raise RuntimeError('Nonfinite moving reward')
        # Preserve the source009 outer reference-env telemetry, including its
        # direct raw-XYZW conversion. Effective reward changes cannot revive
        # the historical mislabeled quaternion diagnostic.
        instance.reference_residual_sample={k:v.detach().cpu().numpy().copy()
            for k,v in instance.reference_residual_target.items()}
        if getattr(instance,'omni_diagnostic_enabled',False):
            sample=instance.omni_diagnostic_sample
            sample.pop('filtered_target_rad',None);sample.pop('unfiltered_target_rad',None)
            raw=d.root_quat_w.torch.detach().cpu().numpy().copy()
            sample['quaternion_world_xyzw']=raw
            sample['quaternion_world_wxyz']=raw[..., [3,0,1,2]].copy()
            sample.update(instance.reference_residual_sample)
        return reward
    env._moving_score=MethodType(score,env);env._get_rewards=MethodType(wrapped,env)
    try:yield
    finally:
        env._get_rewards=original
        del env._moving_score
