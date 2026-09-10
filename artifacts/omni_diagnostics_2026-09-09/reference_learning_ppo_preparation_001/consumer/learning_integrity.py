"""Read-only compact actual reset/history/bootstrap evidence for consumer003."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch

def array(x):return x.detach().cpu().numpy().copy()
def check(value,message):
    if not bool(value):raise RuntimeError('Learning integrity: '+message)

class LearningIntegrity:
    def __init__(self,session):
        self.session=session;self.records=[];self.reset_origins=np.full(session.num_envs,-1,np.int64)
        self.error=None

    def _append(self,kind,values):
        record={'kind':np.asarray(kind),'control':np.asarray(self.session.control,dtype=np.int64),**values}
        self.records.append(record)
        return record

    def history(self):
        e=self.session.encoder
        return {k:array(getattr(e,k)) for k in ('history','history_valid','last_fd','last_fd_valid','ready','episodes','steps')}

    def before_reset(self,ids,final):
        s=self.session;selected=np.zeros(s.num_envs,bool);selected[array(ids)]=True
        result=self._append('reset',dict(selected=selected,episode_before=array(s.episodes),
            terminated=s.rows[-1]['training_terminated'].copy(),truncated=s.rows[-1]['training_truncated'].copy(),
            final_policy=array(final['policy'][ids]),final_critic=array(final['critic'][ids]),
            final_learning_valid=array(final['learning_valid'][ids]),final_episode=array(final['episode_id'][ids]),
            **{'before__'+k:v for k,v in self.history().items()}))
        self.reset_origins[selected]=len(self.records)-1
        return result

    def after_reset(self,record):
        s=self.session;selected=record['selected'];after=self.history();packet=s.get_observations()
        record.update(episode_after=array(s.episodes),next_policy=array(packet['policy'])[selected],
            next_critic=array(packet['critic'])[selected],next_learning_valid=array(packet['learning_valid'])[selected],
            next_episode=array(packet['episode_id'])[selected],**{'after__'+k:v for k,v in after.items()})
        for key,value in after.items():check(np.array_equal(value[~selected],record['before__'+key][~selected]),'unselected encoder changed during physical reset: '+key)
        expected=record['episode_before']+selected.astype(np.int64)
        check(np.array_equal(record['episode_after'],expected),'physical reset episode identity differs')
        check(not after['ready'][selected].any(),'reset actor history still ready')
        check(not record['next_policy'].any() and not record['next_critic'].any() and not record['next_learning_valid'].any(),'reset actor placeholder was not excluded')
        record['verified']=np.asarray(True)

    def before_history_reset(self,selected):
        return self._append('history_initialization',dict(selected=array(selected),episode=array(self.session.episodes),
            origin_record=self.reset_origins[array(selected)].copy(),**{'before__'+k:v for k,v in self.history().items()}))

    def after_history_reset(self,record):
        selected=record['selected'];after=self.history()
        record.update(**{'reset__'+k:v for k,v in after.items()})
        for key,value in after.items():check(np.array_equal(value[~selected],record['before__'+key][~selected]),'unselected history changed during selected history reset: '+key)
        check(not after['history'][selected].any() and not after['history_valid'][selected].any(),'old selected history leaked across recovery')
        check(not after['last_fd_valid'][selected].any(),'reset interval-rate validity was retained')

    def recovered(self,record,packet):
        s=self.session;selected=record['selected'];after=self.history()
        record.update(policy=array(packet['policy'])[selected],critic=array(packet['critic'])[selected],
            learning_valid=array(packet['learning_valid'])[selected],packet_episode=array(packet['episode_id'])[selected],
            **{'encoded__'+k:v for k,v in after.items()})
        check(record['learning_valid'].all(),'recovered actor packet is disabled')
        check((after['history_valid'][selected].sum(-1)==1).all(),'recovered history has stale frames')
        check(not after['last_fd_valid'][selected].any(),'first recovered interval-angle channel is valid')
        check((after['steps'][selected]==0).all(),'recovered packet step is not zero')
        check(np.array_equal(record['packet_episode'][:,0],record['episode'][selected]),'recovered packet episode differs')
        controls=[]
        for row,origin in zip(np.flatnonzero(selected),record['origin_record']):
            if origin>=0:
                previous=self.records[int(origin)]
                check(s.control-int(previous['control'])==200,'recovery is not exactly 200 real controls')
                check(bool(previous.get('verified',False)),'recovery lacks verified physical reset')
                if previous['terminated'][row]:controls.append(int(previous['control']))
        record['finite_reset_origin_controls']=np.asarray(controls,dtype=np.int64)
        record['verified']=np.asarray(True)

    def bootstrap(self,data,next_value,next_obs,*,source,normalization_rows):
        ended=data['terminated']|data['truncated']
        if not ended.any():return
        selected=array(ended);ids=torch.nonzero(ended).flatten();final=data['final_observation']
        record=self._append('bootstrap',dict(selected=selected,episode=array(data['episode_id']),
            terminated=array(data['terminated']),truncated=array(data['truncated']),learnable=array(data['learnable']),
            final_valid=array(data['final_observation_valid']),final_episode=array(data['final_episode_id']),
            final_policy=array(final['policy'][ids]),final_critic=array(final['critic'][ids]),
            next_policy=array(next_obs['policy'][ids]),next_critic=array(next_obs['critic'][ids]),
            next_learning_valid=array(next_obs['learning_valid'][ids]),next_episode=array(next_obs['episode_id'][ids]),
            used_next_value=array(next_value),source=np.asarray(source),normalization_rows=np.asarray(normalization_rows,dtype=np.int64)))
        check(np.isfinite(record['used_next_value']).all(),'nonfinite actual bootstrap value')
        check(not record['used_next_value'][record['terminated']].any(),'true terminal had nonzero bootstrap')
        timed=record['truncated'];check(record['final_valid'][timed].all(),'timeout final packet invalid')
        check(np.array_equal(record['final_episode'][timed],record['episode'][timed]),'timeout bootstrapped from a new episode')
        matches=[r for r in self.records if str(r['kind'])=='reset' and int(r['control'])==int(record['control'])]
        check(len(matches)==1,'bootstrap lacks unique raw physical reset event')
        reset=matches[0]
        for key in ('selected','final_policy','final_critic','next_policy','next_critic','next_learning_valid','next_episode'):
            check(np.array_equal(record[key],reset[key]),'learner reset/final packet differs from session evidence: '+key)
        record['verified']=np.asarray(True)

    def summary(self):
        s=self.session;resets=[r for r in self.records if str(r['kind'])=='reset']
        recovered=[r for r in self.records if str(r['kind'])=='history_initialization' and (r['origin_record']>=0).any()]
        bootstraps=[r for r in self.records if str(r['kind'])=='bootstrap']
        complete=sum(len(r.get('finite_reset_origin_controls',[])) for r in recovered if bool(r.get('verified',False)))
        return {'schema':'moving_learning_integrity_ledger_v1','controls':s.control,'replicas':s.num_envs,
            'record_count':len(self.records),'reset_events':len(resets),
            'finite_reset_events':sum(bool(r['terminated'].any()) for r in resets),'completed_finite_recovery_rows':complete,
            'bootstrap_event_records':len(bootstraps),'all_records_verified':all(bool(r.get('verified',False)) for r in self.records),
            'session_failure':s.failure,'error':self.error,'per_environment_episodes':array(s.episodes).tolist(),
            'physical_admission':False,'original_evaluation_gates_changed':False,'automatic_training_allowed':False}

    def export(self,output):
        output=Path(output);output.mkdir(exist_ok=True)
        files={}
        for i,record in enumerate(self.records):
            path=output/('%05d_%s.npz'%(i,str(record['kind'])))
            if path.exists():raise FileExistsError('Immutable learning evidence already exists')
            np.savez_compressed(path,**record);files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        summary={**self.summary(),'files_sha256':files}
        (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        return summary
