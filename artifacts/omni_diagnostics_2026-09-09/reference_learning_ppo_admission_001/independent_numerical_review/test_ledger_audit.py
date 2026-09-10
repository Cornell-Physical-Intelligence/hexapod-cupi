import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from audit_learning import replay_ledger,sha,progress

def fixture():
    n,t=32,450;mask=np.zeros(n,bool);mask[6]=True
    end=np.zeros((t,n),bool);end[211,6]=True
    active=np.ones((t,n),bool);active[:200]=False;active[212:412,6]=False
    episode=np.zeros((t,n),np.int64);episode[212:,6]=1
    trace={'learning_active':active,'training_terminated':end,'training_truncated':np.zeros_like(end),
        'episode_id':episode,'next_reference_failure':end.copy()}
    before={'history':np.ones((n,5,63)),'history_valid':np.ones((n,5),bool),
        'last_fd':np.ones((n,18)),'last_fd_valid':np.ones(n,bool),'ready':np.ones(n,bool),
        'episodes':np.zeros(n,np.int64),'steps':np.full(n,12,np.int64)}
    after=copy.deepcopy(before);after['ready'][6]=False
    reset={'kind':np.asarray('reset'),'control':np.asarray(212),'selected':mask,'verified':np.asarray(True),
        'terminated':mask.copy(),'truncated':np.zeros(n,bool),'episode_before':np.zeros(n,np.int64),
        'episode_after':mask.astype(np.int64),'final_episode':np.zeros((1,1),np.int64),
        'next_episode':np.ones((1,1),np.int64),'final_policy':np.ones((1,846)),
        'final_critic':np.ones((1,849)),'next_policy':np.zeros((1,846)),
        'next_critic':np.zeros((1,849)),'next_learning_valid':np.zeros((1,1),bool),
        **{'before__'+k:v for k,v in before.items()},**{'after__'+k:v for k,v in after.items()}}
    boot={k:copy.deepcopy(reset[k]) for k in ('selected','final_policy','final_critic','next_policy','next_critic','next_learning_valid','next_episode','terminated','truncated')}
    boot.update(kind=np.asarray('bootstrap'),control=np.asarray(212),verified=np.asarray(True),
        used_next_value=np.zeros(n),normalization_rows=np.asarray(0),source=np.asarray('unoptimized_recovery_probe_no_PPO_storage'))
    cleared=copy.deepcopy(after)
    for key in ('history','history_valid','last_fd_valid'):cleared[key][6]=0
    encoded=copy.deepcopy(cleared);encoded['history_valid'][6,-1]=True;encoded['steps'][6]=0
    packet=np.ones((1,846));packet[:,:315]=encoded['history'][6].reshape(1,315)
    packet[:,315:320]=encoded['history_valid'][6]
    recovered={'kind':np.asarray('history_initialization'),'control':np.asarray(412),
        'verified':np.asarray(True),'selected':mask.copy(),'origin_record':np.asarray([0]),
        'episode':mask.astype(np.int64),'packet_episode':np.ones((1,1),np.int64),
        'learning_valid':np.ones((1,1),bool),'policy':packet,'critic':np.concatenate((packet,np.ones((1,3))),axis=1),
        **{'before__'+k:v for k,v in after.items()},**{'reset__'+k:v for k,v in cleared.items()},
        **{'encoded__'+k:v for k,v in encoded.items()}}
    return trace,[reset,boot,recovered]

def write(path,trace,records):
    files={}
    for i,record in enumerate(records):
        file=path/('%05d_%s.npz'%(i,str(record['kind'])));np.savez_compressed(file,**record);files[file.name]=sha(file)
    (path/'summary.json').write_text(json.dumps({'controls':450,'replicas':32,'files_sha256':files,
        'completed_finite_recovery_rows':1,'per_environment_episodes':trace['training_terminated'].sum(0).tolist()}))

class Tests(unittest.TestCase):
    def test_actual_format_masks_packets_and_completed_recovery(self):
        trace,records=fixture()
        with tempfile.TemporaryDirectory() as d:
            write(Path(d),trace,records);r=replay_ledger(d,trace)
            self.assertEqual(r['completed_finite_recoveries_per_row'][6],1)
            self.assertTrue(r['terminal_bootstrap_zero_replayed'])

    def test_independent_replay_rejects_self_consistently_rehashed_corruption(self):
        for kind in ('unselected_history','stale_history','wrong_critic','nonzero_terminal','early_recovery','normalization'):
            trace,records=fixture()
            if kind=='unselected_history':records[0]['after__history'][0,0,0]=9
            if kind=='stale_history':records[2]['encoded__history_valid'][6]=True
            if kind=='wrong_critic':records[1]['final_critic'][0,0]=9
            if kind=='nonzero_terminal':records[1]['used_next_value'][6]=.1
            if kind=='early_recovery':records[2]['control']=np.asarray(411)
            if kind=='normalization':records[1]['normalization_rows']=np.asarray(1)
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as d:
                write(Path(d),trace,records)
                with self.assertRaises(ValueError):replay_ledger(d,trace)

if __name__=='__main__':unittest.main()
