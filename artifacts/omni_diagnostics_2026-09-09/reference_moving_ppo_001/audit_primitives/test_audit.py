import copy
import unittest
import numpy as np
from audit import moving, ticks

def fixture():
    t,n=460,32;end=np.zeros((t,n),bool);end[214,0]=True
    active=np.ones((t,n),bool);active[:200]=False;active[215:415,0]=False
    episode=np.zeros((t,n),np.int64);episode[215:,0]=1
    command=np.zeros((t,n,3));command[:,:,0]=.005*(active&(np.arange(n)%4!=0))
    scalar=np.zeros((t,n));q=np.zeros((t,n,18));xyz=np.zeros((t,n,3));quat=np.zeros((t,n,4));quat[:,:,3]=1
    torque=np.ones_like(q);support=np.ones((t,n,6),bool);false=np.zeros((t,n),bool)
    trace=dict(learning_active=active,training_terminated=end,training_truncated=false.copy(),
        next_reference_failure=false.copy(),reward_scorable=active&~end,episode_id=episode,
        requested_command=command,base_contact=false.copy(),shaft_contact=~support,coxa_contact=~support,
        femur_contact=~support,distal_contact=support,terminated=end.copy(),joint_position_rad=q,
        joint_velocity_rad_s=q.copy(),position_world_m=xyz,quaternion_world_xyzw=quat,
        velocity_world_mps=xyz.copy(),computed_torque_nm=torque,applied_torque_nm=torque.copy())
    trace['reward_before_event']=(active&~end).astype(np.float32)
    trace['reward']=trace['reward_before_event']-3*end.astype(np.float32)
    trace['reward_component__test']=trace['reward_before_event'].copy()
    raw={}
    mapping={'root_link_position_world_m':'position_world_m','root_link_quaternion_world_xyzw':'quaternion_world_xyzw',
        'root_link_velocity_world_mps':'velocity_world_mps','computed_torque_nm':'computed_torque_nm',
        'applied_torque_nm':'applied_torque_nm','joint_position_rad':'joint_position_rad','joint_velocity_rad_s':'joint_velocity_rad_s'}
    for key,old in mapping.items():raw[key]=np.concatenate([trace[old][:1],np.repeat(trace[old],8,axis=0)])
    size=t*8+1
    raw.update(relative_physics_index=np.arange(size),control_index=np.r_[-1,np.repeat(np.arange(t),8)],
        substep_index=np.r_[0,np.tile(np.arange(1,9),t)],sim_step_counter=100+np.arange(size),
        sdk_sim_timestamp_s=np.arange(size)*.0025,time_s=np.arange(size)*.0025,
        episode_id=np.concatenate([episode[:1],np.repeat(episode,8,axis=0)]))
    raw['crosses_episode_reset']=np.r_[np.zeros((1,n),bool),np.diff(raw['episode_id'],axis=0)!=0]
    raw['computed_torque_nm'][1+214*8+3,0,4]=1.61
    stamp=np.zeros((t,n,14),np.float32);current=np.zeros((n,14),np.float32)
    for i in range(t):
        current=ticks(current);stamp[i]=current
        if i==214:reset_stamp=current.copy();reset_stamp[0]=0;current[0]=0
    clocks=dict(sensor_timestamp_s=stamp,sensor_last_update_s=stamp.copy(),expected_timestamp_s=stamp.copy(),
        sensor_outdated=np.zeros_like(stamp,bool),all_sensors_valid=np.ones((t,n),bool),
        sensor_age_s=np.zeros_like(stamp),contact_valid=np.ones((t,n,6),bool))
    mask=np.zeros(n,bool);mask[0]=True
    resets=dict(reset_mask=mask[None],timestamp=reset_stamp[None],last_update=reset_stamp[None].copy(),
        outdated=np.broadcast_to(mask[None,:,None],(1,n,14)).copy())
    event=[dict(control=215,ended_rows=[0],episodes=[0]*n,reset_kind='original_inherited_reset_only_before_next_action',
        reset_reference_joint_target_rad=[[0.]*18])]
    updates=[dict(controls_per_replica=256,valid_learning_transitions=int(active[200:456].sum()),
        finite_terminal_transitions=1,time_limit_transitions=0,recovery_transitions_excluded=200)]
    return [trace,raw,clocks,{'current_leg':np.full((t,n),-1)},event,updates,resets]

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=fixture()

    def test_mixed_reset_counts_interior_spike_and_observation_scope(self):
        result=moving(*self.data)
        self.assertEqual(result['finite_terminal_transitions'],1)
        self.assertEqual(result['recovery_transitions_excluded'],200)
        self.assertEqual(result['requested_excess_samples'],1)
        self.assertFalse(result['actor_packet_history_directly_exported'])
        self.assertEqual(float(self.data[0]['computed_torque_nm'].max()),1.)

    def test_corrupt_substep_or_episode_or_learning_counts_rejected(self):
        for corruption in ('substep','episode','counts','double_penalty','learn_recovery','cap','missing_field'):
            d=copy.deepcopy(self.data)
            if corruption=='substep':d[1]['substep_index'][9]=8
            if corruption=='episode':d[1]['episode_id'][215*8+1,0]=0
            if corruption=='counts':d[5][0]['valid_learning_transitions']+=1
            if corruption=='double_penalty':d[0]['reward'][214,0]-=3
            if corruption=='learn_recovery':d[0]['learning_active'][216,0]=True
            if corruption=='cap':d[1]['applied_torque_nm'][201*8+2,2,3]=1.61
            if corruption=='missing_field':d[0]['field_present__reward']=np.r_[np.ones(459,bool),False]
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):moving(*d)

    def test_duplicate_all_clock_channels_and_unselected_reset_rejected(self):
        for corruption in ('duplicate','selected','unselected'):
            d=copy.deepcopy(self.data)
            if corruption=='duplicate':
                for k in ('sensor_timestamp_s','sensor_last_update_s','expected_timestamp_s'):d[2][k][300]=d[2][k][299]
            if corruption=='selected':d[6]['timestamp'][0,0]=.1
            if corruption=='unselected':d[6]['timestamp'][0,1]=0
            with self.subTest(corruption=corruption),self.assertRaises(ValueError):moving(*d)

if __name__=='__main__':unittest.main()
