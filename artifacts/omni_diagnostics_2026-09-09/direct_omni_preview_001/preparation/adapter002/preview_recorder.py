"""Native camera and deterministic direct315 actions; imports occur after AppLauncher."""
import math
from pathlib import Path
import numpy as np
from preview_contract import DT,FPS,STEPS,FRAMES,schedule,save,sha,verify_inputs,native_inference_origins
from render_helpers import initial_rgb_frame,draw_live_command

def validate_rgb(raw):
    if raw is None or raw.ndim!=3 or raw.shape[:2]!=(720,1280) or raw.shape[2] not in (3,4) or np.std(raw)<1:
        raise ValueError('Native RGB must be nonblank1280x720')
    return raw

def actual_pose(position,quaternion):
    p=np.asarray(position,float);q=np.asarray(quaternion,float)
    if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-4:raise ValueError('Finite position and unit raw SDK XYZW required')
    x,y,z,w=q
    rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                       [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                       [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    forward=rotation@np.array([0.,-1.,0.])
    if np.linalg.norm(forward[:2])<.1:raise ValueError('Heading undefined near vertical body-forward axis')
    return np.r_[p[:2],math.atan2(forward[1],forward[0])]

def check_packet(obs,command):
    import torch
    for key,width in [('policy',315),('critic',318)]:
        if tuple(obs[key].shape)!=(1,width) or not bool(torch.isfinite(obs[key]).all()):raise ValueError('Wrong/nonfinite native315/318 packet')
    expected=command*torch.tensor([5.,5.,2.5],device=command.device)
    if not torch.equal(obs['policy'][:,-63+6:-63+9],expected):raise ValueError('Actor packet does not contain the command already applied')

def checkpoint_readback(runner,checkpoint):
    import torch
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    for key,module in [('actor_state_dict',runner.alg.actor),('critic_state_dict',runner.alg.critic)]:
        actual=module.state_dict();expected=saved[key]
        if set(actual)!=set(expected):raise ValueError('Loaded model tensor inventory differs')
        for name,value in actual.items():
            if not torch.equal(value.detach().cpu(),expected[name].detach().cpu()):raise ValueError('Loaded checkpoint tensor differs: '+key+'/'+name)
    return {'passed':True,'actor_and_critic_including_normalizers_exact':True,'checkpoint_sha256':sha(checkpoint)}

def command_reference(initial,rows,device):
    import torch
    from omni_flat_math import integrate_body_twist,slew_commands
    pose=torch.tensor(initial,device=device,dtype=torch.float32).reshape(1,3);command=torch.zeros_like(pose)
    poses=[];commands=[]
    for row in rows:
        # Current observation/action uses the current command; parent slew applies after this control.
        pose=integrate_body_twist(pose,command,DT);poses.append(pose[0].clone());commands.append(command[0].clone())
        command=slew_commands(command,torch.tensor(row['requested_command'],device=device).reshape(1,3),DT)
    return torch.stack(poses).cpu().numpy(),torch.stack(commands).cpu().numpy()

def capture_sample(env,terminated,truncated):
    sample={k:v.copy() for k,v in env.omni_diagnostic_sample.items()}
    sample['returned_terminated']=terminated.detach().cpu().numpy().copy()
    sample['returned_truncated']=truncated.detach().cpu().numpy().copy()
    # Preserve the misleading historical raw key; provide explicit names without rewriting it.
    raw=sample['quaternion_world_wxyz'].copy()
    sample['quaternion_world_xyzw']=raw;sample['quaternion_world_wxyz_converted']=raw[..., [3,0,1,2]]
    return sample,bool(terminated.any() or truncated.any())

def annotate(raw,row,observed,measured,requested_peak,saturation,provenance,*,event=None):
    from PIL import Image,ImageDraw
    frame=Image.fromarray(np.asarray(raw[:,:,:3],dtype=np.uint8));draw=ImageDraw.Draw(frame)
    draw.rectangle((8,8,frame.width-8,125),fill=(14,20,29));pilot=provenance['pilot']
    lines=[f"DIRECT PPO PROGRESS - STAGE 2 INCOMPLETE | {pilot['branch']} {pilot['completed_updates']} updates | {row['segment']} | {row['time_start_s']+DT:.2f}s",
           'REQUEST FWD/LEFT/YAW: '+ ' / '.join(f'{x:+.3f}' for x in row['requested_command']),
           'ACTOR COMMAND: '+' / '.join(f'{x:+.3f}' for x in observed)+' | MEASURED: '+' / '.join(f'{x:+.3f}' for x in measured),
           f'REQUESTED TORQUE MAX {requested_peak:.3f} N m | >1.6 N m {100*saturation:.1f}% | full robot / command only / 1x playback',
           'Checkpoint '+provenance['checkpoint_sha256'][:16]+' | short stop clips are not sustained quiet qualification']
    for i,text in enumerate(lines):draw.text((18,16+21*i),text,fill='white')
    if event:
        draw.rectangle((8,136,frame.width-8,178),fill=(150,15,20));draw.text((18,150),event+' | last valid render, no reset continuation',fill='white')
    return np.asarray(frame)

def record(env,runner,plan,output,checkpoint_sha,args,provenance):
    import torch,imageio.v2 as imageio
    from tensordict import TensorDict
    from omni_path_demo import GroundDrawing
    from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
    if checkpoint_sha!=args.checkpoint_sha256:raise ValueError('Native dispatch checkpoint differs')
    if env.num_envs!=1 or abs(env.step_dt-DT)>1e-9 or env._robot.num_joints!=18 or env._robot.num_bodies!=19 or sum(len(s.body_names) for s in env._feet_contact_sensors)!=6:raise ValueError('Expected one full19-body18-joint6-foot native robot')
    rows=schedule();samples=[];actions=[];telemetry=[];trail=[];frames=0;terminal=None;error=None;writer=None;last_raw=None
    before=None;after=None;verified=False;output=Path(output);policy=runner.get_inference_policy(device=env.device)
    env.omni_diagnostic_enabled=True
    try:
        before=checkpoint_readback(runner,args.checkpoint)
        with torch.inference_mode():
            env.reset(seed=args.seed);env.episode_length_buf.zero_();env.set_evaluation_targets(torch.zeros((1,3),device=env.device))
            initial=actual_pose(env._robot.data.root_pos_w.torch[0].cpu().numpy(),env._robot.data.root_quat_w.torch[0].cpu().numpy())
            poses,commands=command_reference(initial,rows,env.device);drawing=GroundDrawing()
            save(output/'native_inference_origins.json',native_inference_origins(args.source_root))
            np.savez_compressed(output/'command_reference.npz',pose_world_xy_heading_rad=poses,
                                actor_command_expected=commands,time_end_s=np.arange(1,STEPS+1)*DT)
            drawing.reference(poses,commands,'DIRECT PPO - COMMAND INTEGRATION / NOT POSE CONTROL')
            p=env._robot.data.root_pos_w.torch[0].cpu().tolist()
            set_kit_renderer_camera_view(eye=(p[0]+.95,p[1]+1.25,1.20),target=(p[0],p[1]-.05,.12))
            last_raw,warmup=initial_rgb_frame(env);save(output/'rgb_warmup.json',warmup);validate_rgb(last_raw)
            obs=env._get_observations();check_packet(obs,env._commands)
            packet=TensorDict(obs,batch_size=[1]);first=policy(packet).clone();second=policy(packet)
            if not torch.equal(first,second):raise ValueError('Repeated deterministic action differs on same packet')
            before['same_packet_deterministic_action_exact']=True;save(output/'checkpoint_readback_before.json',before)
            writer=imageio.get_writer(str(output/'rollout.mp4'),fps=FPS,codec='libx264',quality=8)
            for step,row in enumerate(rows):
                request=torch.tensor(row['requested_command'],device=env.device,dtype=torch.float32).reshape(1,3)
                env.set_evaluation_targets(request);observed=env._commands.clone();obs=env._get_observations();check_packet(obs,observed)
                action=policy(TensorDict(obs,batch_size=[1]))
                if tuple(action.shape)!=(1,18) or not bool(torch.isfinite(action).all()):raise ValueError('Wrong/nonfinite deterministic action')
                actions.append(action.detach().cpu().numpy().copy())
                _,_,terminated,truncated,_=env.step(action)
                sample,event=capture_sample(env,terminated,truncated);samples.append(sample)
                if not np.array_equal(sample['terminated'],sample['returned_terminated']) or not np.array_equal(sample['truncated'],sample['returned_truncated']):raise ValueError('Pre-reset returned event mismatch retained')
                if not np.array_equal(sample['command'],observed.cpu().numpy()):raise ValueError('Pre-reset command differs from actor observation')
                if not all(np.isfinite(v).all() for v in sample.values()):raise ValueError('Nonfinite pre-reset sample retained')
                try:actual=actual_pose(sample['position_world_m'][0],sample['quaternion_world_xyzw'][0])
                except ValueError:
                    if not event:raise
                    actual=np.r_[sample['position_world_m'][0,:2],0.]
                measured=np.r_[sample['velocity_navigation_mps'][0,:2],sample['gyro_navigation_rad_s'][0,2]]
                peak=float(np.abs(sample['computed_torque_nm']).max());sat=float((np.abs(sample['computed_torque_nm'])>1.6).mean())
                trail.append(actual[:2].copy())
                telemetry.append({**row,'time_end_s':(step+1)*DT,'actor_observed_command':observed[0].cpu().tolist(),
                    'measured_forward_left_yaw':measured.tolist(),'position_world_m':sample['position_world_m'][0].tolist(),
                    'requested_torque_abs_max_nm':peak,'applied_torque_abs_max_nm':float(np.abs(sample['applied_torque_nm']).max()),
                    'requested_saturation_fraction':sat,'terminated':bool(terminated[0]),'truncated':bool(truncated[0])})
                if event:
                    terminal={'control_step':step+1,'segment':row['segment'],'terminated':bool(terminated[0]),'truncated':bool(truncated[0])}
                    frame=annotate(last_raw,row,observed[0].cpu().tolist(),measured,peak,sat,provenance,event='TERMINAL EVENT - RECORDING STOPPED')
                    writer.append_data(frame);frames+=1;imageio.imwrite(str(output/'last_valid_frame.png'),frame);break
                if step%2==1:
                    if step%10==1:
                        drawing.line('ActualTrail',np.asarray(trail),(1.,.25,.08),width=.007,z=.008)
                        draw_live_command(drawing,actual,row['requested_command'],row['segment'])
                    pos=sample['position_world_m'][0]
                    set_kit_renderer_camera_view(eye=(float(pos[0])+.95,float(pos[1])+1.25,1.20),target=(float(pos[0]),float(pos[1])-.05,.12))
                    last_raw=validate_rgb(env.render())
                    frame=annotate(last_raw,row,observed[0].cpu().tolist(),measured,peak,sat,provenance)
                    writer.append_data(frame);frames+=1
                    if frames==1:imageio.imwrite(str(output/'first_frame.png'),frame)
                    if step+1==STEPS:imageio.imwrite(str(output/'last_frame.png'),frame)
                if (step+1)%100==0:
                    save(output/'progress.json',{'complete':False,'control_steps':step+1,'frames':frames,'checkpoint_sha256':checkpoint_sha})
                    print(f'DIRECT_PREVIEW {step+1}/{STEPS} {row["segment"]}',flush=True)
        after=checkpoint_readback(runner,args.checkpoint);save(output/'checkpoint_readback_after.json',after)
    except BaseException as exc:error=repr(exc)
    finally:
        env.omni_diagnostic_enabled=False
        if writer is not None:
            try:writer.close()
            except BaseException as exc:error=error or ('writer_close: '+repr(exc))
        try:
            if samples:
                np.savez_compressed(output/'trace.npz',**{k:np.stack([s[k] for s in samples]) for k in samples[0]},
                    deterministic_actor_action=np.stack(actions[:len(samples)]),joint_names=np.array(env._robot.joint_names),
                    time_s=np.arange(1,len(samples)+1)*DT)
            if len(actions)>len(samples):
                np.savez_compressed(output/'unpaired_attempted_actions.npz',action=np.stack(actions[len(samples):]),
                    first_attempted_control_step=np.array(len(samples)+1))
            verified=verify_inputs(args)==provenance
            if not verified:error=error or 'Input provenance changed'
        except BaseException as exc:error=error or ('export_or_integrity: '+repr(exc))
        complete=error is None and terminal is None and len(samples)==STEPS and frames==FRAMES and verified
        result={**provenance,'complete':complete,'recorded_control_steps':len(samples),'frames':frames,
                'physics_duration_s':len(samples)*DT,'playback_duration_s':frames/FPS,'terminal_event':terminal,'error':error,
                'checkpoint_readback_before':before,'checkpoint_readback_after':after,'source_and_inputs_reverified_after_recording':verified,
                'video_sha256':sha(output/'rollout.mp4') if (output/'rollout.mp4').is_file() else None,
                'trace_sha256':sha(output/'trace.npz') if (output/'trace.npz').is_file() else None,'telemetry':telemetry,
                'scope':'Fresh continuous physics preview; short stops and complete frames are not qualification.',
                'legacy_raw_quaternion_key':'quaternion_world_wxyz is retained unchanged as SDK XYZW; explicit XYZW and converted WXYZ fields added.'}
        save(output/'video.json',result)
    if not result['complete']:raise RuntimeError('Incomplete unqualified preview: '+str(error or terminal))
    return result
