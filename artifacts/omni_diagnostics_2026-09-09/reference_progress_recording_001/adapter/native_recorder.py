"""Native RGB observer around unchanged source009 physics; no actor or actions."""
import math
from pathlib import Path
import numpy as np
from recording_contract import LABEL,FPS,DT,STEPS,segment,save_json,digest
from render_helpers import initial_rgb_frame,draw_live_command

class NativeRecorder:
    def __init__(self,args,provenance,startup_ready):
        self.args=args;self.provenance=provenance;self.startup_ready=startup_ready
        self.writer=None;self.steps=0;self.frames=0;self.last_sample=None;self.last_raw=None;self.trail=[];self.telemetry=[];self.terminal=None
    def annotate(self,raw,*,event=None):
        from PIL import Image,ImageDraw,ImageFont
        frame=Image.fromarray(np.asarray(raw[:,:,:3],dtype=np.uint8));draw=ImageDraw.Draw(frame)
        try:font=ImageFont.truetype('DejaVuSans.ttf',18)
        except OSError:
            try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
            except OSError:font=ImageFont.load_default()
        phase,forward=segment(max(0,self.steps-1));row=self.last_sample
        contacts='?' if row is None else str(int(row['distal_contact'].sum()))
        measured='?' if row is None else f"{-float(row['velocity_body_mps'][0,1]):+.4f} m/s"
        lines=[LABEL,f'{phase} | t={self.steps*DT:.2f}s | requested forward {forward:+.3f} m/s | measured {measured}',
               f'Full C robot | actual contacts {contacts}/6 | zero residual | fresh physics recording, source009',
               'Slow-stop reference: original009 needed 5.54s to quiet + 2s settling; no prompt-stop qualification']
        draw.rectangle((8,8,frame.width-8,110),fill=(14,20,29))
        for i,line in enumerate(lines):draw.text((18,14+i*23),line,fill='white',font=font)
        if event:
            draw.rectangle((8,118,frame.width-8,166),fill=(150,15,20));draw.text((18,130),event,fill='white',font=font)
        return np.asarray(frame)
    def start(self,env):
        if self.writer is not None:raise RuntimeError('No reset/restart during recording')
        import imageio.v2 as imageio
        from omni_path_demo import GroundDrawing
        from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
        self.drawing=GroundDrawing();self.camera=set_kit_renderer_camera_view
        pos=env._robot.data.root_pos_w.torch[0].detach().cpu().numpy()
        self.camera(eye=(float(pos[0])+.95,float(pos[1])+1.25,1.20),target=(float(pos[0]),float(pos[1])-.05,.12))
        self.drawing.text('ReferenceLabel','STEPPING REFERENCE / FORWARD 0.005 M/S / NOT PPO',float(pos[0])-.45,float(pos[1])+.48,pitch=.0025)
        self.last_raw,warmup=initial_rgb_frame(env);save_json(self.args.output/'rgb_warmup.json',warmup)
        self.writer=imageio.get_writer(str(self.args.output/'rollout.mp4'),fps=FPS,codec='libx264',quality=8)
        self.startup_ready()
    def capture(self,row):self.last_sample=row;return row
    def after_step(self,env,returned):
        if self.writer is None or self.last_sample is None:raise RuntimeError('Recording lacks ready renderer or pre-reset snapshot')
        self.steps+=1;row=self.last_sample
        if abs(float(row['time_s'][0])-self.steps*DT)>1e-9:raise RuntimeError('Recording/pre-reset physical timestamp mismatch')
        term=returned[2].detach().cpu().numpy();trunc=returned[3].detach().cpu().numpy()
        if not np.array_equal(row['terminated'],term) or not np.array_equal(row['truncated'],trunc):raise RuntimeError('Recording pre-reset terminal mismatch')
        if not np.array_equal(row['quaternion_world_wxyz'],row['quaternion_world_xyzw'][...,[3,0,1,2]]):raise RuntimeError('Raw XYZW conversion mismatch')
        pos=row['position_world_m'][0];R=row['rotation_world_from_body'][0];forward_axis=R@np.array([0.,-1.,0.])
        phase,command=segment(self.steps-1);self.trail.append(pos[:2].copy())
        self.telemetry.append(dict(step=self.steps,time_s=self.steps*DT,phase=phase,requested_forward_mps=command,actual_root_position_world_m=pos.tolist(),distal_contacts=int(row['distal_contact'].sum()),zero_residual=bool(not np.asarray(row['raw_residual_action']).any())))
        if term.any() or trunc.any():
            self.terminal=dict(step=self.steps,terminated=bool(term.any()),truncated=bool(trunc.any()))
            self.writer.append_data(self.annotate(self.last_raw,event='TERMINAL EVENT — last valid render; no post-reset continuation'));self.frames+=1
            return
        if self.steps%2==0:
            if self.steps%10==2:
                if np.linalg.norm(forward_axis[:2])<.1:raise RuntimeError('Projected body heading invalid')
                heading=math.atan2(float(forward_axis[1]),float(forward_axis[0]))
                draw_live_command(self.drawing,np.r_[pos[:2],heading],(command,0.,0.),phase)
                self.drawing.line('ActualTrail',np.asarray(self.trail),(1.,.25,.08),width=.007,z=.008)
            self.camera(eye=(float(pos[0])+.95,float(pos[1])+1.25,1.20),target=(float(pos[0]),float(pos[1])-.05,.12))
            self.last_raw=env.render()
            if self.last_raw is None or self.last_raw.shape!=(720,1280,3) or np.std(self.last_raw)<1:raise RuntimeError('Missing/blank/wrong-resolution native RGB frame')
            frame=self.annotate(self.last_raw);self.writer.append_data(frame);self.frames+=1
            if self.frames==1:
                import imageio.v2 as imageio
                imageio.imwrite(str(self.args.output/'first_frame.png'),frame)
            if self.steps==STEPS:
                import imageio.v2 as imageio
                imageio.imwrite(str(self.args.output/'last_frame.png'),frame)
    def close(self):
        if self.writer is not None:
            self.writer.close();self.writer=None
    def finish(self,reverified):
        self.close();state=json_read(self.args.output/'state.json')
        complete=bool(state.get('status')=='completed' and state.get('gate',{}).get('passed') and self.steps==STEPS and self.frames==1200 and self.terminal is None and reverified)
        report=dict(complete=complete,frames=self.frames,recorded_control_steps=self.steps,planned_control_steps=STEPS,fps=FPS,video_sha256=digest(self.args.output/'rollout.mp4') if (self.args.output/'rollout.mp4').exists() else None,source_manifest_sha256=self.provenance['source_manifest_sha256'],source_and_inputs_reverified_after_recording=reverified,stage2_complete=False,policy_training_started=False,pose_forcing=False,actor_present=False,source_screen_gate_passed=bool(state.get('gate',{}).get('passed')),source_screen_status=state.get('status'),source_screen_failure=state.get('failure',state.get('error')),terminal_event=self.terminal,physics_duration_s=self.steps*DT,playback_duration_s=self.frames/FPS,label=LABEL,fresh_physics_recording=True,original009_video=False,telemetry=self.telemetry,recording_source_hashes=self.provenance['recording_source_hashes'])
        save_json(self.args.output/'video.json',report);return report

def json_read(path):
    import json
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}

def build_recording_environment(args,options,recorder):
    from isaaclab.utils.io import dump_yaml
    import reference_residual_env as module
    from reference_physics_env import build_reference_environment
    original=module.ReferenceResidualPhysicsEnv
    class NativeRecordingEnv(original):
        def __init__(self,*a,**kw):
            cfg=kw['cfg'];overrides={'scene.num_envs':{'from':cfg.scene.num_envs,'to':1},'video_recorder.window_width':{'from':cfg.video_recorder.window_width,'to':1280},'video_recorder.window_height':{'from':cfg.video_recorder.window_height,'to':720},'render_mode':'rgb_array','enable_cameras':True,'unchanged_source009_physics':True}
            if cfg.scene.num_envs!=1:raise ValueError('Exact source009 single-replica wave required')
            cfg.video_recorder.window_width=1280;cfg.video_recorder.window_height=720;kw['render_mode']='rgb_array'
            dump_yaml(str(args.output/'recording_environment.yaml'),cfg);save_json(args.output/'recording_overrides.json',overrides)
            self._native_recording_ready=False;super().__init__(*a,**kw);self._native_recording_ready=True
        def reset(self,*a,**kw):
            result=super().reset(*a,**kw)
            if getattr(self,'_native_recording_ready',False):recorder.start(self)
            return result
        def step(self,actions):
            result=super().step(actions);recorder.after_step(self,result);return result
        def close(self):
            recorder.close();return super().close()
    module.ReferenceResidualPhysicsEnv=NativeRecordingEnv
    try:return build_reference_environment(args,options)
    finally:module.ReferenceResidualPhysicsEnv=original
