"""Actual policy-in-the-loop path demos with non-colliding ground annotations.

The reference is drawn, never imposed on the articulation. A bounded holonomic
feedforward/P pose controller generates body-twist requests for the learned gait.
This is a test controller with simulator pose, not the future deployed estimator/MPPI.
"""
import json
import math
from pathlib import Path
import numpy as np
import torch
from omni_flat_math import integrate_body_twist, slew_commands


def demo_specs():
    return [('straight', '1  STRAIGHT PASS', 8.),
            ('strafe', '2  SIDEWAYS PASS', 8.),
            ('arc', '3  TRANSLATE AND TURN', 10.),
            ('heading_fixed_curve', '4  CURVE WITH FIXED HEADING', 10.),
            ('turn', '5  TURN IN PLACE', 8.)]


def demo_command(name, time_s, moving_duration):
    t = time_s - 2.
    if t < 0 or t >= moving_duration: return [0., 0., 0.]
    if name == 'straight': return [.12, 0., 0.]
    if name == 'strafe': return [0., .12, 0.]
    if name == 'arc': return [.10, .04, .20]
    if name == 'heading_fixed_curve':
        a = .5 * math.pi * t / moving_duration
        return [.12*math.cos(a), .12*math.sin(a), 0.]
    if name == 'turn': return [0., 0., .25 if t < moving_duration/2 else -.25]
    raise ValueError(name)


def path_reference(name, moving_duration, dt, initial_pose):
    """Finite reference with two-second startup and three-second stop window."""
    poses=[];commands=[];pose=initial_pose.clone();command=torch.zeros_like(pose)
    for i in range(round((moving_duration+5.)/dt)):
        target=torch.tensor(demo_command(name,i*dt,moving_duration),device=pose.device,dtype=pose.dtype).reshape(1,3)
        command=slew_commands(command,target,dt)
        pose=integrate_body_twist(pose,command,dt)
        poses.append(pose[0].clone());commands.append(command[0].clone())
    return torch.stack(poses),torch.stack(commands)


def path_follower(actual, reference, feedforward):
    """Body heading is independent of travel tangent; bounded in trained units."""
    c,s=reference[:,2].cos(),reference[:,2].sin()
    world_ff=torch.stack((c*feedforward[:,0]-s*feedforward[:,1],
                          s*feedforward[:,0]+c*feedforward[:,1]),-1)
    world=world_ff+.6*(reference[:,:2]-actual[:,:2])
    world=world*(.20/world.norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp(max=1)
    c,s=actual[:,2].cos(),actual[:,2].sin()
    body=torch.stack((c*world[:,0]+s*world[:,1],-s*world[:,0]+c*world[:,1]),-1)
    error=reference[:,2]-actual[:,2]
    error=torch.atan2(error.sin(),error.cos())
    yaw=(feedforward[:,2]+error).clamp(-.4,.4)
    return torch.cat((body,yaw[:,None]),-1)


def actual_pose(env):
    from isaaclab.utils.math import quat_apply
    d=env._robot.data
    forward=quat_apply(d.root_quat_w.torch,torch.tensor([0.,-1.,0.],device=env.device).expand(env.num_envs,3))
    return torch.cat((d.root_pos_w.torch[:,:2],torch.atan2(forward[:,1],forward[:,0])[:,None]),-1)


class GroundDrawing:
    """USD display meshes only: no rigid bodies or colliders are created."""
    def __init__(self):
        import omni.usd
        from pxr import UsdGeom
        self.stage=omni.usd.get_context().get_stage()
        self.root='/World/OmniPathDemo'
        if self.stage.GetPrimAtPath(self.root): self.stage.RemovePrim(self.root)
        UsdGeom.Xform.Define(self.stage,self.root)

    def mesh(self,name,points,counts,indices,color):
        from pxr import UsdGeom
        mesh=UsdGeom.Mesh.Define(self.stage,self.root+'/'+name)
        mesh.CreatePointsAttr(points);mesh.CreateFaceVertexCountsAttr(counts)
        mesh.CreateFaceVertexIndicesAttr(indices);mesh.CreateDisplayColorAttr([color])
        mesh.CreateDoubleSidedAttr(True);mesh.CreateSubdivisionSchemeAttr('none')
        return mesh

    def line(self,name,xy,color,width=.008,z=.003):
        points=[];counts=[];indices=[]
        for p,q in zip(xy[:-1],xy[1:]):
            delta=np.array(q[:2])-p[:2];length=np.linalg.norm(delta)
            if length<1e-5:continue
            normal=np.array([-delta[1],delta[0]])/length*width/2
            k=len(points)
            points.extend([(float(v[0]),float(v[1]),z) for v in (p[:2]+normal,q[:2]+normal,q[:2]-normal,p[:2]-normal)])
            counts.append(4);indices.extend(range(k,k+4))
        return self.mesh(name,points,counts,indices,color)

    def arrows(self,name,poses,color,length=.075,z=.006):
        points=[];counts=[];indices=[]
        for x,y,a in poses:
            direction=np.array([math.cos(a),math.sin(a)])
            normal=np.array([-direction[1],direction[0]])
            origin=np.array([x,y]);tip=origin+length*direction
            polygon=[origin-.009*normal,origin+.009*normal,
                     origin+.55*length*direction+.009*normal,
                     origin+.55*length*direction+.028*normal,tip,
                     origin+.55*length*direction-.028*normal,
                     origin+.55*length*direction-.009*normal]
            # Explicit triangles make the concave arrow independent of triangulator choice.
            k=len(points);points.extend([(float(v[0]),float(v[1]),z) for v in polygon])
            for face in ((0,1,2),(0,2,6),(3,4,5)):
                counts.append(3);indices.extend(k+j for j in face)
        return self.mesh(name,points,counts,indices,color)

    def text(self,name,text,x,y,color=(.9,.9,.9),pitch=.004):
        # Raster glyph runs become flat USD mesh faces. Labels therefore lie on
        # the physical ground in the rendered scene, not just in a video overlay.
        from PIL import Image,ImageDraw
        mask=Image.new('L',(600,22),0);ImageDraw.Draw(mask).text((1,1),text,fill=255)
        box=mask.getbbox()
        if box is None:return
        bitmap=np.asarray(mask.crop(box))>100
        # Orient raster right/down axes toward the fixed overview camera.
        # Mapping pixel right to world +X mirrored the glyphs from that view.
        right=np.array([-1.2,.85]);right/=np.linalg.norm(right)
        down=np.array([.85,1.2]);down/=np.linalg.norm(down)
        points=[];counts=[];indices=[]
        for row in range(bitmap.shape[0]):
            on=np.flatnonzero(np.diff(np.r_[False,bitmap[row],False].astype(int)))
            for start,end in zip(on[::2],on[1::2]):
                origin=np.array([x,y]);k=len(points)
                corners=[origin+(col-bitmap.shape[1]/2)*pitch*right+r*pitch*down
                         for col,r in ((start,row),(end,row),(end,row+1),(start,row+1))]
                points.extend([(float(v[0]),float(v[1]),.004) for v in corners])
                counts.append(4);indices.extend(range(k,k+4))
        return self.mesh(name,points,counts,indices,color)

    def reference(self,poses,commands,title):
        xy=poses[:,:2]
        self.line('DesiredPath',xy,(.08,.48,1.))
        # Separate markers show travel tangent (blue) and body heading (gold).
        travel=[];heading=[];last=xy[0]-100
        for p,c in zip(poses[::25],commands[::25]):
            if np.linalg.norm(p[:2]-last)<.16 and np.linalg.norm(c[:2])>.01:continue
            if np.linalg.norm(c[:2])>.01:
                travel.append((p[0],p[1],p[2]+math.atan2(c[1],c[0])))
                heading.append((p[0]+.10*math.cos(p[2]+math.pi/2),p[1]+.10*math.sin(p[2]+math.pi/2),p[2]))
                last=p[:2]
        if travel:self.arrows('TravelArrows',travel,(.08,.48,1.))
        if heading:self.arrows('HeadingArrows',heading,(1.,.65,.05),length=.06)
        if not travel:
            self.arrows('TurnHeadingArrows',[(poses[0,0],poses[0,1],float(a)) for a in np.linspace(poses[:,2].min(),poses[:,2].max(),5)],(1.,.65,.05),length=.14)
        center=xy.mean(0);top=xy[:,1].max()+.30
        self.text('Title',title,float(center[0]),float(top))
        self.text('TravelLabel','BLUE: TRAVEL / DESIRED PATH',float(center[0]),float(top+.10),(.08,.48,1.),pitch=.003)
        self.text('HeadingLabel','GOLD: BODY HEADING',float(center[0]),float(top+.18),(1.,.65,.05),pitch=.003)
        self.text('ActualLabel','ORANGE: ACTUAL ROBOT TRAIL',float(center[0]),float(top+.26),(1.,.25,.08),pitch=.003)


@torch.inference_mode()
def record_path_demo(env,runner,plan,output,checkpoint_sha):
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw
    from tensordict import TensorDict
    from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
    from omni_flat_evaluation import save
    if env.num_envs!=1:raise RuntimeError('Path recording requires one full robot')
    policy=runner.get_inference_policy(device=env.device)
    trajectory=[];summaries=[];frames=0;global_step=0
    with imageio.get_writer(str(Path(output)/'rollout.mp4'),fps=25,codec='libx264',quality=8) as writer:
        for index,(name,title,moving_duration) in enumerate(demo_specs()):
            env.reset(seed=plan['omni']['evaluation_seeds'][0]+index);env.episode_length_buf.zero_()
            if env.render() is None: raise RuntimeError('No Isaac RGB output for path demo')
            poses,commands=path_reference(name,moving_duration,env.step_dt,actual_pose(env).clone())
            drawing=GroundDrawing();drawing.reference(poses.cpu().numpy(),commands.cpu().numpy(),title)
            xy=poses[:,:2].cpu().numpy();center=xy.mean(0)
            span=max(float(np.ptp(xy,axis=0).max()),.75)
            camera_height=max(1.65,span*1.8)
            set_kit_renderer_camera_view(eye=(float(center[0])+.85,float(center[1])+1.2,camera_height),
                                         target=(float(center[0]),float(center[1])+.13,0.))
            actual_trail=[];errors=[];nterm=ntrunc=0
            for i in range(len(poses)):
                target=path_follower(actual_pose(env),poses[i:i+1],commands[i:i+1])
                env.set_evaluation_targets(target)
                obs=TensorDict(env._get_observations(),batch_size=[1])
                _,_,term,trunc,_=env.step(policy(obs))
                actual=actual_pose(env);d=env._robot.data
                velocity=env._vector_in_command_frame(d.root_lin_vel_b.torch)[0].cpu().tolist()
                error=float((actual[0,:2]-poses[i,:2]).norm());errors.append(error)
                nterm+=int(term[0]);ntrunc+=int(trunc[0]);global_step+=1
                actual_trail.append(actual[0,:2].cpu().numpy())
                trajectory.append({'time_s':global_step*env.step_dt,'segment':name,
                    'reference_pose_xy_heading':poses[i].cpu().tolist(),'actual_pose_xy_heading':actual[0].cpu().tolist(),
                    'target':target[0].cpu().tolist(),'ramped_command':env._commands[0].cpu().tolist(),
                    'position_m':d.root_pos_w.torch[0].cpu().tolist(),'velocity_mps':velocity,
                    'yaw_rad_s':float(d.root_ang_vel_b.torch[0,2]),'path_error_m':error,
                    'terminated':bool(term[0]),'truncated':bool(trunc[0])})
                if global_step%2==0:
                    if i%10==1:drawing.line('ActualTrail',np.array(actual_trail),(1.,.25,.08),width=.007,z=.008)
                    raw=env.render()
                    if raw is None or raw.ndim!=3 or float(np.std(raw))<1:raise RuntimeError('Blank path-demo frame')
                    frame=Image.fromarray(raw[:,:,:3]);draw=ImageDraw.Draw(frame)
                    draw.rectangle((8,8,900,83),fill=(10,18,28))
                    draw.text((18,17),title+' | actual learned policy, full contact simulation',fill='white')
                    draw.text((18,39),f'Path error: {error:.3f} m | failures: {nterm+ntrunc} | request fwd/left/yaw: '+ ' / '.join(f'{v:+.2f}' for v in target[0].cpu().tolist()),fill='white')
                    draw.text((18,61),'Path controller uses simulator pose; no body motion is prescribed. Separate labeled trials.',fill='white')
                    writer.append_data(np.asarray(frame));frames+=1
            summaries.append({'path':name,'duration_s':len(poses)*env.step_dt,'mean_path_error_m':float(np.mean(errors)),
                              'p95_path_error_m':float(np.quantile(errors,.95)),'final_path_error_m':errors[-1],
                              'terminations':nterm,'truncations':ntrunc})
            print('OMNI_PATH_VIDEO '+json.dumps(summaries[-1]),flush=True)
    report={'complete':True,'variant':'f050_t060','architecture':plan['omni']['architecture'],
            'checkpoint_sha256':checkpoint_sha,'fps':25,'frames':frames,'duration_s':global_step*env.step_dt,
            'controller':'bounded holonomic feedforward plus pose feedback; ideal simulator localization',
            'ground_annotations':{'blue':'reference path and travel direction','gold':'desired body heading','orange':'actual robot trail'},
            'trials':summaries,'trajectory':trajectory}
    save(Path(output)/'video.json',report)
    return report
