"""Build a self-contained HTML page that plays evaluated policies beside their reward.

Each evaluation directory holds ``control_trace.npz`` and optionally
``rollout.mp4`` and ``report.json``. The page syncs the recorded video, or a
top-down footprint view when no video exists, with the per-control reward and
its motionless counterfactual, the reward components, commanded versus
achieved velocity and a foot-contact gait diagram. Rewards and telemetry come
from :mod:`locomotion.reward_scorer`, so the page shows exactly what it scores.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import reward_scorer
from .env_config import LEGS
from .task import TaskConfig

CONTROL_DT_S = .02
# evaluate.py writes frame k after control 2k+1 at 25 fps, so frame time trails
# the control's end time by one video frame.
VIDEO_FPS = 25
VIDEO_OFFSET_S = 1 / VIDEO_FPS


def _series(values, digits=5):
    array = np.asarray(values, dtype=np.float64)
    return [float(f"{value:.{digits}g}") for value in array.tolist()]


def trace_view(path, rewards, nominal_height, output, *, config=None, com_local=None, replica=0):
    config = TaskConfig() if config is None else config
    com_local = reward_scorer.root_com_local() if com_local is None else com_local
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    trace = reward_scorer.load_trace(directory / "control_trace.npz" if path.is_dir() else path)
    summary = reward_scorer.score_trace(trace, rewards, config, nominal_height, com_local)
    controls, replicas = trace.data["command"].shape[:2]
    if not 0 <= replica < replicas:
        raise ValueError(f"{path}: replica {replica} outside 0..{replicas - 1}")

    def pick(value):
        value = torch.as_tensor(value)
        return value.reshape(controls - 1, replicas, *value.shape[1:])[:, replica].numpy()

    data = trace.data
    velocity = reward_scorer.origin_velocity_nav(trace, com_local)[1:, replica].numpy()
    scored = {}
    for name, fn in rewards.items():
        recorded, components = reward_scorer.evaluate(trace, fn, config, nominal_height, com_local)
        motionless, _ = reward_scorer.evaluate(trace, fn, config, nominal_height, com_local, motionless=True)
        scored[name] = {"recorded": _series(pick(recorded)), "motionless": _series(pick(motionless)),
                        "components": {key: _series(pick(value)) for key, value in components.items()}}
    video = directory / "rollout.mp4"
    return {
        "label": ", ".join(trace.case_ids) or directory.name,
        "path": str(trace.path),
        "replica": replica,
        "video": (Path(os.path.relpath(video.resolve(), Path(output).resolve().parent)).as_posix()
                  if video.exists() else None),
        "time_s": _series(data["time_s"][1:], 6),
        "command": [_series(data["command"][1:, replica, axis]) for axis in range(3)],
        "velocity": [_series(velocity[:, 0]), _series(velocity[:, 1]), _series(data["gyro_body_rad_s"][1:, replica, 2])],
        "contact": np.asarray(data["distal_contact"][1:, replica], dtype=int).T.tolist(),
        "toes": np.round(data["toe_xyz_world_m"][1:, replica, :, :2], 4).tolist(),
        "root": np.round(data["root_pose_xyzw"][1:, replica], 4).tolist(),
        "terminated": np.flatnonzero(data["terminated"][1:, replica]).tolist(),
        "rewards": scored,
        "summary": {"totals": summary["totals"], "groups": [{k: g[k] for k in ("command", "rows", "along_command_mps")}
                    for g in summary["groups"]], "nonfoot_contact_controls": summary["nonfoot_contact_controls"]},
    }


def build_page(paths, rewards, nominal_height, output, *, replica=0):
    views = [trace_view(path, rewards, nominal_height, output, replica=replica) for path in paths]
    payload = {"legs": list(LEGS), "control_dt_s": CONTROL_DT_S, "video_offset_s": VIDEO_OFFSET_S,
               "nominal_height_m": nominal_height, "rewards": list(rewards),
               "declared_gaps": list(reward_scorer.DECLARED_GAPS), "traces": views}
    # Keep trace labels and paths from closing the inline script element.
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return PAGE.replace("__DATA__", data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("evaluations", nargs="+", type=Path,
                        help="Evaluation directories, or control_trace.npz files beside their video.")
    parser.add_argument("--nominal-height", type=float, required=True,
                        help="Nominal root height (m) the reward was trained with; the run's "
                             "task_definition.json records it as nominal_plate_height_m.")
    reward_scorer.add_reward_arguments(parser)
    parser.add_argument("--replica", type=int, default=0, help="Replica column to show from batched traces.")
    parser.add_argument("--output", type=Path, required=True, help="HTML file to write.")
    args = parser.parse_args(argv)
    rewards = reward_scorer.selected_rewards(parser, args)
    args.output.write_text(build_page(args.evaluations, rewards, args.nominal_height, args.output, replica=args.replica))
    print(f"Wrote {args.output}")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Policy reward viewer</title>
<style>
:root{--bg:#f6f6f3;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--grid:#eeede8;
--accent:#1f6feb;--ghost:#9a9a94;--good:#1a7f37;--bad:#c2410c;--c1:#1f6feb;--c2:#c2410c;--c3:#1a7f37;
--c4:#8250df;--c5:#b08800;--c6:#0e7490;--c7:#be185d;--c8:#57606a;--c9:#9a6700;--c10:#0969da;--c11:#bf3989;--c12:#4d7c0f;
--swing:#f0efe9;--stance:#1d1d1b}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;--muted:#9a9a94;
--line:#34332f;--grid:#2a2926;--accent:#58a6ff;--ghost:#6b6b66;--good:#3fb950;--bad:#f0883e;--c1:#58a6ff;
--c2:#f0883e;--c3:#3fb950;--c4:#bc8cff;--c5:#e3b341;--c6:#39c5cf;--c7:#f778ba;--c8:#8b949e;
--c9:#d29922;--c10:#79c0ff;--c11:#db61a2;--c12:#8ddb4f;
--swing:#2a2926;--stance:#ecebe6}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header.top{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);
padding:12px 20px;display:flex;flex-wrap:wrap;gap:12px 20px;align-items:center}
header.top h1{font-size:16px;margin:0 8px 0 0}
label{color:var(--muted);display:flex;gap:6px;align-items:center;min-width:0}
#rewardSelect{max-width:min(320px,55vw);text-overflow:ellipsis}
button,select{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
main{padding:16px 20px;display:grid;gap:16px}
section.trace{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.head{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:baseline;margin-bottom:10px}
.head h2{font-size:15px;margin:0}
.head .path{color:var(--muted);font-size:12px;overflow-wrap:anywhere}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 12px}
.chip{border:1px solid var(--line);border-radius:999px;padding:2px 10px;font-size:12px;color:var(--muted)}
.chip b{color:var(--ink);font-weight:600}
.body{display:grid;grid-template-columns:minmax(260px,420px) minmax(0,1fr);gap:16px}
@media (max-width:900px){.body{grid-template-columns:minmax(0,1fr)}}
.media video,.media canvas.topdown{width:100%;border-radius:8px;background:#000;display:block}
.media canvas.topdown{background:var(--bg);aspect-ratio:1/1}
.readout{margin-top:10px;font-size:12px;font-variant-numeric:tabular-nums;display:grid;grid-template-columns:auto 1fr;gap:2px 10px}
.readout dt{color:var(--muted)}.readout dd{margin:0}
.charts{display:grid;gap:10px;min-width:0}
.chart{position:relative}
.chart .title{font-size:12px;color:var(--muted);margin:0 0 2px;display:flex;flex-wrap:wrap;gap:4px 12px}
.chart canvas{width:100%;height:130px;display:block;cursor:crosshair}
.chart.gait canvas{height:84px}
.key{display:inline-flex;align-items:center;gap:4px;cursor:pointer;user-select:none}
.key i{width:14px;height:3px;border-radius:2px;display:inline-block}
.key.off{opacity:.35}
footer{padding:0 20px 24px;color:var(--muted);font-size:12px}
footer li{margin:2px 0}
</style>
</head>
<body>
<header class="top">
  <h1>Policy reward viewer</h1>
  <button class="primary" id="play">Play</button>
  <label>Speed <select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option></select></label>
  <label><input type="checkbox" id="link" checked> Link timelines</label>
  <label>Components of <select id="rewardSelect"></select></label>
</header>
<main id="traces"></main>
<footer><b>Declared gaps</b><ul id="gaps"></ul>
The dashed grey line is the same reward for a motionless robot under the same commands.</footer>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('data').textContent);
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const palette = ['--c1','--c2','--c3','--c4','--c5','--c6','--c7','--c8','--c9','--c10','--c11','--c12'];
const fmt = (v, d=3) => v == null ? '–' : (Math.abs(v) < 1e-4 && v !== 0 ? v.toExponential(1) : v.toFixed(d));
let selected = D.rewards[0], playing = false, speed = 1, lastFrame = null;
const panels = [];

function el(tag, attrs={}, ...kids){const e=document.createElement(tag);
  for(const [k,v] of Object.entries(attrs)){if(k==='class')e.className=v;else if(k==='text')e.textContent=v;else e.setAttribute(k,v);}
  kids.forEach(k=>e.append(k));return e;}

class Chart{
  constructor(root, title, panel){this.panel=panel;this.wrap=el('div',{class:'chart'});this.head=el('div',{class:'title'},el('span',{text:title}));
    this.canvas=el('canvas');this.wrap.append(this.head,this.canvas);root.append(this.wrap);this.series=[];
    const seek=e=>{const r=this.canvas.getBoundingClientRect();panel.seek(this.x0+(e.clientX-r.left-this.pad)/(r.width-this.pad-4)*(this.x1-this.x0));};
    this.canvas.addEventListener('pointerdown',e=>{this.canvas.setPointerCapture(e.pointerId);seek(e);});
    this.canvas.addEventListener('pointermove',e=>{if(e.buttons)seek(e);});}
  setSeries(series, keys=true){this.series=series;this.head.querySelectorAll('.key').forEach(k=>k.remove());
    if(keys)series.filter(s=>s.key!==false).forEach(s=>{const k=el('span',{class:'key'+(s.hidden?' off':'')},el('i'),document.createTextNode(s.name));
      k.firstChild.style.background=css(s.color);k.onclick=()=>{s.hidden=!s.hidden;k.classList.toggle('off');this.drawBase();this.draw();};this.head.append(k);});
    this.drawBase();}
  size(){const r=this.canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;this.w=r.width;this.h=r.height;
    this.canvas.width=Math.max(1,r.width*dpr);this.canvas.height=Math.max(1,r.height*dpr);this.dpr=dpr;}
  drawBase(){this.size();const t=this.panel.t.time_s;this.x0=t[0];this.x1=t[t.length-1];this.pad=46;
    const vis=this.series.filter(s=>!s.hidden),all=vis.flatMap(s=>s.values).sort((a,b)=>a-b);
    // Percentile bounds keep a start-up transient from flattening the steady behaviour.
    let lo=all.length?all[Math.floor(all.length*.01)]:-1,hi=all.length?all[Math.ceil(all.length*.99)-1]:1;
    const clipped=all.length&&(all[0]<lo||all[all.length-1]>hi);
    if(hi-lo<1e-9){hi+=.5;lo-=.5;}const m=(hi-lo)*.12;this.lo=lo-m;this.hi=hi+m;
    if(!this.note){this.note=el('span');this.head.firstChild.after(this.note);}
    this.note.textContent=clipped?' · 1–99% axis':'';
    const c=document.createElement('canvas');c.width=this.canvas.width;c.height=this.canvas.height;const g=c.getContext('2d');g.scale(this.dpr,this.dpr);
    const X=v=>this.pad+(v-this.x0)/(this.x1-this.x0)*(this.w-this.pad-4),Y=v=>this.h-14-(v-this.lo)/(this.hi-this.lo)*(this.h-22);
    g.font='10px -apple-system,sans-serif';g.fillStyle=css('--muted');g.strokeStyle=css('--grid');g.lineWidth=1;
    for(let i=0;i<=3;i++){const v=this.lo+(this.hi-this.lo)*i/3,y=Y(v);g.beginPath();g.moveTo(this.pad,y);g.lineTo(this.w-4,y);g.stroke();g.fillText(fmt(v,2),2,y+3);}
    if(this.lo<0&&this.hi>0){g.strokeStyle=css('--ghost');g.beginPath();g.moveTo(this.pad,Y(0));g.lineTo(this.w-4,Y(0));g.stroke();}
    for(let s=0;s<=Math.floor(this.x1);s+=Math.max(1,Math.round((this.x1-this.x0)/8))){g.fillText(s+'s',X(s)-6,this.h-2);}
    this.panel.t.terminated.forEach(i=>{g.fillStyle=css('--bad');g.fillRect(X(t[i])-1,0,2,this.h-14);});
    g.save();g.beginPath();g.rect(this.pad,0,this.w-this.pad-4,this.h-14);g.clip();
    vis.forEach(s=>{g.strokeStyle=css(s.color);g.lineWidth=s.width||1.2;g.setLineDash(s.dash||[]);g.beginPath();
      s.values.forEach((v,i)=>{const x=X(t[i]),y=Y(v);i?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();g.setLineDash([]);});
    g.restore();this.base=c;this.X=X;}
  draw(){const g=this.canvas.getContext('2d');g.setTransform(1,0,0,1,0,0);g.clearRect(0,0,this.canvas.width,this.canvas.height);
    if(this.base)g.drawImage(this.base,0,0);g.scale(this.dpr,this.dpr);const x=this.X(this.panel.now());
    g.strokeStyle=css('--accent');g.lineWidth=1.5;g.beginPath();g.moveTo(x,0);g.lineTo(x,this.h-14);g.stroke();}
}

class Gait extends Chart{
  drawBase(){this.size();const t=this.panel.t,n=t.time_s.length;this.x0=t.time_s[0];this.x1=t.time_s[n-1];this.pad=46;
    const c=document.createElement('canvas');c.width=this.canvas.width;c.height=this.canvas.height;const g=c.getContext('2d');g.scale(this.dpr,this.dpr);
    const X=v=>this.pad+(v-this.x0)/(this.x1-this.x0)*(this.w-this.pad-4),rows=D.legs.length,rh=(this.h-14)/rows;
    g.font='10px -apple-system,sans-serif';
    D.legs.forEach((leg,r)=>{g.fillStyle=css('--muted');g.fillText(leg,4,r*rh+rh*.7);g.fillStyle=css('--swing');g.fillRect(this.pad,r*rh+1,this.w-this.pad-4,rh-2);
      g.fillStyle=css('--stance');t.contact[r].forEach((on,i)=>{if(on&&i<n-1){const a=X(t.time_s[i]),b=X(t.time_s[i+1]);g.fillRect(a,r*rh+1,Math.max(1,b-a+.3),rh-2);}});});
    this.base=c;this.X=X;}
}

class Panel{
  constructor(t){this.t=t;this.clock=t.time_s[0];
    const sec=el('section',{class:'trace'});this.sec=sec;
    sec.append(el('div',{class:'head'},el('h2',{text:t.label}),el('span',{class:'path',text:t.path+(t.replica?` · replica ${t.replica}`:'')})));
    const chips=el('div',{class:'chips'});
    D.rewards.forEach(r=>chips.append(el('span',{class:'chip'},document.createTextNode(`${r}: mean `),el('b',{text:fmt(t.summary.totals[r].recorded)}),
      document.createTextNode(` · motionless `),el('b',{text:fmt(t.summary.totals[r].motionless)}))));
    t.summary.groups.forEach(gp=>{if(gp.along_command_mps!=null)chips.append(el('span',{class:'chip'},document.createTextNode(`cmd ${gp.command.map(v=>fmt(v,3)).join(', ')}: along command `),el('b',{text:fmt(gp.along_command_mps,4)+' m/s'})));});
    if(t.terminated.length)chips.append(el('span',{class:'chip'},el('b',{text:'terminated'}),document.createTextNode(` at ${fmt(t.time_s[t.terminated[0]],2)} s`)));
    sec.append(chips);
    const body=el('div',{class:'body'}),media=el('div',{class:'media'}),charts=el('div',{class:'charts'});body.append(media,charts);sec.append(body);
    if(t.video){this.video=el('video',{src:t.video,muted:'',playsinline:'',preload:'auto'});this.video.muted=true;media.append(this.video);
      this.video.addEventListener('seeked',()=>render());this.video.addEventListener('error',()=>{media.prepend(el('p',{class:'chip',text:'Video could not load: '+t.video}));this.video.remove();this.video=null;this.makeTopdown(media);});}
    else this.makeTopdown(media);
    this.readout=el('dl',{class:'readout'});media.append(this.readout);
    this.rewardChart=new Chart(charts,'Reward per control',this);
    this.componentChart=new Chart(charts,'Reward components',this);
    this.velocityChart=new Chart(charts,'Velocity: commanded (dashed) vs achieved',this);
    this.gait=new Gait(charts,'Foot contact (dark = stance)',this);
    this.setReward();
    this.velocityChart.setSeries([
      {name:'forward m/s',values:t.velocity[0],color:'--c1'},{name:'',values:t.command[0],color:'--c1',dash:[4,3],key:false},
      {name:'left m/s',values:t.velocity[1],color:'--c2'},{name:'',values:t.command[1],color:'--c2',dash:[4,3],key:false},
      {name:'yaw rad/s',values:t.velocity[2],color:'--c3'},{name:'',values:t.command[2],color:'--c3',dash:[4,3],key:false}]);
    this.gait.setSeries([],false);}
  makeTopdown(media){this.topdown=el('canvas',{class:'topdown'});media.prepend(this.topdown);
    media.prepend(el('p',{class:'chip',text:'No rollout.mp4: top-down footprint view'}));}
  setReward(){const t=this.t,r=t.rewards,comp=r[selected].components;
    const series=D.rewards.map((name,i)=>({name,values:r[name].recorded,color:palette[i%palette.length],width:name===selected?1.6:1}));
    series.push({name:`${selected} · motionless`,values:r[selected].motionless,color:'--ghost',dash:[5,4]});
    this.rewardChart.setSeries(series);
    const mags=Object.entries(comp).map(([k,v])=>[k,v.reduce((a,b)=>a+Math.abs(b),0)/v.length]),top=Math.max(...mags.map(m=>m[1]),1e-12);
    this.componentChart.setSeries(mags.map(([k,m],i)=>({name:k,values:comp[k],color:palette[i%palette.length],hidden:m<.02*top})));}
  duration(){const t=this.t.time_s;return t[t.length-1];}
  now(){return this.video?Math.min(this.video.currentTime+D.video_offset_s,this.duration()):this.clock;}
  index(time){const t=this.t.time_s;let i=Math.round((time-t[0])/D.control_dt_s);return Math.max(0,Math.min(t.length-1,i));}
  seek(time,linked=true){time=Math.max(this.t.time_s[0],Math.min(this.duration(),time));
    if(linked&&document.getElementById('link').checked){panels.forEach(p=>p.seek(time,false));return;}
    if(this.video)this.video.currentTime=Math.max(0,time-D.video_offset_s);else this.clock=time;render();}
  tick(dt){if(!this.video){this.clock=Math.min(this.duration(),this.clock+dt*speed);}}
  done(){return this.video?this.video.ended||this.now()>=this.duration():this.clock>=this.duration();}
  draw(){[this.rewardChart,this.componentChart,this.velocityChart,this.gait].forEach(c=>c.draw());
    const i=this.index(this.now()),t=this.t,r=t.rewards[selected];
    const comps=Object.entries(r.components).map(([k,v])=>[k,v[i]]).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,4);
    const rows=[['time',fmt(t.time_s[i],2)+' s · control '+(i+1)],['command',t.command.map(c=>fmt(c[i],3)).join(', ')],
      ['achieved',`${fmt(t.velocity[0][i],3)} fwd, ${fmt(t.velocity[1][i],3)} left, ${fmt(t.velocity[2][i],3)} yaw`],
      ['reward',`${fmt(r.recorded[i])} (motionless ${fmt(r.motionless[i])})`],...comps.map(([k,v])=>[k,fmt(v,4)]),
      ['feet down',D.legs.filter((l,j)=>t.contact[j][i]).join(' ')||'none']];
    this.readout.replaceChildren(...rows.flatMap(([a,b])=>[el('dt',{text:a}),el('dd',{text:b})]));
    if(this.topdown)this.drawTopdown(i);}
  drawTopdown(i){const c=this.topdown,r=c.getBoundingClientRect(),dpr=devicePixelRatio||1;c.width=r.width*dpr;c.height=r.height*dpr;
    const g=c.getContext('2d');g.scale(dpr,dpr);const root=this.t.root[i],span=1,s=r.width/span;
    const P=(x,y)=>[r.width/2+(x-root[0])*s,r.height/2-(y-root[1])*s];
    g.strokeStyle=css('--grid');for(let k=-3;k<=3;k++){const gx=Math.round(root[0]*10)/10+k*.1,gy=Math.round(root[1]*10)/10+k*.1;
      let [a]=P(gx,0);g.beginPath();g.moveTo(a,0);g.lineTo(a,r.height);g.stroke();let [,b]=P(0,gy);g.beginPath();g.moveTo(0,b);g.lineTo(r.width,b);g.stroke();}
    g.strokeStyle=css('--ghost');g.beginPath();this.t.root.slice(0,i+1).forEach((p,k)=>{const [x,y]=P(p[0],p[1]);k?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();
    const toes=this.t.toes[i];g.strokeStyle=css('--muted');g.beginPath();[0,1,2,5,4,3,0].forEach((j,k)=>{const [x,y]=P(...toes[j]);k?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();
    toes.forEach((p,j)=>{const [x,y]=P(p[0],p[1]);g.beginPath();g.arc(x,y,6,0,7);if(this.t.contact[j][i]){g.fillStyle=css('--stance');g.fill();}else{g.strokeStyle=css('--stance');g.stroke();}
      g.fillStyle=css('--muted');g.font='10px sans-serif';g.fillText(D.legs[j],x+8,y+3);});
    const [qx,qy,qz,qw]=root.slice(3),fx=-(2*(qx*qy-qw*qz)),fy=-(1-2*(qx*qx+qz*qz)),[cx,cy]=P(root[0],root[1]);
    g.strokeStyle=css('--accent');g.lineWidth=2;g.beginPath();g.moveTo(cx,cy);g.lineTo(cx+fx*40,cy-fy*40);g.stroke();g.lineWidth=1;
    g.fillStyle=css('--muted');g.fillText('0.1 m grid · arrow = anatomical forward',6,r.height-6);}
}

function render(){panels.forEach(p=>p.draw());}
function loop(ts){if(!playing){lastFrame=null;return;}const dt=lastFrame==null?0:(ts-lastFrame)/1000;lastFrame=ts;
  panels.forEach(p=>p.tick(dt));render();if(panels.every(p=>p.done()))setPlaying(false);else requestAnimationFrame(loop);}
function setPlaying(on){playing=on;document.getElementById('play').textContent=on?'Pause':'Play';
  panels.forEach(p=>{if(p.video){p.video.playbackRate=speed;if(on){if(p.done())p.video.currentTime=0;p.video.play().catch(()=>{});}else p.video.pause();}
    else if(on&&p.done())p.clock=p.t.time_s[0];});
  if(on)requestAnimationFrame(loop);}

const sel=document.getElementById('rewardSelect');D.rewards.forEach(r=>sel.append(el('option',{text:r})));
sel.onchange=()=>{selected=sel.value;panels.forEach(p=>p.setReward());render();};
document.getElementById('play').onclick=()=>setPlaying(!playing);
document.getElementById('speed').onchange=e=>{speed=+e.target.value;panels.forEach(p=>{if(p.video)p.video.playbackRate=speed;});};
D.declared_gaps.forEach(g=>document.getElementById('gaps').append(el('li',{text:g})));
const root=document.getElementById('traces');
D.traces.forEach(t=>{const p=new Panel(t);root.append(p.sec);panels.push(p);});
panels.forEach(p=>{p.setReward();p.velocityChart.drawBase();p.gait.drawBase();});
new ResizeObserver(()=>{panels.forEach(p=>[p.rewardChart,p.componentChart,p.velocityChart,p.gait].forEach(c=>c.drawBase()));render();}).observe(root);
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',()=>{panels.forEach(p=>{p.setReward();p.velocityChart.drawBase();p.gait.drawBase();});render();});
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
