"""Explain the current training reward and edit its coefficients against recorded rollouts.

Writes a self-contained HTML page. It describes every term of
``task.measured_reward`` with its formula, coefficients and share of the reward
on the given evaluation traces, and plots each term's response. Edited
coefficients recompute the reward live on those traces; the page checks its
arithmetic against ``measured_reward`` when it loads and exports the edit as a
``reward_config_v1`` file. ``locomotion.reward_scorer`` and
``locomotion.reward_viewer`` accept that file with ``--reward-config`` and
evaluate it through ``measured_reward`` itself.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from . import reward_scorer
from .task import REWARD_VERSION, TaskConfig, measured_reward

# Component: (what it measures, formula, when it applies, coefficients).
TERMS = {
    "linear_tracking": ("Planar velocity tracking",
                        "linear_tracking_weight · exp(−‖v_xy − c_xy‖² / linear_error_variance)",
                        "every control", ("linear_tracking_weight", "linear_error_variance")),
    "yaw_tracking": ("Yaw-rate tracking",
                     "yaw_tracking_weight · exp(−(ω_z − c_yaw)² / yaw_error_variance)",
                     "every control", ("yaw_tracking_weight", "yaw_error_variance")),
    "quiet_joint_rate": ("Joint motion while commanded to stand",
                         "−quiet_joint_rate_weight · φ(mean_j (q̇_j / quiet_joint_rate_scale_rad_s)²)",
                         "zero command only", ("quiet_joint_rate_weight", "quiet_joint_rate_scale_rad_s")),
    "quiet_target_motion": ("Joint-target changes while commanded to stand",
                            "−quiet_target_step_weight · φ(mean_j (Δtarget_j / quiet_target_step_scale_rad)²)",
                            "zero command only", ("quiet_target_step_weight", "quiet_target_step_scale_rad")),
    "tilt": ("Body tilt", "−tilt_weight · (g_x² + g_y²), g = gravity in the body frame",
             "every control", ("tilt_weight",)),
    "roll_pitch_rate": ("Roll and pitch rate", "−angular_xy_weight · mean(ω_x², ω_y²)",
                        "every control", ("angular_xy_weight",)),
    "vertical_velocity": ("Vertical velocity", "−vertical_velocity_weight · v_z²",
                          "every control", ("vertical_velocity_weight",)),
    "height": ("Body height error", "−height_weight · (z − nominal height)²",
               "every control", ("height_weight",)),
    "effort": ("Motor effort", "−normalized_effort_weight · mean_j Σ_substeps τ_j² / (8 · 1.6²)",
               "every control", ("normalized_effort_weight",)),
    "target_motion": ("Joint-target changes", "−target_movement_weight · mean_j (Δtarget_j / 0.04)²",
                      "every control", ("target_movement_weight",)),
    "nonfoot_contact": ("Body or upper-leg floor contact",
                        "−nonfoot_event_weight · [F > 1 N] − nonfoot_force_weight · F, F = largest non-tibia floor force",
                        "every control", ("nonfoot_event_weight", "nonfoot_force_weight")),
    "termination": ("Fall or joint-limit termination", "−terminal_penalty · [terminated]",
                    "terminal control", ("terminal_penalty",)),
}


def features(telemetry, previous_target, terminated, nominal_height):
    """Per-control inputs to each term before its coefficients apply."""
    t = {key: value.to(torch.float64) for key, value in telemetry.items()}
    velocity, gyro, pose, command = t["linear_velocity_nav"], t["angular_velocity_body"], t["root_pose_xyzw"], t["command"]
    delta = t["joint_target_rad"] - previous_target.to(torch.float64)
    x, y, z, w = (pose[:, 3:] / torch.linalg.vector_norm(pose[:, 3:], dim=-1, keepdim=True)).unbind(-1)
    force = t["other_body_force_max_400hz"]
    values = {
        "linear_error_sq": (velocity[:, :2] - command[:, :2]).square().sum(-1),
        "yaw_error_sq": (gyro[:, 2] - command[:, 2]).square(),
        "quiet": (command == 0).all(-1).to(torch.float64),
        "joint_rate_ms": t["joint_velocity_rad_s"].square().mean(-1),
        "target_step_ms": delta.square().mean(-1),
        "tilt": 4 * ((x*z - w*y).square() + (y*z + w*x).square()),
        "roll_pitch_ms": gyro[:, :2].square().mean(-1),
        "vertical_velocity_sq": velocity[:, 2].square(),
        "height_error_sq": (pose[:, 2] - nominal_height).square(),
        "effort": t["torque_square_sum_400hz"].mean(-1) / (8 * 1.6**2),
        "target_motion": (delta / .04).square().mean(-1),
        "nonfoot_event": (force > 1.).to(torch.float64),
        "nonfoot_force": force.clamp_min(0),
        "terminated": terminated.to(torch.float64),
    }
    return {key: value.numpy() for key, value in values.items()}


def _tail(u):
    return np.where(u <= 1, u, 1 + np.log(np.maximum(u, 1)))


def recombine(f, c):
    """Reference for the page's live arithmetic; must equal measured_reward's components."""
    return {
        "linear_tracking": c.linear_tracking_weight * np.exp(-f["linear_error_sq"] / c.linear_error_variance),
        "yaw_tracking": c.yaw_tracking_weight * np.exp(-f["yaw_error_sq"] / c.yaw_error_variance),
        "quiet_joint_rate": -c.quiet_joint_rate_weight * f["quiet"] * _tail(f["joint_rate_ms"] / c.quiet_joint_rate_scale_rad_s**2),
        "quiet_target_motion": -c.quiet_target_step_weight * f["quiet"] * _tail(f["target_step_ms"] / c.quiet_target_step_scale_rad**2),
        "tilt": -c.tilt_weight * f["tilt"],
        "roll_pitch_rate": -c.angular_xy_weight * f["roll_pitch_ms"],
        "vertical_velocity": -c.vertical_velocity_weight * f["vertical_velocity_sq"],
        "height": -c.height_weight * f["height_error_sq"],
        "effort": -c.normalized_effort_weight * f["effort"],
        "target_motion": -c.target_movement_weight * f["target_motion"],
        "nonfoot_contact": -c.nonfoot_event_weight * f["nonfoot_event"] - c.nonfoot_force_weight * f["nonfoot_force"],
        "termination": -c.terminal_penalty * f["terminated"],
    }


def _series(values):
    return [float(f"{value:.8g}") for value in np.asarray(values, dtype=np.float64).tolist()]


def trace_data(path, nominal_height, *, replica=0, com_local=None):
    com_local = reward_scorer.root_com_local() if com_local is None else com_local
    path = Path(path)
    trace = reward_scorer.load_trace(path / "control_trace.npz" if path.is_dir() else path)
    summary = reward_scorer.score_trace(trace, {"current": measured_reward}, TaskConfig(), nominal_height, com_local)
    controls, replicas = trace.data["command"].shape[:2]
    if not 0 <= replica < replicas:
        raise ValueError(f"{path}: replica {replica} outside 0..{replicas - 1}")
    rows = np.arange(controls - 1) * replicas + replica

    def phase(motionless):
        telemetry, previous, terminated = reward_scorer.reward_inputs(trace, com_local, motionless=motionless)
        reward, _ = measured_reward(telemetry, telemetry["command"], previous, terminated, TaskConfig(), nominal_height)
        values = features(telemetry, previous, terminated, nominal_height)
        return {key: _series(value[rows]) for key, value in values.items()}, _series(reward.numpy()[rows])

    recorded, recorded_reward = phase(False)
    motionless, motionless_reward = phase(True)
    return {"label": ", ".join(trace.case_ids) or trace.path.parent.name,
            "source": "/".join(trace.path.parent.parts[-3:]),
            "path": str(trace.path), "replica": replica, "time_s": _series(trace.data["time_s"][1:]),
            "recorded": recorded, "motionless": motionless,
            "python_reward": recorded_reward, "python_motionless_reward": motionless_reward,
            "groups": [{k: group[k] for k in ("command", "rows", "along_command_mps")} for group in summary["groups"]]}


def build_page(paths, nominal_height, *, replica=0):
    config = TaskConfig()
    payload = {
        "reward_version": REWARD_VERSION, "config_schema": reward_scorer.REWARD_CONFIG_SCHEMA,
        "nominal_height_m": nominal_height,
        "defaults": {field: getattr(config, field) for field in reward_scorer.REWARD_FIELDS},
        "terms": [{"name": name, "measures": measures, "formula": formula, "applies": applies, "fields": list(fields)}
                  for name, (measures, formula, applies, fields) in TERMS.items()],
        "probes": {"speeds_mps": list(config.speed_levels_mps), "yaw_rate_rad_s": config.yaw_rate_rad_s,
                   "arc": [config.arc_speed_mps, config.arc_yaw_rate_rad_s]},
        "criterion": .10,
        "declared_gaps": list(reward_scorer.DECLARED_GAPS),
        "traces": [trace_data(path, nominal_height, replica=replica) for path in paths],
    }
    # Keep trace labels and paths from closing the inline script element.
    data = json.dumps(payload, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    return PAGE.replace("__DATA__", data)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("evaluations", nargs="*", type=Path,
                        help="Evaluation directories or control_trace.npz files to score edits on.")
    parser.add_argument("--nominal-height", type=float, required=True,
                        help="Nominal root height (m) the reward was trained with; the run's "
                             "task_definition.json records it as nominal_plate_height_m.")
    parser.add_argument("--replica", type=int, default=0, help="Replica column to use from batched traces.")
    parser.add_argument("--output", type=Path, required=True, help="HTML file to write.")
    args = parser.parse_args(argv)
    args.output.write_text(build_page(args.evaluations, args.nominal_height, replica=args.replica))
    print(f"Wrote {args.output}")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reward workbench</title>
<style>
:root{--bg:#f6f6f3;--panel:#fff;--ink:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--grid:#eeede8;--accent:#1f6feb;
--ghost:#9a9a94;--good:#1a7f37;--good-bg:#e6f4ea;--bad:#c2410c;--bad-bg:#fdeee6;--changed:#fff5d6;--c1:#1f6feb;--c2:#c2410c;--c3:#1a7f37;--c4:#8250df}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe6;--muted:#9a9a94;--line:#34332f;
--grid:#2a2926;--accent:#58a6ff;--ghost:#6b6b66;--good:#3fb950;--good-bg:#12261a;--bad:#f0883e;--bad-bg:#2d1a0f;
--changed:#3a3217;--c1:#58a6ff;--c2:#f0883e;--c3:#3fb950;--c4:#bc8cff}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:16px 24px 12px;border-bottom:1px solid var(--line)}
header h1{font-size:18px;margin:0 0 4px}
header p{margin:0;color:var(--muted)}
.badge{display:inline-block;margin-top:8px;padding:2px 10px;border-radius:999px;font-size:12px}
.badge.ok{background:var(--good-bg);color:var(--good)}.badge.fail{background:var(--bad-bg);color:var(--bad)}
.layout{display:grid;grid-template-columns:320px minmax(0,1fr);gap:20px;padding:20px 24px;align-items:start}
@media (max-width:960px){.layout{grid-template-columns:minmax(0,1fr)}}
aside{position:sticky;top:12px;max-height:calc(100vh - 24px);overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
@media (max-width:960px){aside{position:static;max-height:none}}
aside h2,section h2{font-size:15px;margin:0 0 8px}
fieldset{border:none;border-top:1px solid var(--line);margin:10px 0 0;padding:8px 0 0}
legend{font-size:12px;color:var(--muted);padding:0}
.field{display:grid;grid-template-columns:1fr 104px;gap:6px;align-items:center;margin:4px 0;font-size:12px}
.field span{overflow-wrap:anywhere}
.field input{font:inherit;width:100%;padding:3px 6px;border:1px solid var(--line);border-radius:5px;background:var(--panel);color:var(--ink);font-variant-numeric:tabular-nums}
.field.changed input{background:var(--changed)}.field.invalid input{border-color:var(--bad)}
.buttons{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}
button{font:inherit;color:var(--ink);background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
main{display:grid;gap:20px;min-width:0}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
section>p{margin:0 0 10px;color:var(--muted)}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600}
td.formula{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;min-width:260px}
.source{color:var(--muted);font-size:11px;overflow-wrap:anywhere}
.bar{height:6px;border-radius:3px;background:var(--accent);min-width:1px}
.bar.old{background:var(--ghost);opacity:.6;margin-top:2px}
.pass{color:var(--good);font-weight:600}.fail{color:var(--bad);font-weight:600}
.delta.up{color:var(--good)}.delta.down{color:var(--bad)}
.curves{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
.curve p{margin:0 0 2px;font-size:12px;color:var(--muted)}
canvas{width:100%;height:150px;display:block}
.trace-chart{margin-top:8px}.trace-chart canvas{height:110px}
.trace-chart p{margin:8px 0 0;font-size:12px;color:var(--muted)}
.legend{font-size:12px;color:var(--muted);display:flex;gap:14px;flex-wrap:wrap;margin-bottom:6px}
.legend i{display:inline-block;width:14px;height:3px;border-radius:2px;vertical-align:middle;margin-right:4px}
textarea{width:100%;min-height:120px;font:12px ui-monospace,SFMono-Regular,Menlo,monospace;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink);padding:8px}
code{font:12px ui-monospace,SFMono-Regular,Menlo,monospace}
footer{padding:0 24px 24px;color:var(--muted);font-size:12px}
</style>
</head>
<body>
<header>
  <h1>Reward workbench</h1>
  <p id="subtitle"></p>
  <span class="badge" id="check"></span>
</header>
<div class="layout">
  <aside>
    <h2>Edit coefficients</h2>
    <div id="form"></div>
    <div class="buttons"><button id="reset">Reset to current</button><button class="primary" id="exportButton">Export edit</button></div>
  </aside>
  <main>
    <section>
      <h2>How the current reward works</h2>
      <p>Every 20 ms control earns the sum of the terms below: two tracking terms pay up to their weights, and the rest subtract.
      φ(u) = u for u ≤ 1 and 1 + ln u above, so the standing penalties grow slowly past their scale.
      Share is each term's mean absolute contribution across the loaded traces; the grey bar is the current reward.</p>
      <div class="scroll"><table id="terms"></table></div>
    </section>
    <section>
      <h2>Standing-still floor</h2>
      <p>Tracking reward a motionless robot collects under each command. docs/TRAINING.md asks for under 10% on the commanded
      component; the combined column adds the other tracking term, which a motionless robot matches exactly.</p>
      <div class="scroll"><table id="floor"></table></div>
    </section>
    <section>
      <h2>Response curves</h2>
      <div class="legend"><span><i style="background:var(--c1)"></i>edited</span><span><i style="background:var(--ghost)"></i>current</span></div>
      <div class="curves" id="curves"></div>
    </section>
    <section>
      <h2>Effect on recorded policies</h2>
      <p>Mean reward per control on each trace, and its lead over the same commands with the robot motionless.
      A useful reward should rank real walking above policies that do not move along the command.</p>
      <div class="scroll"><table id="effects"></table></div>
      <div id="traceCharts"></div>
    </section>
    <section id="exportSection" hidden>
      <h2>Exported edit</h2>
      <p id="exportNote"></p>
      <textarea id="exportText" readonly></textarea>
      <div class="buttons"><button id="copy">Copy</button><a id="download" download="reward_edit.json"><button>Download reward_edit.json</button></a></div>
      <p>Score it exactly, through <code>measured_reward</code>:</p>
      <textarea id="commands" readonly style="min-height:70px"></textarea>
      <p>Training still uses <code>locomotion/task.py</code>. To train with an edit, add it there as a new reward version and keep version 1 (docs/TRAINING.md).</p>
    </section>
  </main>
</div>
<footer id="footer"></footer>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const $=id=>document.getElementById(id);
const pct=v=>(100*v).toFixed(1)+'%';
const num=(v,d=4)=>v==null||!isFinite(v)?'–':v.toFixed(d);
function el(tag,attrs={},...kids){const e=document.createElement(tag);for(const[k,v]of Object.entries(attrs)){if(k==='class')e.className=v;else if(k==='text')e.textContent=v;else e.setAttribute(k,v);}kids.forEach(k=>e.append(k));return e;}

const tail=u=>u<=1?u:1+Math.log(u);
// Mirrors reward_workbench.recombine, which the tests hold equal to task.measured_reward.
function components(f,i,c){const q=f.quiet[i];return{
  linear_tracking:c.linear_tracking_weight*Math.exp(-f.linear_error_sq[i]/c.linear_error_variance),
  yaw_tracking:c.yaw_tracking_weight*Math.exp(-f.yaw_error_sq[i]/c.yaw_error_variance),
  quiet_joint_rate:q?-c.quiet_joint_rate_weight*tail(f.joint_rate_ms[i]/c.quiet_joint_rate_scale_rad_s**2):0,
  quiet_target_motion:q?-c.quiet_target_step_weight*tail(f.target_step_ms[i]/c.quiet_target_step_scale_rad**2):0,
  tilt:-c.tilt_weight*f.tilt[i],roll_pitch_rate:-c.angular_xy_weight*f.roll_pitch_ms[i],
  vertical_velocity:-c.vertical_velocity_weight*f.vertical_velocity_sq[i],height:-c.height_weight*f.height_error_sq[i],
  effort:-c.normalized_effort_weight*f.effort[i],target_motion:-c.target_movement_weight*f.target_motion[i],
  nonfoot_contact:-c.nonfoot_event_weight*f.nonfoot_event[i]-c.nonfoot_force_weight*f.nonfoot_force[i],
  termination:-c.terminal_penalty*f.terminated[i]};}
function evaluate(f,c){const n=f.quiet.length,total=new Float64Array(n),abs={};D.terms.forEach(t=>abs[t.name]=0);
  for(let i=0;i<n;i++){const k=components(f,i,c);let s=0;for(const name in k){s+=k[name];abs[name]+=Math.abs(k[name]);}total[i]=s;}
  return{total,abs,n};}
const mean=a=>a.reduce((x,y)=>x+y,0)/Math.max(1,a.length);

let edited={...D.defaults};const inputs={};
const SCALE=new Set(['linear_error_variance','yaw_error_variance','quiet_joint_rate_scale_rad_s','quiet_target_step_scale_rad']);
function valid(field,v){return isFinite(v)&&(SCALE.has(field)?v>0:v>=0);}

function buildForm(){const form=$('form');
  D.terms.forEach(t=>{const fs=el('fieldset',{},el('legend',{text:t.name}));
    t.fields.forEach(field=>{const input=el('input',{type:'number',step:'any',value:String(D.defaults[field]),'aria-label':field});
      const row=el('label',{class:'field'},el('span',{text:field}),input);inputs[field]={input,row};
      input.addEventListener('input',()=>{const v=parseFloat(input.value);row.classList.toggle('invalid',!valid(field,v));
        if(valid(field,v))edited[field]=v;row.classList.toggle('changed',valid(field,v)&&v!==D.defaults[field]);update();});
      fs.append(row);});form.append(fs);});}

function lineChart(canvas,xs,series,{marks=[],robust=false,xUnit=''}={}){
  const r=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;canvas.width=Math.max(1,r.width*dpr);canvas.height=Math.max(1,r.height*dpr);
  const g=canvas.getContext('2d');g.scale(dpr,dpr);const W=r.width,H=r.height,pad=46,bot=16;
  let all=series.flatMap(s=>Array.from(s.values)).filter(isFinite).sort((a,b)=>a-b);
  let lo=all.length?(robust?all[Math.floor(all.length*.01)]:all[0]):0,hi=all.length?(robust?all[Math.ceil(all.length*.99)-1]:all[all.length-1]):1;
  if(hi-lo<1e-9){hi+=.5;lo-=.5;}const m=(hi-lo)*.1;lo-=m;hi+=m;
  const x0=xs[0],x1=xs[xs.length-1],X=v=>pad+(v-x0)/(x1-x0)*(W-pad-6),Y=v=>H-bot-(v-lo)/(hi-lo)*(H-bot-6);
  g.font='10px -apple-system,sans-serif';g.strokeStyle=css('--grid');g.fillStyle=css('--muted');
  for(let k=0;k<=3;k++){const v=lo+(hi-lo)*k/3,y=Y(v);g.beginPath();g.moveTo(pad,y);g.lineTo(W-6,y);g.stroke();g.fillText(v.toFixed(2),2,y+3);}
  const raw=(x1-x0)/4,mag=10**Math.floor(Math.log10(raw)),step=[1,2,5,10].map(s=>s*mag).find(s=>s>=raw);
  for(let v=Math.ceil(x0/step)*step;v<=x1+1e-9;v+=step){const text=+v.toPrecision(3)+xUnit,w=g.measureText(text).width;
    g.fillText(text,Math.min(Math.max(X(v)-w/2,pad-6),W-w-2),H-2);}
  marks.forEach(mk=>{g.strokeStyle=css('--ghost');g.setLineDash([2,3]);g.beginPath();g.moveTo(X(mk.x),6);g.lineTo(X(mk.x),H-bot);g.stroke();g.setLineDash([]);
    g.fillStyle=css('--muted');g.fillText(mk.label,X(mk.x)+3,14);});
  g.save();g.beginPath();g.rect(pad,0,W-pad-6,H-bot);g.clip();
  series.forEach(s=>{g.strokeStyle=css(s.color);g.lineWidth=s.width||1.4;g.setLineDash(s.dash||[]);g.beginPath();
    s.values.forEach((v,i)=>{const x=X(xs[i]),y=Y(v);i?g.lineTo(x,y):g.moveTo(x,y);});g.stroke();g.setLineDash([]);});g.restore();}

function termsTable(cur,now){const t=$('terms');const sum=o=>Object.values(o).reduce((a,b)=>a+b,0);
  const sc=sum(cur)||1,sn=sum(now)||1,top=Math.max(...Object.values(now).map(v=>v/sn),...Object.values(cur).map(v=>v/sc),1e-9);
  t.replaceChildren(el('tr',{},...['Term','Measures','Formula','Applies','Coefficients','Share'].map(h=>el('th',{text:h}))));
  D.terms.forEach(term=>{const share=now[term.name]/sn,old=cur[term.name]/sc;
    const coeffs=term.fields.map(f=>`${f} = ${edited[f]}`+(edited[f]!==D.defaults[f]?` (was ${D.defaults[f]})`:'')).join('\n');
    const bars=el('div',{},el('div',{class:'bar',style:`width:${100*share/top}%`}),el('div',{class:'bar old',style:`width:${100*old/top}%`}));
    t.append(el('tr',{},el('td',{text:term.name}),el('td',{text:term.measures}),el('td',{class:'formula',text:term.formula}),
      el('td',{text:term.applies}),el('td',{class:'formula',text:coeffs,style:'white-space:pre-line;min-width:200px'}),
      el('td',{style:'min-width:110px'},document.createTextNode(`${pct(share)} (was ${pct(old)})`),bars)));});}

function floor(c){const rows=[];const lin=(v)=>Math.exp(-v*v/c.linear_error_variance),yaw=(w)=>Math.exp(-w*w/c.yaw_error_variance);
  const combined=(v,w)=>(c.linear_tracking_weight*lin(v)+c.yaw_tracking_weight*yaw(w))/(c.linear_tracking_weight+c.yaw_tracking_weight||1);
  D.probes.speeds_mps.forEach(v=>rows.push({label:`translation ${v} m/s`,commanded:lin(v),combined:combined(v,0)}));
  rows.push({label:`yaw ${D.probes.yaw_rate_rad_s} rad/s`,commanded:yaw(D.probes.yaw_rate_rad_s),combined:combined(0,D.probes.yaw_rate_rad_s)});
  const[av,aw]=D.probes.arc;rows.push({label:`arc ${av} m/s, ${aw} rad/s`,commanded:Math.max(lin(av),yaw(aw)),combined:combined(av,aw)});return rows;}
function floorTable(){const cur=floor(D.defaults),now=floor(edited),t=$('floor');
  t.replaceChildren(el('tr',{},...['Command','Commanded component','Combined tracking',`Under ${pct(D.criterion)}?`].map(h=>el('th',{text:h}))));
  now.forEach((r,i)=>{const ok=r.commanded<D.criterion;t.append(el('tr',{},el('td',{text:r.label}),
    el('td',{text:`${pct(r.commanded)} (was ${pct(cur[i].commanded)})`}),el('td',{text:`${pct(r.combined)} (was ${pct(cur[i].combined)})`}),
    el('td',{class:ok?'pass':'fail',text:ok?'yes':'no'})));});}

function curves(){const box=$('curves');if(!box.children.length){['Linear tracking vs forward speed (solid 0.05 m/s command, dotted 0.025)','Yaw tracking vs yaw-rate error','Standing penalty vs joint RMS rate'].forEach((title,i)=>{
    box.append(el('div',{class:'curve'},el('p',{text:title}),el('canvas',{id:'curve'+i})));});}
  const xs=Array.from({length:161},(_,i)=>-.05+.2*i/160),ys=Array.from({length:161},(_,i)=>-.6+1.2*i/160),rs=Array.from({length:161},(_,i)=>.3*i/160);
  const lin=(c,v,cmd)=>c.linear_tracking_weight*Math.exp(-((v-cmd)**2)/c.linear_error_variance),[s0,s1]=D.probes.speeds_mps;
  lineChart($('curve0'),xs,[{values:xs.map(v=>lin(D.defaults,v,s1)),color:'--ghost',dash:[4,3]},{values:xs.map(v=>lin(D.defaults,v,s0)),color:'--ghost',dash:[1,3]},
    {values:xs.map(v=>lin(edited,v,s1)),color:'--c1'},{values:xs.map(v=>lin(edited,v,s0)),color:'--c1',dash:[1,3]}],
    {marks:[{x:0,label:'motionless'},{x:s1,label:`${s1} cmd`}],xUnit:''});
  const yaw=(c,e)=>c.yaw_tracking_weight*Math.exp(-e*e/c.yaw_error_variance);
  lineChart($('curve1'),ys,[{values:ys.map(e=>yaw(D.defaults,e)),color:'--ghost',dash:[4,3]},{values:ys.map(e=>yaw(edited,e)),color:'--c1'}],{marks:[{x:0,label:'exact'}]});
  const quiet=(c,r)=>-c.quiet_joint_rate_weight*tail((r/c.quiet_joint_rate_scale_rad_s)**2);
  lineChart($('curve2'),rs,[{values:rs.map(r=>quiet(D.defaults,r)),color:'--ghost',dash:[4,3]},{values:rs.map(r=>quiet(edited,r)),color:'--c1'}],
    {marks:[{x:edited.quiet_joint_rate_scale_rad_s,label:'scale'}]});}

let baseline=null;
function effects(){const t=$('effects'),charts=$('traceCharts');if(!D.traces.length){t.replaceChildren(el('tr',{},el('td',{text:'No traces loaded: pass evaluation directories to see the effect of an edit.'})));return {cur:{},now:{}};}
  const results=D.traces.map(tr=>({tr,cur:evaluate(tr.recorded,D.defaults),now:evaluate(tr.recorded,edited),
    curStill:evaluate(tr.motionless,D.defaults),nowStill:evaluate(tr.motionless,edited)}));
  const rank=key=>results.map((r,i)=>[mean(r[key].total),i]).sort((a,b)=>b[0]-a[0]).map(x=>x[1]);
  const rc=rank('cur'),rn=rank('now');
  t.replaceChildren(el('tr',{},...['Trace','Along command','Mean reward','Lead over motionless','Rank'].map(h=>el('th',{text:h}))));
  results.forEach((r,i)=>{const mc=mean(r.cur.total),mn=mean(r.now.total),lc=mc-mean(r.curStill.total),ln=mn-mean(r.nowStill.total);
    const along=r.tr.groups.filter(g=>g.along_command_mps!=null).map(g=>`${num(g.along_command_mps)} m/s`).join(', ')||'stand only';
    const d=(a,b)=>el('span',{class:'delta '+(b>a+1e-9?'up':b<a-1e-9?'down':''),text:` (was ${num(a)})`});
    t.append(el('tr',{},el('td',{},document.createTextNode(r.tr.label),el('div',{class:'source',text:r.tr.source})),el('td',{text:along}),el('td',{},document.createTextNode(num(mn)),d(mc,mn)),
      el('td',{},document.createTextNode(num(ln)),d(lc,ln)),el('td',{text:`${rn.indexOf(i)+1} (was ${rc.indexOf(i)+1})`})));});
  if(!charts.children.length)D.traces.forEach((tr,i)=>charts.append(el('div',{class:'trace-chart'},el('p',{text:`${tr.label} (${tr.source}) · reward per control, edited in blue, current in grey`}),el('canvas',{id:'trace'+i}))));
  results.forEach((r,i)=>lineChart($('trace'+i),r.tr.time_s,[{values:r.cur.total,color:'--ghost',width:1},{values:r.now.total,color:'--c1',width:1.1}],{robust:true,xUnit:'s'}));
  const sumAbs=key=>{const o={};D.terms.forEach(t=>o[t.name]=results.reduce((a,r)=>a+r[key].abs[t.name],0));return o;};
  return{cur:sumAbs('cur'),now:sumAbs('now')};}

function update(){const shares=effects();termsTable(shares.cur,shares.now);floorTable();curves();if(!$('exportSection').hidden)exportEdit();}

function exportEdit(){const overrides={};for(const f in edited)if(edited[f]!==D.defaults[f])overrides[f]=edited[f];
  const zero=Object.entries(edited).filter(([,v])=>v<=0).map(([k])=>k);
  const doc={schema:D.config_schema,base_reward_version:D.reward_version,overrides};const text=JSON.stringify(doc,null,2)+'\n';
  $('exportText').value=text;$('download').href=URL.createObjectURL(new Blob([text],{type:'application/json'}));
  $('exportNote').textContent=(Object.keys(overrides).length?`${Object.keys(overrides).length} coefficient(s) changed from ${D.reward_version}.`:'No coefficients changed yet.')+
    (zero.length?` Training requires every coefficient above zero, so the scorer rejects this edit: ${zero.join(', ')}.`:'');
  $('commands').value=`uv run python -m locomotion.reward_scorer <traces...> --nominal-height ${D.nominal_height_m} --reward-config edited=reward_edit.json\n`+
    `uv run python -m locomotion.reward_viewer <evaluation dirs...> --nominal-height ${D.nominal_height_m} --reward-config edited=reward_edit.json --output viewer.html`;
  $('exportSection').hidden=false;}

function selfCheck(){let worst=0,rows=0;D.traces.forEach(tr=>{[[tr.recorded,tr.python_reward],[tr.motionless,tr.python_motionless_reward]].forEach(([f,py])=>{
    const js=evaluate(f,D.defaults).total;py.forEach((v,i)=>{worst=Math.max(worst,Math.abs(v-js[i]));rows++;});});});
  const b=$('check');if(!rows){b.className='badge';b.textContent='No traces loaded, so the live arithmetic is unchecked';return;}
  const ok=worst<1e-4;b.className='badge '+(ok?'ok':'fail');
  b.textContent=ok?`Live arithmetic matches task.measured_reward on ${rows} controls (largest difference ${worst.toExponential(1)})`:
    `Live arithmetic differs from task.measured_reward by up to ${worst.toExponential(2)}; trust only exported, re-scored edits`;}

$('subtitle').textContent=`${D.reward_version} · locomotion/task.py measured_reward · nominal height ${D.nominal_height_m} m · ${D.traces.length} trace(s)`;
$('footer').append(el('b',{text:'Declared gaps. '}),document.createTextNode(D.declared_gaps.join(' ')));
$('reset').onclick=()=>{edited={...D.defaults};for(const f in inputs){inputs[f].input.value=String(D.defaults[f]);inputs[f].row.classList.remove('changed','invalid');}update();};
$('exportButton').onclick=()=>{exportEdit();$('exportSection').scrollIntoView({behavior:'smooth'});};
$('copy').onclick=()=>{$('exportText').select();navigator.clipboard?.writeText($('exportText').value).catch(()=>document.execCommand('copy'));};
buildForm();selfCheck();update();
new ResizeObserver(()=>update()).observe(document.querySelector('main'));
matchMedia('(prefers-color-scheme: dark)').addEventListener('change',update);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
