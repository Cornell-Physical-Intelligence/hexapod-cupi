"""Read-only independent reproductions against the partner's exact input snapshot."""
from pathlib import Path
import hashlib,importlib.util,json,sys
from unittest.mock import Mock
import torch
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'tmp/claude_fable_terrain_001/inputs/isaaclab/hexapod_phase3/sensor_model.py'
spec=importlib.util.spec_from_file_location('_exact_sensor_snapshot',SRC)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
report={'source':str(SRC.relative_to(ROOT)),'source_sha256':hashlib.sha256(SRC.read_bytes()).hexdigest(),'torch':torch.__version__,'scope':'CPU-only counterexamples, no source edits or GPU; original code preserved'}
# A legal transport configuration permits receipt order to differ from acquisition order.
cfg=m.TransportModelCfg(.02,.04,.04,0.,.2)
s=m._DelayedStream(cfg,1);s._jitter_rng.uniform=Mock(side_effect=[.04,-.04])
s.enqueue({'range':torch.tensor([[10.]])},0.,torch.tensor([True]))
s.enqueue({'range':torch.tensor([[20.]])},.02,torch.tensor([True]))
a=s.read(.025);first={'capture_time_s':a.capture_time_s.tolist(),'value':a.values['range'].tolist()}
b=s.read(.09);second={'capture_time_s':b.capture_time_s.tolist(),'value':b.values['range'].tolist()}
assert first['value']==[[20.]] and second['value']==[[10.]]
report['older_arrival_overwrites_newer_acquisition']={'legal_transport':cfg.__dict__,'after_new_arrives':first,'after_old_arrives':second,'reproduced':True,'qualification':'Default sensor cadence/jitter combinations do not themselves create this reorder; a permitted custom latency ablation does.'}
# Match RSL inference rollout storage followed by an external selective reset.
s=m._DelayedStream(m.TransportModelCfg(.02,0.,0.,0.,.2),1)
with torch.inference_mode():
 s.enqueue({'range':torch.tensor([[10.],[20.]])},0.,torch.tensor([True,True]))
 frame=s.read(.01)
created={k:bool(torch.is_inference(v)) for k,v in frame.values.items()}
try:s.reset([0]);error=None
except RuntimeError as exc:error=str(exc)
assert error and 'inference tensor' in error
report['inference_rollout_then_selective_reset']={'persistent_value_inference':created,'error':error,'reproduced':True,'qualification':'This reproduces a context-dependent adapter integration failure, not an already deployed terrain failure.'}
# A monotonic clock guard is absent in the sensor transport itself.
s=m._DelayedStream(m.TransportModelCfg(.02,0.,0.,0.,.2),1)
s.enqueue({'range':torch.tensor([[1.]])},1.,torch.tensor([True]));s.read(1.01)
frame=s.read(.5)
assert frame.valid.item() and frame.age_s.item()<0
report['rewound_transport_read_admits_negative_age']={'age_s':frame.age_s.tolist(),'valid':frame.valid.tolist(),'reproduced':True,'qualification':'Caller misuse is rejected by the new observation adapter, but not by this reused sensor primitive; a future bridge must enforce one monotonic clock.'}
p=Path(__file__).with_name('report.json');p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
