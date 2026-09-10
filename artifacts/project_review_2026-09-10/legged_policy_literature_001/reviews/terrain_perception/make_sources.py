from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
rows=[
 ('lee2020','Learning Quadrupedal Locomotion over Challenging Terrain',2020,'Science Robotics','blind history', ['https://arxiv.org/abs/2010.11251','https://leggedrobotics.github.io/rl-blindloco/'], [],'Author project method/results and primary abstract; no training reproduction.'),
 ('rma2021','RMA: Rapid Motor Adaptation for Legged Robots',2021,'RSS','proprioceptive adaptation', ['https://arxiv.org/abs/2107.04034','https://ashish-kmr.github.io/rma-legged-robots/'], [],'Author project and primary abstract; adaptation concept, not exact port parameters.'),
 ('miki2022','Learning robust perceptive locomotion for quadrupedal robots in the wild',2022,'Science Robotics','privileged teacher / recurrent perceptive student', ['https://arxiv.org/abs/2201.08117','https://leggedrobotics.github.io/rl-perceptiveloco/'], [],'Author method summary and primary paper entry; no implementation validation.'),
 ('cms2023','Learning Visual Locomotion with Cross-Modal Supervision',2023,'ICRA','real visual supervision from proprioception', ['https://antonilo.github.io/vision_locomotion/'], ['https://github.com/antonilo/rl_locomotion'],'Author method, experiments, bibliography and official repository overview.'),
 ('extreme_parkour2024','Extreme Parkour with Legged Robots',2024,'ICRA','privileged geometry to depth / obstacle-conditioned heading', ['https://arxiv.org/abs/2309.14341','https://extreme-parkour.github.io/','https://extreme-parkour.github.io/resources/parkour.pdf'], ['https://github.com/chengxuxin/extreme-parkour'],'Author project, paper method excerpts, official repository setup; no reproduction.'),
 ('anymal_parkour2024','ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots',2024,'Science Robotics','hierarchical position / heading / time skills', ['https://arxiv.org/abs/2306.14874','https://sites.google.com/leggedrobotics.com/agile-navigation'], [],'Author pipeline and explicit position/heading/time formulation; primary paper entry.'),
 ('vpnav2022','Coupling Vision and Proprioception for Navigation of Legged Robots',2022,'CVPR','planner capability feedback / velocity commands', ['https://arxiv.org/abs/2112.02094','https://navigation-locomotion.github.io/','https://openaccess.thecvf.com/content/CVPR2022/papers/Fu_Coupling_Vision_and_Proprioception_for_Navigation_of_Legged_Robots_CVPR_2022_paper.pdf'], [],'Author method, command pipeline and primary paper; no claimed safety proof.'),
 ('abs2024','Agile But Safe: Learning Collision-Free High-Speed Legged Locomotion',2024,'RSS','learned reach-avoid / recovery / depth rays', ['https://arxiv.org/abs/2401.17583','https://agile-but-safe.github.io/','https://www.roboticsproceedings.org/rss20/p059.pdf'], ['https://github.com/LeCAR-Lab/ABS'],'Author four-module method and official repository overview.'),
 ('elevation2018','Probabilistic Terrain Mapping for Mobile Robots with Uncertain Localization',2018,'IEEE RA-L','uncertainty-aware robot-centric map', ['https://doi.org/10.1109/LRA.2018.2849506'], ['https://github.com/ANYbotics/elevation_mapping'],'Official implementation overview, citation and maintenance notice; not numerical reproduction.'),
 ('gpu_elevation2022','Elevation Mapping for Locomotion and Navigation using GPU',2022,'arXiv preprint','GPU elevation map', ['https://arxiv.org/abs/2204.12876'], ['https://github.com/leggedrobotics/elevation_mapping_cupy'],'Primary abstract and official software overview; no onboard timing claim.'),
 ('fast_lio2','FAST-LIO2: Fast Direct LiDAR-inertial Odometry',2022,'IEEE T-RO; 2021 preprint','LiDAR / IMU local odometry', ['https://arxiv.org/abs/2107.06829'], ['https://github.com/hku-mars/FAST_LIO'],'Primary paper entry and official setup/input requirements; no Mid-360 driver qualification.'),
 ('fast_livo2','FAST-LIVO2: Fast, Direct LiDAR-Inertial-Visual Odometry',2024,'arXiv preprint; author repository notes T-RO acceptance 2024','LiDAR / IMU / image odometry', ['https://arxiv.org/abs/2408.14035'], ['https://github.com/hku-mars/FAST-LIVO2'],'Primary abstract and official synchronization/calibration resources; no hardware integration.'),
]
works=[]
for ident,title,year,venue,category,primary,code,read in rows:
 works.append(dict(id=ident,title=title,year=year,venue=venue,category=category,primary_urls=primary,official_code_urls=code,read_scope=read,accessed_date='2026-09-10',code_executed=False,transfer_claim='Review recommendations are inferences; published demonstrations are not hexapod validation.'))
project_paths=[
 'tmp/sequential_footprint_reacquisition_001/README.md',
 'tmp/sequential_footprint_reacquisition_001/summary.json',
 'tmp/sequential_footprint_reacquisition_001/FREEZE_SHA256.json',
 'tmp/reference_physics_results_009/run/wave/state.json',
 'tmp/reference_physics_results_009/run/wave/reference_states.json',
 'tmp/reference_pair_root_gap_review_001/README.md',
 'robot/sensors/livox_mid360/source/Livox_Mid-360_User_Manual_EN.pdf',
 'artifacts/sensor_mount_study_2026-09-09/profiles.json',
]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
project=[]
for rel in project_paths:
 p=ROOT/rel
 project.append({'repository_relative_path':rel,'sha256':sha(p),'bytes':p.stat().st_size})
actual=json.loads((ROOT/'tmp/reference_physics_results_009/run/wave/reference_states.json').read_text())
final=actual[-1]['result']
# Preserve exact recorded timing values wherever nested within the serialized controller result.
def collect(v,pre=''):
 ans={}
 if isinstance(v,dict):
  for k,x in v.items():
   if k in ('stop_requested_time_s','reference_quiet_time_s'): ans[pre+k]=x
   if isinstance(x,(dict,list)):ans.update(collect(x,pre+k+'.'))
 elif isinstance(v,list):
  for i,x in enumerate(v):ans.update(collect(x,pre+str(i)+'.'))
 return ans
obj={
 'schema':'breadth_first_primary_paper_review_v1',
 'date':'2026-09-10',
 'scope':'12 locomotion/mapping works and one separate tentative drone-racing identification; no GPU, source adoption, hardware qualification or reproduced paper result.',
 'selection':'Representative mechanism coverage, not an exhaustive survey or novelty claim; 2020-2024 papers chosen for the current decision.',
 'works':works,
 'cross_domain_candidate':{
   'id':'swift2023','title':'Champion-level drone racing using deep reinforcement learning','year':2023,'venue':'Nature',
   'identification':'Tentative: the user referred to a super drone paper without a title.',
   'primary_urls':['https://www.nature.com/articles/s41586-023-06419-4','https://rpg.ifi.uzh.ch/docs/Nature23_Kaufmann.pdf'],
   'read_scope':'Primary paper pages 1-3 and methods context on empirical residual identification; no code reproduction.',
   'transfer_scope':'Empirical sensing/dynamics identification and useful-perception objectives only; known racing map and reference calibration differ from terrain exploration.',
   'accessed_date':'2026-09-10'},
 'hardware_primary_sources':[
  {'id':'realsense_d400_rev020','url':'https://www.realsenseai.com/download/21345/?tmstv=1780360410','revision':'020, August 2025','read_scope':'PDF table 4-11 (printed page85), sections4.4-4.7; screenshot requested and parsed table cross-checked.','D455_minimum_optical_depth_m':{'1280x720':0.52,'848x480':0.35,'640x360':0.26},'limit':'Not a guarantee of depth validity or outdoor accuracy.'},
  {'id':'livox_mid360_manual2024','url':'https://terra-1-g.djicdn.com/851d20f7b9f64838a34cd02351370894/Livox/Livox_Mid-360_User_Manual_EN.pdf','local_path':'robot/sensors/livox_mid360/source/Livox_Mid-360_User_Manual_EN.pdf','read_scope':'Local manufacturer PDF full text of printed pages2,8,9,20. Web PDF fetch failed; existing exact local primary document read instead.','azimuth_fov_degrees':360,'elevation_fov_degrees':[-7,52],'minimum_inverted_mount_surface_height_m':0.5,'near_range_limit':'0.1-0.2m reference-only precision; small/thin/low-reflectivity objects within1m may be unreliable.'}
 ],
 'project_evidence':project,
 'actual009_final_result_timing_fields':collect(final),
 'critical_distinctions':['ideal teacher vs sensor observation','instantaneous visibility vs accumulated usable map vs required-footprint coverage','measured planted contact vs future support','new architecture proposal vs old checkpoint compatibility','local odometry without RTK vs bounded global accuracy','finite reference quiet vs physically safe prompt stop'],
 'recommendations_are_inferences':True,
 'unchanged_map_lease_s':0.25,
 'lease_relaxation_adopted':False,
 'hardware_purchase_recommended':False,
 'current_prototype_used_as_hardware_evidence':False
}
(OUT/'sources.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
print('wrote',OUT/'sources.json',len(works),'core works; timing',obj['actual009_final_result_timing_fields'])
