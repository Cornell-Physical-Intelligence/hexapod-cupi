"""New canonical smoke-only proposal; no historical task/checkpoint compatibility."""
import copy

SCHEMA='canonical_direct_drive_ppo_smoke_v1'
TASK_ID='Hexapod-Canonical-DirectDrive-PPO-Smoke-v1'
ADAPTER_SCHEMA='canonical_held_target_history81x5_v1'
NUM_ENVS=32
CONTROLS_PER_UPDATE=24
UPDATES=2
ACTOR_WIDTH=405
CRITIC_WIDTH=408
ACTION_WIDTH=18
CONTROL_DT=.020
PHYSICS_DT=.0025
TARGET_SCALE_RAD=.10
SLEW_RAD_PER_CONTROL=.040
INITIAL_STD=.02
SEED=20260910


def protocol():
    return {'schema':SCHEMA,'task_id':TASK_ID,'model_id':'hexapod_mkii_updated_v1',
            'urdf_sha256':'9492fde54c50170e940b49ccb3037d7b413743a5b77bc1d8707a539d44349e78',
            'model_sha256':'7126935b35398d2a7f03271335a8a1b8f2059354af2b19d9474f22b8e73e4881',
            'usd_sha256':'3be98420adc7923923757d6557d8492d5930f84128c1b1366b9c6db1bb03207c',
            'adapter_schema':ADAPTER_SCHEMA,'frames_oldest_to_newest':5,'frame_width':81,'actor_width':ACTOR_WIDTH,'critic_width':CRITIC_WIDTH,
            'actor_fields':['body_angular_velocity_forward_left_up:3','projected_gravity_forward_left_up:3','body_command_forward_left_yaw:3','q_minus_admitted_neutral:18','raw_sdk_joint_rate:18','previous_clipped_policy_action:18','actual_held_target_minus_q:18'],
            'critic_extra':'privileged_native_body_linear_velocity_forward_left_up:3','feature_unit_scales':'all1; SI and radians','empirical_normalization':True,
            'replicas':NUM_ENVS,'controls_per_update':CONTROLS_PER_UPDATE,'updates':UPDATES,'total_controls':48,'transitions':1536,
            'control_dt':CONTROL_DT,'physics_dt':PHYSICS_DT,'substeps':8,'external_forces_every_iteration':True,
            'target_scale_rad':TARGET_SCALE_RAD,'executed_target_slew_rad_per20ms':SLEW_RAD_PER_CONTROL,
            'joint_clamp':'exact named admitted lower/upper, no old soft-limit template',
            'initial_policy':'fresh; zero final actor linear layer; Gaussian standard deviation .02; parameter log(.02)','checkpoint_input':None,
            'command':'allzero48controls; smoke-only','auto_reset':False,'initial_reset':'exact admitted canonical cold reset once',
            'safety_failure':'abort and preserve prefix; never replace/reset row silently',
            'reward_schema':'canonical_quiet_smoke_reward_v1','walking_objective_adopted':False,
            'automatic_continuation':False,'Stage2_complete':False,'standing_quality_presumed':False}


def runner_config():
    network={'class_name':'MLPModel','hidden_dims':[512,256,128],'activation':'elu','obs_normalization':True}
    actor=copy.deepcopy(network);actor['distribution_cfg']={'class_name':'GaussianDistribution','init_std':INITIAL_STD,'std_type':'log'}
    critic=copy.deepcopy(network);critic['distribution_cfg']=None
    return {'num_steps_per_env':24,'obs_groups':{'actor':['policy'],'critic':['critic']},'multi_gpu':None,
            'actor':actor,'critic':critic,
            'algorithm':{'class_name':'PPO','num_learning_epochs':5,'num_mini_batches':4,'learning_rate':3e-4,
                         'schedule':'adaptive','gamma':.99,'lam':.95,'entropy_coef':.005,'desired_kl':.01,'max_grad_norm':1.,
                         'optimizer':'adam','value_loss_coef':1.,'use_clipped_value_loss':True,'clip_param':.2,
                         'normalize_advantage_per_mini_batch':False,'share_cnn_encoders':False,'rnd_cfg':None,'symmetry_cfg':None}}
