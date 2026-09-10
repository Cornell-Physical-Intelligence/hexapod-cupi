"""One exact48-fixture geometry/ray/contact phase; no robot, actor or training."""
from pathlib import Path
import hashlib,json,math

CATALOG = 'artifacts/terrain_readiness_2026-09-09/mild_curriculum_002/terrain_catalog.json'
CATALOG_SHA256 = 'e494a82c767a368c4105ce3a2a88675b47bede0ea1a88bb91de32b992719fc02'
EXPECTED_IDS = ('mild0_train_smooth_rough_1103', 'mild0_train_smooth_rough_2207', 'mild0_train_smooth_rough_3301', 'mild0_train_smooth_rough_4409', 'mild0_train_ramp_1103', 'mild0_train_ramp_2207', 'mild0_train_ramp_3301', 'mild0_train_ramp_4409', 'mild0_heldout_smooth_rough_7103', 'mild0_heldout_smooth_rough_8209', 'mild0_heldout_ramp_7103', 'mild0_heldout_ramp_8209', 'mild1_train_smooth_rough_1103', 'mild1_train_smooth_rough_2207', 'mild1_train_smooth_rough_3301', 'mild1_train_smooth_rough_4409', 'mild1_train_ramp_1103', 'mild1_train_ramp_2207', 'mild1_train_ramp_3301', 'mild1_train_ramp_4409', 'mild1_heldout_smooth_rough_7103', 'mild1_heldout_smooth_rough_8209', 'mild1_heldout_ramp_7103', 'mild1_heldout_ramp_8209', 'mild2_train_smooth_rough_1103', 'mild2_train_smooth_rough_2207', 'mild2_train_smooth_rough_3301', 'mild2_train_smooth_rough_4409', 'mild2_train_ramp_1103', 'mild2_train_ramp_2207', 'mild2_train_ramp_3301', 'mild2_train_ramp_4409', 'mild2_train_step_1103', 'mild2_train_step_2207', 'mild2_train_step_3301', 'mild2_train_step_4409', 'mild2_train_ridge_1103', 'mild2_train_ridge_2207', 'mild2_train_ridge_3301', 'mild2_train_ridge_4409', 'mild2_heldout_smooth_rough_7103', 'mild2_heldout_smooth_rough_8209', 'mild2_heldout_ramp_7103', 'mild2_heldout_ramp_8209', 'mild2_heldout_step_7103', 'mild2_heldout_step_8209', 'mild2_heldout_ridge_7103', 'mild2_heldout_ridge_8209')
PROTOCOL = {'name':'mild48_fixture_geometry_contact001','catalog_sha256':CATALOG_SHA256,'fixture_ids':list(EXPECTED_IDS),'fixture_count':48,'steps':500,'dt_s':.005,'phase_timeout_s':600,'app_ready_timeout_s':90,'app_ready_marker':'TERRAIN_PHASE loading_geometry_helpers','robot_validation_performed':False,'ready_for_terrain_training':False,'PPO_permitted':False}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def catalog_identity(source):
    source=Path(source).resolve();path=source/CATALOG
    if digest(path)!=CATALOG_SHA256:raise ValueError('Exact mild_curriculum_002 catalog required')
    catalog=json.loads(path.read_text());entries=catalog.get('fixtures',[])
    if tuple(e['id'] for e in entries)!=EXPECTED_IDS or len(entries)!=48:raise ValueError('All48 ordered fixture IDs required')
    hashes={CATALOG:digest(path)}
    for e in entries:
        usd=(path.parent/e['usda']).resolve();npz=usd.with_suffix('.npz')
        if not usd.is_relative_to(path.parent) or not npz.is_relative_to(path.parent):raise ValueError('Fixture escapes its catalog')
        if digest(usd)!=e['sha256'] or digest(npz)!=e['npz_sha256']:raise ValueError('USD/NPZ identity differs: '+e['id'])
        hashes[str(usd.relative_to(source))]=digest(usd);hashes[str(npz.relative_to(source))]=digest(npz)
    if len(hashes)!=97:raise ValueError('Expected catalog plus96 distinct geometry files')
    return entries,hashes


def validate_result(state,source):
    """Bind all declared IDs/geometry and exact recorded mesh/ray/contact verdicts."""
    source=Path(source);entries,_=catalog_identity(source)
    if (state.get('status')!='completed' or state.get('cpu_only') is not False
            or state.get('all_fixture_smokes_passed') is not True
            or state.get('robot_validation_performed') is not False
            or state.get('ready_for_terrain_training') is not False
            or state.get('simulated_seconds')!=2.5 or state.get('catalog_sha256')!=CATALOG_SHA256
            or state.get('harness_sha256')!=digest(source/'tools/validate_terrain_fixtures.py')):
        raise ValueError('Exact completed500x5ms fixture-only result required')
    cpu=state.get('cpu_rows',[]);runtime=state.get('runtime_rows',[])
    if tuple(r.get('id') for r in cpu)!=EXPECTED_IDS or tuple(r.get('id') for r in runtime)!=EXPECTED_IDS:
        raise ValueError('Missing, duplicated, reordered or extra fixture result')
    finite=lambda x:isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
    for index,(entry,c,r) in enumerate(zip(entries,cpu,runtime)):
        if (c.get('cpu_passed') is not True or c.get('physical_validation') is not False
                or c.get('collision_approximation')!='none' or c.get('meters_per_unit')!=1 or c.get('up_axis')!='Z'
                or c.get('usd_sha256')!=entry['sha256'] or c.get('npz_sha256')!=entry['npz_sha256']
                or not finite(c.get('max_vertex_error_m')) or c['max_vertex_error_m']>1e-7):
            raise ValueError('CPU mesh/schema evidence differs: '+entry['id'])
        if (r.get('passed') is not True or r.get('ray_caster_passed') is not True or r.get('ray_caster_rays')!=81
                or not finite(r.get('ray_caster_max_height_error_m')) or r['ray_caster_max_height_error_m']>=2e-4):
            raise ValueError('RayCaster evidence missing/rejected: '+entry['id'])
        queries=r.get('physx_queries',[]);probes=r.get('probes',[])
        if len(queries)!=4 or len(probes)!=3:raise ValueError('Expected4 PhysX queries and3 physical probes per fixture')
        xy=([-1.,.11],[0.,.031],[.51,-.21],[1.75,0.])
        for j,q in enumerate(queries):
            if q.get('passed') is not True or q.get('xy_local_m')!=xy[j]:raise ValueError('PhysX query missing/rejected')
            if j==3:
                if q.get('expected_height_m') is not None or q.get('height_m') is not None:raise ValueError('Outside mesh query must miss')
            elif (not finite(q.get('expected_height_m')) or not finite(q.get('height_m'))
                    or abs(q['height_m']-q['expected_height_m'])>=2e-4
                    or q.get('collision')!=f'/World/Fixtures/f_{index:03d}/Terrain'):
                raise ValueError('PhysX terrain identity/height differs')
        for p in probes:
            if (p.get('passed') is not True or not finite(p.get('sphere_bottom_gap_m')) or abs(p['sphere_bottom_gap_m'])>.003
                    or not finite(p.get('contact_fraction_last_40_steps')) or not .8<=p['contact_fraction_last_40_steps']<=1.
                    or not finite(p.get('mean_contact_force_n')) or p['mean_contact_force_n']<0
                    or len(p.get('final_position_m',[]))!=3 or not all(finite(x) for x in p['final_position_m'])
                    or not finite(p.get('surface_height_m'))):
                raise ValueError('Physical probe evidence missing/rejected')
    return {'passed':True,'fixture_ids':list(EXPECTED_IDS),'fixture_count':48,'ray_caster_rays':48*81,
            'physx_queries':48*4,'physical_probes':48*3,'harness_sha256':state['harness_sha256'],
            'catalog_sha256':CATALOG_SHA256,'robot_validation_performed':False,'ready_for_terrain_training':False}
