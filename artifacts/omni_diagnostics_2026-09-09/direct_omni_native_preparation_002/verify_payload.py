"""Portable immutable preparation check; source maps are references, not runtime execution."""
from pathlib import Path
import ast,hashlib,json,sys
sys.dont_write_bytecode=True
SOURCE='64144b2c2466f17c809554ef53f82b736b4199136cc821451d7ac2788752bd2e'
OLD_SOURCE='37b4b6d5d75f6d697a122fc2a1903f8f625f6a67d1727e1257e1533300e44da6'
PLAN='9fb7d0b01480523972b4f1c17e717756d7dcf1c54b9caa436245671937de538c'
OWNERS={'native003':('20fbcd40a869b39d8e3143a6c9715ec62ada5cc6211eaf473f60f8b7167678cb',26),
        'host002':('19545653a00e635cc14ed9415f9799978da34674e84f6cad76eb42bd2a961ac4',10),
        'guard003':('123c34a1830be1051fe17f35e728bfcf795b0d497fe5d4b043dd78e484906128',16),
        'independent_guard_review':('f544018d92b7a0b1a2b1a294ceec52a78dfb1445f5ff6f141f2652f59431acbd',2)}

def need(value,message):
    if not value:raise ValueError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def inventory(root):return {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
def freeze(root,bound,count):
    p=root/'FREEZE_SHA256.json';need(sha(p)==bound,'Original manifest changed')
    expected=read(p);actual=inventory(root);actual.pop('FREEZE_SHA256.json')
    need(len(expected)==count and actual==expected,'Original payload inventory changed')

def evidence(root):
    for name,(bound,count) in OWNERS.items():freeze(root/name,bound,count)
    new=root/'inputs/source002_sha256.json';old=root/'inputs/source001_sha256.json'
    need(sha(new)==SOURCE and sha(old)==OLD_SOURCE,'Source map identity changed')
    a,b=read(old),read(new);need(set(a)==set(b) and len(b)==598,'Source inventory changed')
    changed={k:{'source001':a[k],'source002':b[k]} for k in b if a[k]!=b[k]}
    need(set(changed)=={'source_origin.json','tools/direct_training.py'},'Unexpected native source delta')
    delta=read(root/'SOURCE_DELTA.json');need(delta['inventory_equal'] is True and delta['files']==598 and delta['changed_entries']==changed,'Source delta report differs')
    build=read(root/'root_checks/source_build002.json')
    need(build=={'source_manifest_sha256':SOURCE,'files':598,'plan_sha256':PLAN,'origin_sha256':b['source_origin.json']},'Build receipt differs')
    need(b['robot/hexapod_mkii_length_study/training_plan.json']==PLAN,'Plan changed')
    before=root/'inputs/native002_direct_training.py';after=root/'native003/direct_training.py'
    need(sha(before)==a['tools/direct_training.py'] and sha(after)==b['tools/direct_training.py'],'Native before/after files unbound')
    def remove_reload(p):
        tree=ast.parse(p.read_text());found=0
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name=='verify_reload':node.body=[ast.Pass()];found+=1
        need(found==1,'Native reload site changed')
        return ast.dump(tree)
    need(remove_reload(before)==remove_reload(after),'Native behavior changed outside verify_reload')
    parentmap=root/'inputs/host001_sha256.json'
    need(sha(parentmap)=='332fad29e0a0f88ba07012700f2a21c3a4f0453f3d71224da84fcc477e48394f','Host parent manifest changed')
    host_old=root/'inputs/host001_launch_train_spark.py';host_new=root/'host002/launch_train_spark.py'
    need(sha(host_old)==read(parentmap)['launch_train_spark.py'],'Host original runtime unbound')
    for name in ('run_train_entry.py','LEGACY_RUNTIME_SHA256.json'):
        need(sha(root/'host002'/name)==read(parentmap)[name],'Host entry or legacy16 changed')
    def host_normalized(p):
        tree=ast.parse(p.read_text());found=set()
        for node in tree.body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('SOURCE_MAP','CONTRACT_FREEZE'):
                found.add(node.targets[0].id);node.value=ast.Constant('PIN')
        need(found=={'SOURCE_MAP','CONTRACT_FREEZE'},'Host pin sites differ')
        return ast.dump(tree)
    need(host_normalized(host_old)==host_normalized(host_new),'Host behavior changed outside pins')
    preflight=read(root/'root_checks/host_preflight002.json');identity=preflight['identity']
    need(preflight['no_GPU'] is True and preflight['host_freeze_sha256']==OWNERS['host002'][0],'Wrong preflight host')
    need(identity['source_manifest_sha256']==SOURCE and identity['plan_sha256']==PLAN,'Wrong preflight source/plan')
    need(identity['checkpoint_sha256']=='1971b782327408f8446b3271cc665dad8c9c515523cf8c87303141dc487d06e8','Original warm start changed')
    need(identity['selection']['allocation']=='smoke' and identity['selection']['branch']=='caps' and identity['selection']['updates']==2 and identity['selection']['replicas']==32,'Wrong smoke allocation')
    review=read(root/'independent_guard_review/review.json')
    need(review['guard_freeze_sha256']==OWNERS['guard003'][0] and review['tests_passed']==29,'Independent guard receipt differs')
    need(review['bindings']['SOURCE_SHA256']==SOURCE and review['bindings']['HOST_FREEZE_SHA256']==OWNERS['host002'][0] and review['bindings']['CONTRACT_SHA256']==OWNERS['native003'][0],'Guard source/host/contract mismatch')
    for name,count in [('native_root_tests002.log',16),('guard_root_tests003.log',29)]:
        text=(root/'root_checks'/name).read_text();need(('Ran '+str(count)+' tests') in text and text.rstrip().endswith('OK'),'Root test receipt differs')
    return {'preparation_only':True,'original_frozen_payloads':54,'source_files':598,
        'native_runtime_changed_functions':['verify_reload'],'host_changed_constants':['SOURCE_MAP','CONTRACT_FREEZE'],
        'source_manifest_sha256':SOURCE,'plan_sha256':PLAN,'root_native_tests':16,'root_guard_tests':29,
        'test_counts_are_separate_executions_not_added':True,'GPU_launched_by_verifier':False,'smoke_result_claimed':False,'Stage2_complete':False}

def verify(root):
    need(not any(p.is_symlink() for p in root.rglob('*')),'Symlink payload')
    expected=read(root/'BUNDLE_SHA256.json');actual=inventory(root);actual.pop('BUNDLE_SHA256.json')
    need(actual==expected,'Missing, changed or unlisted bundle payload')
    return {'passed':True,'payloads':len(actual),'bundle_sha256':sha(root/'BUNDLE_SHA256.json'),**evidence(root)}

if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parent),indent=2))
