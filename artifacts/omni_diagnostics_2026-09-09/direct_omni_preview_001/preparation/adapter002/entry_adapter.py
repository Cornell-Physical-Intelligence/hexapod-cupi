"""Exact AST seams: metadata, late render-only cfg, RGB keyword and callback."""
import ast,json
from preview_contract import HERE,SOURCE,read,sha

def same(node,text):return ast.dump(node)==ast.dump(ast.parse(text).body[0])

def instrument(text):
    expected=read(HERE/'inputs/native_source_manifest.json')['tools/train_length_study.py']
    import hashlib
    if hashlib.sha256(text.encode()).hexdigest()!=expected:raise ValueError('Unpinned native entry bytes')
    tree=ast.parse(text);counts={'app':0,'save':0,'config':0,'render':0,'dispatch':0}
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self,node):
            self.generic_visit(node)
            if same(node,'app = AppLauncher(args).app'):
                counts['app']+=1;return [node,ast.copy_location(ast.parse('_preview_app_ready()').body[0],node)]
            return node
        def visit_FunctionDef(self,node):
            self.generic_visit(node)
            if node.name=='save':
                counts['save']+=1;return [node,ast.copy_location(ast.parse('save = _preview_bind_save(save)').body[0],node)]
            return node
        def visit_Expr(self,node):
            self.generic_visit(node)
            if same(node,'dump_yaml(str(args.output/"environment.yaml"),cfg)'):
                counts['config']+=1;return [ast.copy_location(ast.parse('_preview_render_config(cfg)').body[0],node),node]
            if same(node,'function(env, runner, plan, args.output, digest(args.checkpoint))'):
                counts['dispatch']+=1;return ast.copy_location(ast.parse('_preview_record(env, runner, plan, args.output, digest(args.checkpoint))').body[0],node)
            return node
        def visit_Call(self,node):
            self.generic_visit(node)
            if isinstance(node.func,ast.Name) and node.func.id=='OmniFlatEnv':
                kw=next(x for x in node.keywords if x.arg=='render_mode')
                if ast.dump(kw.value)!=ast.dump(ast.parse('"rgb_array" if args.mode=="video" else None',mode='eval').body):raise ValueError('Unknown native render keyword')
                kw.value=ast.Constant('rgb_array');counts['render']+=1
            return node
    result=ast.fix_missing_locations(Rewrite().visit(tree))
    if counts!={'app':1,'save':1,'config':1,'render':1,'dispatch':1}:raise ValueError('Unexpected render seam counts '+str(counts))
    return result,counts

def configure_rendering(cfg,output):
    from isaaclab.utils.io import dump_yaml
    from preview_contract import save
    before=cfg.to_dict()
    if cfg.scene.num_envs!=48:raise ValueError('Native matched evaluation must start at48 environments')
    dump_yaml(str(output/'matched_evaluation_environment.yaml'),cfg)
    cfg.scene.num_envs=1;cfg.video_recorder.window_width=1280;cfg.video_recorder.window_height=720
    after=cfg.to_dict()
    def flatten(obj,prefix=''):
        if isinstance(obj,dict):
            result={}
            for k,v in obj.items():result.update(flatten(v,prefix+'.'+str(k) if prefix else str(k)))
            return result
        return {prefix:json.dumps(obj,sort_keys=True,default=repr)}
    a,b=flatten(before),flatten(after);changed={k for k in set(a)|set(b) if a.get(k)!=b.get(k)}
    allowed={'scene.num_envs','video_recorder.window_width','video_recorder.window_height'}
    if not changed<=allowed or 'scene.num_envs' not in changed:raise ValueError('Unexpected non-render config change')
    save(output/'rendering_config_delta.json',{'changed':{k:{'before':a.get(k),'after':b.get(k)} for k in sorted(changed)},
         'additional_explicit_overrides':{'render_mode':'rgb_array','enable_cameras':True,'evaluation_dispatch':'38s command-only preview'},
         'all_other_configuration_fields_equal':True,'source_manifest_sha256':SOURCE})
