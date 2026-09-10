from pathlib import Path
import ast,copy,importlib.util,sys
from types import SimpleNamespace
import unittest

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'reference_physics_adapter_004/source_004/tools')]
from solver_comparison import configure_comparison,readback_comparison,PROTOCOL
from screen_contract import preflight
from launch_reference_physics_spark import command

class Physics:
    def __init__(self):self.solver_type=1;self.enable_stabilization=False;self.enable_external_forces_every_iteration=False;self.gpu_heap_capacity=16777216
    def to_dict(self):return vars(self).copy()

class Attribute:
    def __init__(self,v,valid=True):self.v=v;self.valid=valid
    def Get(self):return self.v
    def IsValid(self):return self.valid

class Prim:
    def __init__(self,values):self.values=values
    def IsValid(self):return True
    def GetAttribute(self,name):return Attribute(self.values.get(name),name in self.values)

class SolverComparisonTests(unittest.TestCase):
    def test_one_option_delta_and_preserved_parent(self):
        cfg=SimpleNamespace(sim=SimpleNamespace(physics=Physics()));original=cfg.sim.physics.to_dict()
        receipt=configure_comparison(cfg)
        self.assertEqual(receipt['physics_config_before'],original)
        self.assertEqual(receipt['physics_config_after'],dict(original,enable_external_forces_every_iteration=True))
        self.assertIs(receipt['protocol']['production_default_adopted'],False)
        with self.assertRaises(ValueError):configure_comparison(cfg)

    def test_wrong_parent_solver_or_stabilization_fail_closed(self):
        for key,value in [('solver_type',0),('enable_stabilization',True)]:
            cfg=SimpleNamespace(sim=SimpleNamespace(physics=Physics()));setattr(cfg.sim.physics,key,value)
            with self.assertRaises(ValueError):configure_comparison(cfg)
            self.assertFalse(cfg.sim.physics.enable_external_forces_every_iteration)

    def test_scene_readback_required_true_and_matching_solver(self):
        values={'physxScene:solverType':'TGS','physxScene:enableStabilization':False,'physxScene:enableExternalForcesEveryIteration':True}
        env=SimpleNamespace(sim=SimpleNamespace(stage=SimpleNamespace(GetPrimAtPath=lambda _:Prim(values))),cfg=SimpleNamespace(sim=SimpleNamespace(physics_prim_path='/physicsScene')))
        receipt=readback_comparison(env,{'protocol':PROTOCOL})
        self.assertEqual(receipt['actual_scene_attributes'],values)
        values['physxScene:enableExternalForcesEveryIteration']=False
        with self.assertRaises(RuntimeError):readback_comparison(env,{})
        del values['physxScene:enableExternalForcesEveryIteration']
        with self.assertRaises(RuntimeError):readback_comparison(env,{})

    def test_runtime_and_host_both_reject_wave(self):
        with self.assertRaises(ValueError):preflight(SimpleNamespace(mode='wave'),HERE)
        with self.assertRaises(ValueError):command(HERE,HERE/'unused','test','wave')
        output=command(HERE,HERE/'unused','test','standing')
        self.assertEqual(output[output.index('--mode')+1],'standing')
        self.assertEqual(output[output.index('--num-envs')+1],'32')
        self.assertEqual(output[output.index('--steps')+1],'1000')
        tree=ast.parse((HERE/'launch_reference_physics_spark.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='run_owned']
        self.assertEqual([n.args[1].value for n in calls],['standing'])

if __name__=='__main__':unittest.main()
