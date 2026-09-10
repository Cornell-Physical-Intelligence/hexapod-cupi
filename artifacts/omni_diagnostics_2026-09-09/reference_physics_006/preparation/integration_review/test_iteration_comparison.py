from pathlib import Path
import sys,unittest
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'reference_physics_adapter_005/source_005/tools')]
from solver_comparison import configure_iteration_comparison,validate_iteration_readback,iteration_scene_readback,PROTOCOL
from screen_contract import preflight


def cfg():return SimpleNamespace(sim=SimpleNamespace(physics=SimpleNamespace(enable_external_forces_every_iteration=True)),robot=SimpleNamespace(spawn=SimpleNamespace(articulation_props=SimpleNamespace(solver_position_iteration_count=16,solver_velocity_iteration_count=4))))
def rows():return [dict(path=f'/World/envs/env_{i}/Robot/body',kind='articulation',position=16,velocity=1) for i in range(32)]+[dict(path=f'/World/envs/env_{i}/Robot/link{j}',kind='body',position=4,velocity=1) for i in range(32) for j in range(19)]

class IterationTests(unittest.TestCase):
    def test_only_articulation_velocity_changes_and_true_flag_preserved(self):
        c=cfg();before=vars(c.robot.spawn.articulation_props).copy();r=configure_iteration_comparison(c)
        self.assertEqual(vars(c.robot.spawn.articulation_props),dict(before,solver_velocity_iteration_count=1))
        self.assertTrue(c.sim.physics.enable_external_forces_every_iteration)
        self.assertEqual(r,{'baseline005':{'position':16,'velocity':4},'comparison006':{'position':16,'velocity':1}})
        self.assertEqual(PROTOCOL['baseline_value'],4);self.assertEqual(PROTOCOL['comparison_value'],1)

    def test_wrong_baseline_flag_or_iterations_rejected(self):
        for change in ['flag','position','velocity']:
            c=cfg()
            if change=='flag':c.sim.physics.enable_external_forces_every_iteration=False
            elif change=='position':c.robot.spawn.articulation_props.solver_position_iteration_count=8
            else:c.robot.spawn.articulation_props.solver_velocity_iteration_count=2
            with self.assertRaises(ValueError):configure_iteration_comparison(c)

    def test_every_actual_root_and_body_readback_required(self):
        validate_iteration_readback(rows(),32)
        for problem in ['missing_body','root_wrong','body_override','body_unknown']:
            r=rows()
            if problem=='missing_body':r.pop()
            elif problem=='root_wrong':r[2]['velocity']=4
            elif problem=='body_override':r[-1]['velocity']=4
            else:r[-1]['position']=None
            with self.assertRaises(RuntimeError):validate_iteration_readback(r,32)

    def test_failed_traversal_preserves_readback_rows(self):
        env=SimpleNamespace(num_envs=32,sim=SimpleNamespace(stage=SimpleNamespace(Traverse=lambda:[])))
        with self.assertRaises(RuntimeError) as caught:iteration_scene_readback(env)
        self.assertEqual(caught.exception.iteration_readback['rows'],[])
        self.assertEqual(caught.exception.iteration_readback['articulation_count'],0)

    def test_runtime_remains_standing_only(self):
        with self.assertRaises(ValueError):preflight(SimpleNamespace(mode='wave'),HERE)
        self.assertTrue(PROTOCOL['standing_only']);self.assertFalse(PROTOCOL['automatic_wave_or_PPO'])

if __name__=='__main__':unittest.main()
