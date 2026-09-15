import copy,unittest
from types import SimpleNamespace
import solver_recipe as recipe

class Attr:
    def __init__(self,value,authored=False):self.value=value;self.authored=authored
    def Get(self):return self.value
    def HasAuthoredValueOpinion(self):return self.authored
class API:
    def __init__(self,prim):self.prim=prim
    def GetSolverPositionIterationCountAttr(self):return self.prim['position']
    def GetSolverVelocityIterationCountAttr(self):return self.prim['velocity']
    def CreateSolverPositionIterationCountAttr(self,v):self.prim['position']=Attr(v,True)
    def CreateSolverVelocityIterationCountAttr(self,v):self.prim['velocity']=Attr(v,True)
class Stage:
    def __init__(self,roots):self.data={r+'/body':{'position':Attr(32),'velocity':Attr(1)}for r in roots}
    def GetPrimAtPath(self,p):return self.data[p]

class RecipeTests(unittest.TestCase):
    def setUp(self):self.roots=['/Robot','/Robot_001'];self.stage=Stage(self.roots);self.schema=SimpleNamespace(PhysxArticulationAPI=API)
    def test_preserves_position_and_changes_only_velocity_in_each_root(self):
        r=recipe.configure(self.stage,self.roots,self.schema)
        for root in self.roots:
            self.assertEqual(r['before'][root],{'position_iterations':32,'velocity_iterations':1,'authored':[False,False]})
            self.assertEqual(r['after_authoring'][root],{'position_iterations':32,'velocity_iterations':0,'authored':[True,True]})
    def test_late_parent_mismatch_rejects_before_mutating_first_root(self):
        self.stage.data['/Robot_001/body']['position'].value=16
        with self.assertRaisesRegex(ValueError,'parent solver defaults'):recipe.configure(self.stage,self.roots,self.schema)
        self.assertEqual(recipe.read(self.stage,['/Robot'],self.schema)['/Robot']['authored'],[False,False])
    def test_already_authored_parent_is_not_claimed_as_observed_default(self):
        self.stage.data['/Robot/body']['position'].authored=True
        with self.assertRaisesRegex(ValueError,'parent solver defaults'):recipe.configure(self.stage,self.roots,self.schema)
    def test_later_change_rejects(self):
        recipe.configure(self.stage,self.roots,self.schema)
        self.stage.data['/Robot_001/body']['velocity'].value=1
        with self.assertRaisesRegex(ValueError,'recipe changed'):recipe.verify(self.stage,self.roots,self.schema)
    def test_missing_and_boolean_values_reject(self):
        for value in (None,True):
            self.stage.data['/Robot/body']['velocity'].value=value
            with self.assertRaisesRegex(ValueError,'noninteger'):recipe.read(self.stage,self.roots,self.schema)

if __name__=='__main__':unittest.main()
