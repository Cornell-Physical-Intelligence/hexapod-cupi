from parity import *
import unittest
class CandidateTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.session,cls.geometry,cls.poses,cls.mapping=load()
  with(ACTUAL/'contacts.jsonl').open()as stream:cls.row=next(json.loads(s)for s in stream if len(json.loads(s)['patches'])>0)
 def test_preserves_all_nonclassifier_functions_and_Geometry_AST(self):
  import ast
  a=ast.parse((SOURCE/'standing_math.py').read_text());b=ast.parse((HERE/'standing_math_candidate.py').read_text())
  aa={x.name:ast.dump(x)for x in a.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))and x.name!='classify_contacts'}
  bb={x.name:ast.dump(x)for x in b.body if isinstance(x,(ast.FunctionDef,ast.ClassDef))and x.name!='classify_contacts'}
  self.assertEqual(aa,bb)
 def fixture(self,n=2,dtype=np.float32):
  body_names=[x[1]for x in self.mapping];f=np.zeros((1024*n,1),dtype);point=np.zeros((1024*n,3),dtype);normal=point.copy();sep=f.copy();counts=np.zeros((19*n,1),np.uint32);starts=counts.copy();poses=np.repeat(self.poses[self.row['sequence']],n,axis=0);maps=[]
  for e in range(n):
   for i,(_,body)in enumerate(self.mapping):
    maps.append((e,body));k=(e*19+i)*4;counts[e*19+i]=4;starts[e*19+i]=k
    f[k:k+4,0]=[.6,.6,-.2,0];normal[k:k+3]=[0.,0.,1.];normal[k+3]=[0,0,0]
    if body.endswith('_tibia'):
     shape=self.geometry.shapes[body];T=np.asarray(shape['shape_to_link']);p=poses[e,self.geometry.names.index(body)];R=old.rotation(p[3:])
     for j,x in enumerate([.09,.13,.115, .13]):point[k+j]=p[:3]+R@(T[:3,:3]@np.array([x,0.,0.])+T[:3,3])
  return[f,point,normal,sep,counts,starts],maps,poses
 def test_mixed_bodies_shaft_toe_zeros_order_and_dtypes(self):
  for n in (1,2,32):
   for dtype in (np.float32,np.float64):
    d,m,p=self.fixture(n,dtype);a=old.classify_contacts(d,m,p,self.geometry,n);b=new.classify_contacts(d,m,p,self.geometry,n);equal(a,b)
    self.assertEqual(len(a['patches']),19*n*4)
    self.assertEqual({x['category']for x in a['patches']},{'body','coxa','femur','toe','shaft'})
 def test_adversarial_error_class_and_message(self):
  def negative_count(d,m,p):d[4]=d[4].astype(np.int64);d[4][0,0]=-1
  def negative_start(d,m,p):d[5]=d[5].astype(np.int64);d[5][0,0]=-1
  def overlap(d,m,p):d[5][1,0]=0
  def overflow(d,m,p):d[5][0,0]=len(d[0])-1
  def nonfinite_force(d,m,p):d[0][0,0]=np.nan
  def nonfinite_point(d,m,p):d[1][0,0]=np.inf
  def nonfinite_normal(d,m,p):d[2][0,0]=np.nan
  def nonfinite_sep(d,m,p):d[3][0,0]=np.inf
  def invalid_normal(d,m,p):d[2][0]=[0,0,0]
  def invalid_zero_tuple(d,m,p):d[3][3]=1e-9
  def exhausted(d,m,p):
   for i in range(4):d[i]=d[i][:19*4]
  def wrong_shape(d,m,p):d[4]=d[4].reshape(-1)
  for mutate in [negative_count,negative_start,overlap,overflow,nonfinite_force,nonfinite_point,nonfinite_normal,nonfinite_sep,invalid_normal,invalid_zero_tuple,exhausted,wrong_shape]:
   d,m,p=self.fixture(1);mutate(d,m,p);errors=[]
   for function in (old.classify_contacts,new.classify_contacts):
    try:function(d,m,p,self.geometry,1)
    except Exception as e:errors.append((type(e).__name__,str(e)))
    else:self.fail(mutate.__name__+' unexpectedly accepted')
   self.assertEqual(errors[0],errors[1],mutate.__name__)
 def test_cancelled_vector_still_keeps_individual_nonfoot_violation(self):
  d,m,p=self.fixture(1);d[0][:4,0]=[2.,-2.,0,0]
  a=old.classify_contacts(d,m,p,self.geometry,1);b=new.classify_contacts(d,m,p,self.geometry,1);equal(a,b);self.assertTrue(a['nonfoot_contact'][0])
 def test_body_aggregation_does_not_merge_different_bodies(self):
  d,m,p=self.fixture(1);d[4][:]=0;d[0][:]=0;d[4][0]=1;d[5][0]=0;d[4][1]=1;d[5][1]=4;d[0][0]=.6;d[0][4]=.6
  a=old.classify_contacts(d,m,p,self.geometry,1);b=new.classify_contacts(d,m,p,self.geometry,1);equal(a,b);self.assertFalse(a['nonfoot_contact'][0])
 def test_empty_same_bytes(self):
  d,m,p=self.fixture(1);d[4][:]=0;equal(old.classify_contacts(d,m,p,self.geometry,1),new.classify_contacts(d,m,p,self.geometry,1))
 def test_nonfoot_subthreshold_accumulation_not_vector_cancellation(self):
  d,m,p=self.fixture(1)
  # Body records are individually subthreshold, but their ordered sum exceeds1N.
  d[0][:4,0]=[.6,.6,0,0]
  equal(old.classify_contacts(d,m,p,self.geometry,1),new.classify_contacts(d,m,p,self.geometry,1))
  self.assertTrue(new.classify_contacts(d,m,p,self.geometry,1)['nonfoot_contact'][0])
if __name__=='__main__':unittest.main()
