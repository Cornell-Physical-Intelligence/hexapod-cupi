"""Host-safe diagnostics evidence gates, including missing/altered sparse rows."""
import copy,math,unittest
from direct_contract import validate_optimizer_diagnostics

def diagnostic_receipt(updates,replicas):
 result={'optimizer_diagnostics_schema':'direct315_actor_gradients_v1','optimizer_updates':[]}
 for u in range(1,updates+1):
  rows=[]
  for m in range(1,21):
   g=None
   if u in (1,10,25,50) and m in (1,20):g={'norms':{'ppo_actor':.2,'quiet_temporal':.3,'moving_temporal':.1,'spatial':.01},'ppo_quiet_cosine':-.1,'component_sum_norm':.4,'combined_actor_before_clip':.4,'combined_actor_after_clip':.4}
   rows.append({'update':u,'minibatch':m,'kl_mean':.01,'learning_rate_before':1e-5,'learning_rate_after':1e-5,'gradient':g,'pair_counts':{'valid_pairs':replicas*6,'quiet_pairs':replicas*3,'moving_pairs':replicas*3}})
  result['optimizer_updates'].append({'completed_update':u,'learning_rate':1e-5,'minibatches':rows})
 return result

class Contract(unittest.TestCase):
 def test_exact_complete_cadence_and_sparse_coverage(self):
  self.assertEqual(validate_optimizer_diagnostics(diagnostic_receipt(50,1024),{'updates':50,'replicas':1024,'branch':'quiet_priority'})['sparse_actor_gradient_rows'],8)
 def test_missing_reordered_nonfinite_or_unbound_diagnostics_rejected(self):
  base=diagnostic_receipt(2,32);sel={'updates':2,'replicas':32,'branch':'quiet_priority'}
  variants=[]
  v=copy.deepcopy(base);v['optimizer_updates'][0]['minibatches'].pop();variants.append(v)
  v=copy.deepcopy(base);v['optimizer_updates'][0]['minibatches'].reverse();variants.append(v)
  v=copy.deepcopy(base);v['optimizer_updates'][1]['minibatches'][8]['kl_mean']=float('nan');variants.append(v)
  v=copy.deepcopy(base);v['optimizer_updates'][0]['minibatches'][19]['gradient']=None;variants.append(v)
  v=copy.deepcopy(base);v['optimizer_updates'][1]['minibatches'][4]['pair_counts']['quiet_pairs']=1;variants.append(v)
  v=copy.deepcopy(base);v['optimizer_updates'][1]['learning_rate']=2e-5;variants.append(v)
  v=copy.deepcopy(base);v['optimizer_updates'][0]['minibatches'][0]['gradient']['combined_actor_after_clip']=float('inf');variants.append(v)
  for v in variants:
   with self.assertRaises(ValueError):validate_optimizer_diagnostics(v,sel)
 def test_no_quiet_pairs_zero_gradient_cosine_is_null_not_nan(self):
  v=diagnostic_receipt(2,32);g=v['optimizer_updates'][0]['minibatches'][0]['gradient'];g['norms']['quiet_temporal']=0.;g['ppo_quiet_cosine']=None
  self.assertEqual(validate_optimizer_diagnostics(v,{'updates':2,'replicas':32,'branch':'quiet_priority'})['updates'],2)

if __name__=='__main__':unittest.main()
