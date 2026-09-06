"""Synthetic native API checks: exact62-source mapping,1922pairs and UInt32 safety."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
import torch
import contact_audit as audit

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'configs/mkii_fourbar_v3_kinematics.json').is_file())
KIN = json.loads((ROOT/'configs/mkii_fourbar_v3_kinematics.json').read_text())


def native_view():
    force = torch.zeros(1,31,3)
    count, start = torch.zeros(1,31,dtype=torch.uint32), torch.zeros(1,31,dtype=torch.uint32)
    data = [torch.zeros(audit.CAPACITY,1),torch.zeros(audit.CAPACITY,3),torch.zeros(audit.CAPACITY,3),
            torch.zeros(audit.CAPACITY,1),count,start]
    view = SimpleNamespace(sensor_count=1,filter_count=31,max_contact_data_count=audit.CAPACITY,
                           check=lambda:True,get_contact_force_matrix=lambda **_:force,get_contact_data=lambda **_:data)
    return view, force, data


def metrics():
    return {'samples':256,'finite':True,'max_pair_contact_count':0,'max_pair_force_n':0.,
        'contact_buffer_capacity_reached':False,'ground_support_observed_per_robot':[True,True],
        'source_body_excitation':[{'contact_count_observed':False,'force_above_1mN_observed':False} for _ in range(62)]}


class FullBodyContactAuditTests(unittest.TestCase):
    def test_all31_exact_source_bodies_against31_foreign_targets_in_both_directions(self):
        bindings = audit.expected_bindings(KIN['body_paths'],['/World/envs/env_0','/World/envs/env_1'])
        self.assertEqual(len(bindings),62)
        self.assertEqual(len({row['source_body'] for row in bindings}),62)
        pairs = {(row['source_body'],target) for row in bindings for target in row['target_filters']}
        self.assertEqual(len(pairs),1922)
        self.assertEqual(pairs,{(target,source) for source,target in pairs})
        for row in bindings:
            self.assertEqual(row['target_names'],list(KIN['body_paths']))
            self.assertTrue(all(f"/env_{1-row['source_environment']}/" in path for path in row['target_filters']))
        for mapping in (dict(list(KIN['body_paths'].items())[:-1]),dict(KIN['body_paths'],body='../body')):
            with self.assertRaises(ValueError): audit.expected_bindings(mapping,['/env0','/env1'])

    def test_uint32_conversion_is_lossless_before_unsupported_comparisons(self):
        native = torch.tensor([0,2**31,2**32-1],dtype=torch.uint32)
        result = audit.native_integer(native)
        self.assertEqual(result.dtype,torch.int64)
        self.assertEqual(result.tolist(),[0,2**31,2**32-1])
        for value in (torch.tensor([-1]),torch.tensor([1.]),torch.tensor([True]),torch.tensor([2**63],dtype=torch.uint64)):
            with self.assertRaises(ValueError): audit.native_integer(value)

    def test_every_target_count_and_force_retained_with_dtype_and_buffer_clones(self):
        view, force, data = native_view()
        data[4][0,30] = 2; data[5][0,30] = 17; force[0,30,2] = 8.
        result_force,count,start,capacity,dtypes = audit.read_pair_sample(view,lambda value:value,.00125)
        force.zero_(); data[4].zero_()
        self.assertEqual(result_force.shape,(31,3));self.assertEqual(count.shape,(31,))
        self.assertEqual(result_force[30,2].item(),8.);self.assertEqual(count[30].item(),2)
        self.assertEqual(start[30].item(),17);self.assertFalse(capacity)
        self.assertEqual(dtypes,{'counts':'torch.uint32','start_indices':'torch.uint32'})

    def test_capacity_is_shared_across_filters_and_loaded_ranges_must_not_overlap(self):
        view,force,data = native_view()
        data[4][0,0] = 1024;data[4][0,1] = 1024;data[5][0,1] = 1024
        _,_,_,full,_ = audit.read_pair_sample(view,lambda value:value,.00125)
        self.assertTrue(full)  # Neither individual count reaches2048.
        data[5][0,1] = 1000
        with self.assertRaises(ValueError):audit.read_pair_sample(view,lambda value:value,.00125)
        data[5][0,1] = 2047
        with self.assertRaises(ValueError):audit.read_pair_sample(view,lambda value:value,.00125)

    def test_only_loaded_patch_data_must_be_finite_but_all_matrix_forces_must(self):
        view,force,data = native_view();data[1].fill_(float('nan'))
        audit.read_pair_sample(view,lambda value:value,.00125)  # Unloaded storage has no valid meaning.
        data[4][0,4] = 1;data[5][0,4] = 3
        with self.assertRaises(ValueError):audit.read_pair_sample(view,lambda value:value,.00125)
        data[1].zero_();force[0,30,0] = float('nan')
        with self.assertRaises(ValueError):audit.read_pair_sample(view,lambda value:value,.00125)

    def test_unsupported_or_incomplete_native_mapping_is_rejected(self):
        for key,value in (('sensor_count',2),('filter_count',30),('max_contact_data_count',512),('check',lambda:False)):
            view,_,_=native_view();setattr(view,key,value)
            with self.assertRaises(ValueError):audit.read_pair_sample(view,lambda value:value,.00125)

    def test_filtered_and_negative_controls_have_distinct_coverage_obligations(self):
        value=metrics();self.assertEqual(audit.grade('filtered',value,256),[])
        value['max_pair_contact_count']=1
        self.assertTrue(audit.grade('filtered',value,256))
        value['max_pair_force_n']=8.;value['foreign_contact_observed_per_direction']=[True,True]
        self.assertEqual(audit.grade('unfiltered_negative',value,256),[])
        # Incomplete excitation remains an explicit limitation, not a fabricated universal proof.
        self.assertFalse(all(row['contact_count_observed'] for row in value['source_body_excitation']))
        value['foreign_contact_observed_per_direction']=[True,False]
        self.assertTrue(audit.grade('unfiltered_negative',value,256))
        value['source_body_excitation'].pop()
        self.assertTrue(audit.grade('unfiltered_negative',value,256))

    def test_live_loop_uses_all_views_every_physics_step_and_no_postconstruction_origin_mutation(self):
        tree=ast.parse(Path(__file__).with_name('live_full_body_overlap.py').read_text())
        loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='step')
        self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='read_pair_sample' for n in ast.walk(loop)))
        self.assertFalse(any(isinstance(n,ast.Assign) and any(isinstance(target,ast.Subscript) and ast.unparse(target).startswith('origins[') for target in n.targets) for n in ast.walk(tree)))
        self.assertIn('os.environ["HEXAPOD_MKII_ENVIRONMENT_LAYOUT"] = LAYOUT',ast.unparse(tree).replace("'",'"'))


if __name__=='__main__':unittest.main()
