import copy
import json
from pathlib import Path
import unittest
import torch
from bfcl_fixed_rollout_inputs import pack,tensor_distance


class FixedInputTests(unittest.TestCase):
    def test_padding_external_masks_and_corruption(self):
        rows=[dict(rollout_index=i,task_id='t',reward=float(i==0),prompt_ids=[10]*(i+1),
            completion_ids=[20,21]+[22]*i,model_token_mask=[1,0]+[1]*i) for i in range(4)]
        group=dict(task_ids=['t']*4,rewards=[1.,0.,0.,0.],advantages=[1.5,-.5,-.5,-.5],
                   model_tokens_in_loss=10,external_tokens_excluded=4)
        x=pack(rows,group,0)
        self.assertEqual(x['prompt_ids'][0].tolist(),[0,0,0,10])
        self.assertEqual(x['completion_mask'][0].tolist(),[1,1,0,0,0])
        self.assertEqual(x['tool_mask'][0].tolist(),[1,0,1,1,1])
        self.assertEqual((x['completion_mask']*x['tool_mask'])[0].tolist(),[1,0,0,0,0])
        broken=copy.deepcopy(rows);broken[0]['model_token_mask'][1]=1
        with self.assertRaises(ValueError):pack(broken,group,0)
        with self.assertRaises(ValueError):pack(rows[::-1],group,0)

    def test_real_archived_group_and_reward_advantages(self):
        root=Path(__file__).resolve().parents[1]/'rl/bfcl_intervention_dose_20261010/train'
        rows=[json.loads(x) for x in (root/'rollouts.jsonl').read_text().splitlines()]
        groups=[json.loads(x) for x in (root/'groups.jsonl').read_text().splitlines()]
        rows=[x for x in rows if x['policy_step']==8];group=next(x for x in groups if x['policy_step']==8)
        x=pack(rows,group,151643)
        r=torch.tensor(group['rewards'])
        self.assertTrue(torch.equal(x['advantages'],(r-r.mean())/(r.std()+1e-4)))
        for i,row in enumerate(rows):
            self.assertEqual(x['prompt_ids'][i][x['prompt_mask'][i].bool()].tolist(),row['prompt_ids'])
            self.assertEqual(x['completion_ids'][i][x['completion_mask'][i].bool()].tolist(),row['completion_ids'])
        self.assertGreater(int((x['completion_mask']-x['completion_mask']*x['tool_mask']).sum()),0)

    def test_distance_retains_scale_and_rejects_bad_values(self):
        a={'x':torch.tensor([3.,4.])};b={'x':torch.tensor([3.,5.])}
        result=tensor_distance(a,b)
        self.assertEqual(result['different_elements'],1)
        self.assertEqual(result['max_abs'],1.)
        self.assertEqual(result['relative_l2'],.2)
        self.assertTrue(tensor_distance(a,a)['exact'])
        with self.assertRaises(ValueError):tensor_distance(a,{'x':torch.tensor([float('nan'),4.])})


if __name__=='__main__':unittest.main()
