"""Exercise independent limits through the actual shared generate backend."""
import unittest
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from bfcl_token_rollout import ROOT,TokenEpisode,development_tasks
from bfcl_sampling import sample_episodes


class FakeModel(torch.nn.Module):
    def __init__(self,answer):
        super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(1));self.answer=answer;self.allowances=[]
    def generate(self,input_ids,attention_mask,generation_config,**kwargs):
        cap=generation_config.max_new_tokens;self.allowances.append(cap)
        assert input_ids.shape[0]==1
        tokens=[1]*64 if cap==64 else self.answer
        assert len(tokens)<=cap
        return torch.cat([input_ids,torch.tensor([tokens],device=input_ids.device)],dim=1)


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=AutoTokenizer.from_pretrained(ROOT/'smoke/model',local_files_only=True)
        cls.tasks,cls.answers=development_tasks(['multi_turn_base_26'])
    def episodes(self):
        return [TokenEpisode(self.tokenizer,self.tasks['multi_turn_base_26'],self.answers['multi_turn_base_26']) for _ in range(2)]
    def test_generate_allowance_not_reduced_by_peer(self):
        a,b=self.episodes()
        for e,used in [(a,4032),(b,1096)]:e.completion_ids=[1]*used;e.mask=[1]*used
        answer=self.tokenizer.encode('word '*90,add_special_tokens=False)+[self.tokenizer.eos_token_id]
        model=FakeModel(answer);calls=[]
        stats=sample_episodes(model,self.tokenizer,[a,b],SimpleNamespace(max_new_tokens=0),calls.append)
        self.assertEqual(model.allowances[:2],[64,1024])
        self.assertEqual(a.stop_reason,'generation_cap');self.assertEqual(b.stop_reason,'completed')
        self.assertEqual(calls[1]['rollout_index'],1);self.assertGreater(len(calls[1]['generated_ids']),64)
        self.assertEqual(stats['sampled_tokens'],sum(len(c['generated_ids']) for c in calls))
    def test_global_stop_is_separate_censoring(self):
        episodes=self.episodes();model=FakeModel([self.tokenizer.eos_token_id]);calls=[]
        stats=sample_episodes(model,self.tokenizer,episodes,SimpleNamespace(max_new_tokens=0),calls.append,
            should_stop=lambda:'global_wall_limit')
        self.assertTrue(stats['censored']);self.assertFalse(model.allowances);self.assertFalse(calls)
        self.assertTrue(all(e.stop_reason=='global_wall_limit' for e in episodes))


if __name__=='__main__':unittest.main(verbosity=2)
