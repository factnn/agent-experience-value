"""Check pre-rollout routing, cost collection, and evaluation isolation."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from train_rl_allocation import AllocationTrainer, EvidenceTrainer
from rl_allocation import TaskAllocator


class RoutingTests(unittest.TestCase):
    def make_trainer(self, path):
        t=object.__new__(AllocationTrainer)
        t.model=SimpleNamespace(training=True);t.state=SimpleNamespace(global_step=0)
        t.training_rows={'train:a':{'task_id':'train:a','prompt':[]}}
        t.allocator=TaskAllocator(['train:a']);t.allocation_records=[]
        t.evidence_phase='train:0';t.sampled_tokens=0;t.token_budget=100
        t.pilot_seed=20261008;t.current_fingerprint='policy0';t.evidence_dir=Path(path)
        t.environments=[SimpleNamespace(_reward=lambda:0) for _ in range(4)]
        return t

    def test_pre_rollout_choice_and_all_failed_cost(self):
        with tempfile.TemporaryDirectory() as path:
            t=self.make_trainer(path)
            def generate(trainer, inputs):
                self.assertEqual([r['task_id'] for r in inputs],['train:a']*4)
                self.assertEqual(trainer.allocator.pending,'train:a')
                trainer.sampled_tokens+=37
                return {'loss_inputs':'official'}
            with patch.object(EvidenceTrainer,'_generate_and_score_completions',generate):
                result=t._generate_and_score_completions([{'task_id':'placeholder'}]*4)
            self.assertEqual(result,{'loss_inputs':'official'})
            self.assertEqual(t.allocator.total_sampled_tokens,37)
            self.assertEqual(t.allocation_records[0]['rewards'],[0]*4)
            with self.assertRaises(AssertionError):t._generate_and_score_completions([{}]*4)

    def test_evaluation_does_not_enter_allocator(self):
        with tempfile.TemporaryDirectory() as path:
            t=self.make_trainer(path);t.evidence_phase='eval:0';t.model.training=False
            inputs=[{'task_id':'eval:z'}]*4
            with patch.object(EvidenceTrainer,'_generate_and_score_completions',return_value='official') as call:
                self.assertEqual(t._generate_and_score_completions(inputs),'official')
                call.assert_called_once_with(inputs)
            self.assertEqual(t.allocator.selection_count,0)
            self.assertFalse(t.allocation_records)

    def test_no_new_group_after_budget(self):
        with tempfile.TemporaryDirectory() as path:
            t=self.make_trainer(path);t.allocator.total_sampled_tokens=100
            with self.assertRaises(AssertionError):t._generate_and_score_completions([{}]*4)
            self.assertEqual(t.allocator.selection_count,0)


if __name__=='__main__':unittest.main()
