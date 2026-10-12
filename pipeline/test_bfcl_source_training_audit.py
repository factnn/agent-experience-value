import copy
import unittest
from report_bfcl_source_training import audit_trace


class TrainingAuditTests(unittest.TestCase):
    def fixture(self):
        job = {'source_step': 0, 'task_pool': ['a', 'b'], 'rollout_seed_base': 123, 'token_budget': 8}
        summary = {'branch_local_optimizer_steps': 1, 'stop_reason': 'token_budget_after_complete_group',
                   'nominal_token_budget': 8, 'new_sampled_tokens': 8, 'branch_sampled_tokens': 8, 'final_fingerprint': 'after'}
        allocations = [{'policy_step': 0, 'cumulative_policy_step': 0, 'seed': 123, 'task_id': 'a',
                        'rewards': [1, 0, 1, 0], 'probabilities': {'a': .5, 'b': .5},
                        'policy_fingerprint': 'before', 'sampled_tokens': 8, 'branch_cost_after_group': 8}]
        groups = [{'policy_step': 0, 'task_ids': ['a']*4, 'rewards': [1, 0, 1, 0], 'advantages': [1, -1, 1, -1]}]
        updates = [{'step': 1, 'before_sha256': 'before', 'after_sha256': 'after'}]
        micro = [{'micro_step_after': i+1, 'local_optimizer_step': 0, 'accelerate_backward_divisor': 1,
                  'grpo_loss_divisor': 4, 'loss': .1} for i in range(4)]
        return job, summary, allocations, groups, updates, micro

    def test_complete_boundary_and_corrupted_seed_reward_probability_or_scaling(self):
        fixture = self.fixture()
        self.assertEqual(audit_trace(*fixture), 8)
        for index, key, value in [(2, 'seed', 124), (2, 'probabilities', {'a': 1, 'b': 0}),
                                   (3, 'rewards', [1, 1, 1, 1]), (5, 'accelerate_backward_divisor', 4),
                                   (5, 'micro_step_after', 2)]:
            changed = copy.deepcopy(fixture)
            changed[index][0][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                audit_trace(*changed)

    def test_incomplete_budget_and_disconnected_policy_rejected(self):
        for key, value in [('new_sampled_tokens', 7), ('final_fingerprint', 'unrelated'),
                           ('stop_reason', 'soft_wall_before_budget')]:
            fixture = self.fixture()
            fixture[1][key] = value
            with self.subTest(key=key), self.assertRaises(AssertionError):
                audit_trace(*fixture)


if __name__ == '__main__':
    unittest.main()
