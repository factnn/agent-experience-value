"""Check official GRPO loss excludes external feedback and follows reward signs."""
from collections import defaultdict
from types import SimpleNamespace
import unittest
import torch
from train_rl_smoke import GRPOTrainer  # same explicitly documented import compatibility


class GRPOLossTest(unittest.TestCase):
    def compute(self, feedback_logp=-2.0, advantages=(1.0, -1.0)):
        logps = torch.tensor([[-1., feedback_logp, -3.], [-1., feedback_logp, -3.]], requires_grad=True)
        stub = SimpleNamespace(
            _get_per_token_logps_and_entropies=lambda *a, **k: (logps, torch.zeros_like(logps)),
            top_entropy_quantile=1.0, off_policy_mask_threshold=None,
            importance_sampling_level='token', beta=0.0, loss_type='grpo',
            epsilon_low=0.2, epsilon_high=0.2, args=SimpleNamespace(delta=None),
            use_vllm=False, model=SimpleNamespace(training=True),
            current_gradient_accumulation_steps=1,
            accelerator=SimpleNamespace(gather=lambda x: x),
            _metrics={'train': defaultdict(list)},
        )
        inputs = {'prompt_ids': torch.ones((2, 1), dtype=torch.long),
                  'prompt_mask': torch.ones((2, 1), dtype=torch.long),
                  'completion_ids': torch.ones((2, 3), dtype=torch.long),
                  'completion_mask': torch.tensor([[1, 1, 1], [1, 1, 0]]),
                  'tool_mask': torch.tensor([[1, 0, 1], [1, 0, 1]]),
                  'advantages': torch.tensor(advantages)}
        loss = GRPOTrainer._compute_loss(stub, stub.model, inputs)
        loss.backward()
        return loss.detach(), logps.grad

    def test_feedback_and_padding_have_zero_gradient(self):
        _, grad = self.compute()
        self.assertEqual(grad[:, 1].tolist(), [0., 0.])
        self.assertEqual(grad[1, 2].item(), 0.)
        self.assertLess(grad[0, 0].item(), 0.)
        self.assertGreater(grad[1, 0].item(), 0.)

    def test_feedback_logprob_cannot_change_loss_or_gradient(self):
        loss_a, grad_a = self.compute(-2.)
        loss_b, grad_b = self.compute(-100.)
        torch.testing.assert_close(loss_a, loss_b)
        torch.testing.assert_close(grad_a, grad_b)

    def test_zero_advantage_produces_no_learning_signal(self):
        _, grad = self.compute(advantages=(0., 0.))
        self.assertEqual(grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
