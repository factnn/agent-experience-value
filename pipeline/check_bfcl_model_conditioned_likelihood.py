"""CPU tiny-Qwen/PEFT numerical checks; no real learner/reference likelihoods."""
import json

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from bfcl_model_conditioned_likelihood import feedback_summary, reference_summary, token_logprobs
from rl_learning_state import tensor_hash


def rejects(function):
    try:
        function()
    except ValueError:
        return
    raise AssertionError('Invalid measurement input accepted')


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261017)
    config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, attention_dropout=0., pad_token_id=0, eos_token_id=2)
    model = get_peft_model(Qwen3ForCausalLM(config),
        LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM')).eval()
    before = tensor_hash(model.named_parameters())
    ids = [1, 12, 7, 5, 31, 9, 20, 3, 25, 8, 2, 4]
    positions = [2, 6, 9, 11]
    selected = token_logprobs(model, ids, positions)
    with torch.inference_mode():
        x = torch.tensor([ids], dtype=torch.long)
        all_logits = model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits
        expected = torch.log_softmax(all_logits[0, torch.tensor(positions) - 1].float(), dim=-1).gather(
            1, x[0, torch.tensor(positions), None]).squeeze(1)
    assert torch.allclose(selected, expected, atol=1e-6, rtol=0)
    changed = ids.copy()
    changed[5] = 13
    ablated = token_logprobs(model, changed, positions)
    summary = feedback_summary(selected, ablated, [1, 2, 3])
    assert summary['prefix_max_abs_logprob_difference'] <= 1e-5
    assert feedback_summary([-1., -2., -3.], [-1., -4., -8.], [1, 2])['value'] == 3.5
    assert feedback_summary([-1., -4., -8.], [-1., -2., -3.], [1, 2])['value'] == -3.5
    assert feedback_summary([-1.], [-1.], [])['value'] == 0.
    assert reference_summary([-1., -2., -3.])['value'] == -2.
    rejects(lambda: feedback_summary([-1., -2.], [-1.1, -3.], [1]))
    rejects(lambda: feedback_summary([-1., -2., -3.], [-1., -2., -3.], [0, 2]))
    rejects(lambda: reference_summary([]))
    rejects(lambda: reference_summary([float('nan')]))
    rejects(lambda: token_logprobs(model, ids, [0]))
    assert token_logprobs(model, ids, []).numel() == 0
    assert all(p.grad is None for p in model.parameters())
    assert tensor_hash(model.named_parameters()) == before
    print(json.dumps({'status': 'passed', 'checks': ['PEFT forwards selected-logit tensor',
        'causal target shift equals full logits', 'future feedback leaves earlier logP unchanged',
        'positive/negative/empty-feedback aggregation', 'fixed-reference token averaging',
        'invalid index/prefix/nonfinite/empty-reference rejected', 'parameters unchanged and no gradients'],
        'fixture': 'Tiny random CPU Qwen3(1 layer,hidden16,vocab64)+LoRA2; no actual4B/8B source-model likelihoods',
        'native_model_likelihoods_computed': False, 'gpu_jobs': 0, 'new_model_tokens': 0}))


if __name__ == '__main__':
    main()
