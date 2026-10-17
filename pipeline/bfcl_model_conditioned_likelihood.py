"""Read-only selected-position likelihoods and E368 scalar aggregations."""
import math

import torch


@torch.inference_mode()
def token_logprobs(model, input_ids, target_positions, chunk_size=256):
    """Return CPU float32 logP of IDs at target_positions, using causal shift."""
    if model.training:
        raise ValueError('Likelihood measurements require model.eval()')
    if chunk_size != 256:
        raise ValueError('Use the preregistered position chunk size256')
    if target_positions != sorted(set(target_positions)):
        raise ValueError('Target positions must be unique and ordered')
    if any(type(i) is not int or i < 1 or i >= len(input_ids) for i in target_positions):
        raise ValueError('Invalid causal target position')
    if not target_positions:
        return torch.empty(0, dtype=torch.float32)
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    positions = torch.tensor(target_positions, dtype=torch.long, device=device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=positions - 1)
    logits = output.logits
    if tuple(logits.shape[:2]) != (1, len(target_positions)):
        raise ValueError('Model did not return the requested shifted logit positions')
    labels = ids[0, positions]
    pieces = []
    for start in range(0, len(target_positions), chunk_size):
        stop = min(start + chunk_size, len(target_positions))
        block = logits[0, start:stop].float()
        values = torch.log_softmax(block, dim=-1).gather(1, labels[start:stop, None]).squeeze(1)
        if not torch.isfinite(values).all():
            raise ValueError('Nonfinite model-token likelihood')
        pieces.append(values.cpu())
    return torch.cat(pieces)


def finite_values(values):
    values = [float(x) for x in values]
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Nonfinite stored token likelihood')
    return values


def feedback_summary(factual, ablated, eligible_indices, prefix_tolerance=1e-5):
    factual, ablated = finite_values(factual), finite_values(ablated)
    if len(factual) != len(ablated):
        raise ValueError('Different factual/ablated target counts')
    if prefix_tolerance != 1e-5:
        raise ValueError('Causality tolerance is preregistered, not tuned to results')
    if eligible_indices and eligible_indices != list(range(eligible_indices[0], len(factual))):
        raise ValueError('Post-feedback target indices must form a causal suffix')
    if any(type(i) is not int or not 0 <= i < len(factual) for i in eligible_indices):
        raise ValueError('Invalid post-feedback target index')
    cutoff = eligible_indices[0] if eligible_indices else len(factual)
    prefix_error = max((abs(factual[i] - ablated[i]) for i in range(cutoff)), default=0.0)
    if prefix_error > prefix_tolerance:
        raise ValueError('Factual/ablated causal prefix logP changed: ' + str(prefix_error))
    difference_sum = sum(factual[i] - ablated[i] for i in eligible_indices)
    return {'value': difference_sum / len(eligible_indices) if eligible_indices else 0.0,
            'eligible_model_tokens': len(eligible_indices), 'difference_sum': difference_sum,
            'unchanged_prefix_model_tokens': cutoff, 'prefix_max_abs_logprob_difference': prefix_error,
            'prefix_tolerance': prefix_tolerance}


def reference_summary(logprobs):
    values = finite_values(logprobs)
    if not values:
        raise ValueError('Fixed reference corpus unexpectedly contains an empty model-token trajectory')
    total = sum(values)
    return {'value': total / len(values), 'model_tokens': len(values), 'logprob_sum': total}
