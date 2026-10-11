"""Pack archived BFCL groups exactly as the installed TRL padding path does."""
import torch


def pack(rows, group, pad_id):
    if len(rows) != 4 or [r['rollout_index'] for r in rows] != list(range(4)):
        raise ValueError('Expected one ordered four-rollout group')
    if [r['task_id'] for r in rows] != group['task_ids']:
        raise ValueError('Group/task mismatch')
    if [r['reward'] for r in rows] != group['rewards']:
        raise ValueError('Group/reward mismatch')
    pmax = max(len(r['prompt_ids']) for r in rows)
    cmax = max(len(r['completion_ids']) for r in rows)
    data = {k: [] for k in ('prompt_ids', 'prompt_mask', 'completion_ids', 'completion_mask', 'tool_mask')}
    for row in rows:
        p, c, mask = row['prompt_ids'], row['completion_ids'], row['model_token_mask']
        if not p or not c or len(c) != len(mask) or set(mask) - {0, 1}:
            raise ValueError('Malformed token/mask lineage')
        data['prompt_ids'].append([pad_id] * (pmax-len(p)) + p)
        data['prompt_mask'].append([0] * (pmax-len(p)) + [1] * len(p))
        data['completion_ids'].append(c + [pad_id] * (cmax-len(c)))
        data['completion_mask'].append([1] * len(c) + [0] * (cmax-len(c)))
        # TRL tool mask padding is 1; completion_mask excludes that padding.
        data['tool_mask'].append(mask + [1] * (cmax-len(c)))
    result = {k: torch.tensor(v, dtype=torch.long) for k, v in data.items()}
    result['advantages'] = torch.tensor(group['advantages'], dtype=torch.float32)
    if int((result['completion_mask'] * result['tool_mask']).sum()) != group['model_tokens_in_loss']:
        raise ValueError('Loss token count mismatch')
    external = sum(len(r['model_token_mask'])-sum(r['model_token_mask']) for r in rows)
    if external != group['external_tokens_excluded']:
        raise ValueError('External token count mismatch')
    if not torch.isfinite(result['advantages']).all():
        raise ValueError('Nonfinite advantages')
    return result


def tensor_distance(left, right):
    if left.keys() != right.keys():
        raise ValueError('Tensor names differ')
    count = different = 0
    max_abs = sum_squared = reference_squared = 0.0
    for key in left:
        a, b = left[key].double(), right[key].double()
        if a.shape != b.shape or left[key].dtype != right[key].dtype:
            raise ValueError('Tensor shape/dtype differs')
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Nonfinite tensors')
        delta = a-b
        count += a.numel()
        different += int(torch.count_nonzero(delta))
        max_abs = max(max_abs, float(delta.abs().max()))
        sum_squared += float((delta*delta).sum())
        reference_squared += float((a*a).sum())
    return {'numel':count, 'different_elements':different, 'exact':different == 0,
            'max_abs':max_abs, 'l2_difference':sum_squared**0.5,
            'reference_l2':reference_squared**0.5,
            'relative_l2':(sum_squared/reference_squared)**0.5 if reference_squared else None}
