"""Completed-group dose labels and a checked, transparent component snapshot.

This saves components; it does not implement HuggingFace dataloader resume.
"""
import copy
import random
import time
import numpy as np
import torch
from rl_learning_state import save_learning_state, restore_learning_state, tensor_hash


def crossed_thresholds(thresholds, saved, before, after):
    if after < before or sorted(set(thresholds)) != list(thresholds):
        raise ValueError('invalid dose order')
    return [b for b in thresholds if b not in saved and before < b <= after]


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state().clone(),
            'cuda': [s.clone() for s in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized() else None}


def assert_equal(a, b):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): assert_equal(x, y)
    else:
        assert a == b


def checked_snapshot(path, model, optimizer, scheduler, allocator, progress, provenance,
                     save_adapter=None):
    """Verify serialization and same-live-state restoration at a zero-grad boundary.

    No model sampling occurs here. The original allocator stays live; its full
    history and private RNG must equal the recovered allocator. Nonzero-step
    trainer continuation/fork integration is a separate, future experiment.
    """
    started = time.perf_counter()
    before_rng = rng_state()
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    before_allocator = copy.deepcopy(allocator.state_dict())
    before_weights = tensor_hash((n,p) for n,p in model.named_parameters() if p.requires_grad)
    before_buffers = tensor_hash(model.named_buffers())
    before_modes = {n:m.training for n,m in model.named_modules()}
    manifest = save_learning_state(path, model, optimizer, scheduler, allocator, progress, provenance)
    if save_adapter is not None: save_adapter()
    # Check saving itself before restoration can hide any side effects.
    assert_equal(before_rng, rng_state())
    assert_equal(before_optimizer, optimizer.state_dict())
    assert_equal(before_scheduler, scheduler.state_dict())
    assert_equal(before_allocator, allocator.state_dict())
    assert before_weights == tensor_hash((n,p) for n,p in model.named_parameters() if p.requires_grad)
    assert before_buffers == tensor_hash(model.named_buffers())
    assert before_modes == {n:m.training for n,m in model.named_modules()}
    recovered, restored_progress = restore_learning_state(
        path, model, optimizer, scheduler, provenance)
    assert_equal(progress, restored_progress)
    assert_equal(before_rng, rng_state())
    assert_equal(before_optimizer, optimizer.state_dict())
    assert_equal(before_scheduler, scheduler.state_dict())
    assert_equal(before_allocator, recovered.state_dict())
    assert before_weights == tensor_hash((n,p) for n,p in model.named_parameters() if p.requires_grad)
    assert before_buffers == tensor_hash(model.named_buffers())
    assert before_modes == {n:m.training for n,m in model.named_modules()}
    assert all(p.grad is None for p in model.parameters())
    return manifest, {'status':'passed', 'wall_seconds':time.perf_counter()-started,
        'checks':['save has no RNG/model/buffer/mode/optimizer/scheduler/allocator effects',
                  'same-live-state component restoration is exact', 'zero pending gradients'],
        'scope':'Component roundtrip; does not validate nonzero-step HF trainer resume.'}
