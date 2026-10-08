"""Single-process learning-state snapshots at completed optimizer boundaries.

Stores trainable weights, buffers, optimizer, scheduler, RNG and allocation
history. Frozen base weights are referenced and fully hashed, not duplicated.
Does not yet wrap the HuggingFace trainer's resume/dataloader integration.
"""
import copy
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch

REQUIRED_PROVENANCE={'base_revision','task_manifest_sha256','algorithm_config_sha256'}


def tensor_hash(named):
    digest=hashlib.sha256()
    for name,tensor in named:
        tensor=tensor.detach().cpu().contiguous()
        digest.update(str((name,str(tensor.dtype),tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def signature(model):
    return [(n,tuple(p.shape),str(p.dtype),p.requires_grad) for n,p in model.named_parameters()]


def optimizer_signature(model,optimizer):
    names={id(p):n for n,p in model.named_parameters()}
    return [[names[id(p)] for p in group['params']] for group in optimizer.param_groups]


def save_learning_state(path,model,optimizer,scheduler,allocator,progress,provenance):
    if not REQUIRED_PROVENANCE<=provenance.keys():raise ValueError('missing base/task/algorithm provenance')
    if not progress.get('at_optimizer_boundary') or any(p.grad is not None for p in model.parameters()):
        raise ValueError('save only after optimizer update and zero_grad(set_to_none=True)')
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    allocation=allocator.state_dict()
    # deepcopy optimizer state: save must not alias a live branch's Adam moments.
    state={'version':1,'provenance':copy.deepcopy(provenance),'model_signature':signature(model),
        'frozen_hash':tensor_hash((n,p) for n,p in model.named_parameters() if not p.requires_grad),
        'trainable':{n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
        'buffers':{n:b.detach().cpu().clone() for n,b in model.named_buffers()},
        'module_training':{n:m.training for n,m in model.named_modules()},
        'optimizer_type':type(optimizer).__qualname__,'optimizer_signature':optimizer_signature(model,optimizer),
        'optimizer':copy.deepcopy(optimizer.state_dict()),
        'scheduler_type':type(scheduler).__qualname__,'scheduler':copy.deepcopy(scheduler.state_dict()),
        'allocator':allocation,'progress':copy.deepcopy(progress),
        'rng':{'python':random.getstate(),'numpy':np.random.get_state(),
               'torch_cpu':torch.get_rng_state(),
               'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}}
    torch.save(state,path/'learning_state.pt')
    payload_hash=hashlib.sha256((path/'learning_state.pt').read_bytes()).hexdigest()
    metadata={'version':1,'provenance':provenance,'progress':progress,
        'frozen_base_sha256':state['frozen_hash'],'trainable_sha256':tensor_hash(state['trainable'].items()),
        'payload_sha256':payload_hash,'source_allocation_rule':allocator.rule,
        'cumulative_acquisition_tokens':allocator.total_sampled_tokens,
        'scope':'single-process components; trainer resume/microbatch integration acceptance remains'}
    (path/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n');return metadata


def restore_learning_state(path,model,optimizer,scheduler,expected_provenance,rule=None):
    from rl_allocation import TaskAllocator
    path=Path(path);metadata=json.loads((path/'manifest.json').read_text())
    if hashlib.sha256((path/'learning_state.pt').read_bytes()).hexdigest()!=metadata['payload_sha256']:
        raise ValueError('learning-state payload checksum mismatch')
    # Local, self-created trusted artifact; contains optimizer and NumPy RNG state.
    state=torch.load(path/'learning_state.pt',map_location='cpu',weights_only=False)
    if state['version']!=1 or state['provenance']!=expected_provenance:raise ValueError('checkpoint provenance mismatch')
    if state['model_signature']!=signature(model):raise ValueError('model architecture/trainability mismatch')
    if state['frozen_hash']!=tensor_hash((n,p) for n,p in model.named_parameters() if not p.requires_grad):
        raise ValueError('frozen base parameters differ')
    if state['optimizer_type']!=type(optimizer).__qualname__ or state['optimizer_signature']!=optimizer_signature(model,optimizer):
        raise ValueError('optimizer or parameter-group ordering mismatch')
    if state['scheduler_type']!=type(scheduler).__qualname__:raise ValueError('scheduler type mismatch')
    if set(state['buffers'])!=dict(model.named_buffers()).keys():raise ValueError('model buffer registry mismatch')
    if set(state['module_training'])!=dict(model.named_modules()).keys():raise ValueError('model module registry mismatch')
    if state['rng']['torch_cuda'] is not None and len(state['rng']['torch_cuda'])!=torch.cuda.device_count():
        raise ValueError('visible CUDA device count mismatch')
    if state['allocator'].get('kind')=='bfcl_combination':
        from bfcl_allocation import CombinationAllocator
        allocator=CombinationAllocator.from_state_dict(state['allocator'],rule=rule)
    else:
        allocator=TaskAllocator.from_state_dict(state['allocator'],rule=rule)
    with torch.no_grad():
        for name,param in model.named_parameters():
            if param.requires_grad:param.copy_(state['trainable'][name].to(param.device))
        for name,buffer in model.named_buffers():buffer.copy_(state['buffers'][name].to(buffer.device))
    for name,module in model.named_modules():module.training=state['module_training'][name]
    optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
    optimizer.zero_grad(set_to_none=True)
    random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch_cpu'])
    if state['rng']['torch_cuda'] is not None:torch.cuda.set_rng_state_all(state['rng']['torch_cuda'])
    return allocator,copy.deepcopy(state['progress'])
