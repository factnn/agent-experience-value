"""Explicit lineage and intervention contract for corrected source-group RL jobs."""
NORMALIZATION = {'accelerate_backward_divisor': 1, 'grpo_loss_divisor': 4,
                 'hf_microbatch_accumulation': 4, 'deterministic_algorithms': True}


def validate_source(source_algorithm, target_algorithm, progress, optimizer_state, allow_legacy_zero):
    if target_algorithm.get('num_generations') != 4 or target_algorithm.get('num_iterations') != 1:
        raise ValueError('Only four-rollout single-iteration GRPO is supported')
    if any(target_algorithm.get(k) != v for k, v in NORMALIZATION.items()):
        raise ValueError('Corrected deterministic algorithm contract required')
    step = progress.get('cumulative_learner_optimizer_steps', progress.get('global_step'))
    if type(step) is not int or step < 0 or progress.get('micro_step') != 4*step:
        raise ValueError('Invalid cumulative learner position')
    if not progress.get('at_optimizer_boundary'):
        raise ValueError('Incomplete optimizer boundary')
    if step:
        if not progress.get('fresh_rollout_on_next_group'):
            raise ValueError('Cannot discard a live rollout buffer')
        origin = progress.get('source_branch_start_step')
        local = progress.get('branch_local_optimizer_steps')
        if type(origin) is not int or type(local) is not int or origin < 0 or local <= 0:
            raise ValueError('Explicit nested branch counters required')
        if origin+local != step or progress.get('global_step') != local:
            raise ValueError('HF local/cumulative counter mismatch')
        if source_algorithm != target_algorithm:
            raise ValueError('Do not import historical-scaled nonzero moments into the new scientific recipe')
    elif optimizer_state['state']:
        raise ValueError('A step-zero ancestor must have empty optimizer state')
    if source_algorithm != target_algorithm:
        if not allow_legacy_zero or step != 0 or optimizer_state['state']:
            raise ValueError('Algorithm transition only allowed at declared zero-optimizer ancestor')
        if target_algorithm != source_algorithm | NORMALIZATION:
            raise ValueError('Undeclared algorithm change')
    return step


def validate_intervention(job, registry):
    pool = job['task_pool']
    if not pool or len(set(pool)) != len(pool) or not set(pool) <= set(registry):
        raise ValueError('Invalid source task pool')
    role = job['role']
    if role in {'uniform_control', 'build_stage'}:
        if set(pool) != set(registry):raise ValueError('Uniform comparator/stage builder must use full training pool')
    elif role == 'source_group':
        group = job['source_group']
        if set(pool) != {t for t,g in registry.items() if g == group}:
            raise ValueError('Source intervention must include its entire registered group')
    else:raise ValueError('Unknown intervention role')
    if type(job['token_budget']) is not int or job['token_budget'] <= 0:
        raise ValueError('Positive token budget required')
    for name in ['allocator_seed','rollout_seed_base']:
        if type(job[name]) is not int or not 0 <= job[name] < 2**32:
            raise ValueError('Explicit valid seed required')


def learner_progress(source_step, local_step, inherited_tokens, new_tokens):
    if any(type(x) is not int or x < 0 for x in [source_step,local_step,inherited_tokens,new_tokens]):
        raise ValueError('Invalid progress or cost')
    return {'global_step':local_step,'branch_local_optimizer_steps':local_step,
            'source_branch_start_step':source_step,'cumulative_learner_optimizer_steps':source_step+local_step,
            'micro_step':4*(source_step+local_step),'at_optimizer_boundary':True,
            'fresh_rollout_on_next_group':True,'inherited_acquisition_tokens':inherited_tokens,
            'branch_sampled_tokens':new_tokens}
