"""Explicit HF-local versus cumulative learner counters for this paired fork."""


def validate_pair_boundary(progress, algorithm):
    if algorithm['num_generations'] != 4 or algorithm['num_iterations'] != 1:
        raise ValueError('Only a completed four-rollout single-iteration group is supported')
    origin = progress['source_branch_start_step']
    local = progress['branch_local_optimizer_steps']
    cumulative = progress['cumulative_learner_optimizer_steps']
    if any(type(x) is not int for x in [origin,local,cumulative]) or origin < 0 or local <= 0:
        raise ValueError('Invalid branch counters')
    if progress['global_step'] != local or origin+local != cumulative:
        raise ValueError('HF-local/cumulative counter mismatch')
    if progress['micro_step'] != 4*cumulative:
        raise ValueError('Incomplete microbatch boundary')
    if not progress.get('at_optimizer_boundary') or not progress.get('fresh_rollout_on_next_group'):
        raise ValueError('Cannot discard pending gradients or rollouts')
    if progress['inherited_acquisition_tokens'] < 0 or progress['branch_sampled_tokens'] < 0:
        raise ValueError('Negative charged cost')
    return cumulative
