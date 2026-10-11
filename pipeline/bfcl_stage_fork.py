"""Boundary contract for fresh BFCL forks from trained component snapshots.

Only supports the existing one-generation-group-per-update GRPO configuration.
HF counters are local to the fork; learner progress and charged history persist.
"""


def validate_boundary(progress, algorithm):
    if algorithm['num_generations'] != 4 or algorithm['num_iterations'] != 1:
        raise ValueError('Only four-rollout, single-iteration boundaries are supported')
    step = progress['cumulative_learner_optimizer_steps']
    if type(step) is not int or step <= 0:
        raise ValueError('Expected a nonzero learner state')
    if not progress.get('at_optimizer_boundary') or not progress.get('fresh_rollout_on_next_group'):
        raise ValueError('Cannot discard an unfinished rollout buffer')
    if progress['micro_step'] != 4 * step:
        raise ValueError('Incomplete or incompatible accumulation boundary')
    # Initial acceptance uses snapshots of the original un-forked dose trajectory.
    if progress['global_step'] != step or progress['branch_local_optimizer_steps'] != step:
        raise ValueError('Nested forks require a separately accepted seed/progress map')
    return step


def continuation_seed(base_seed, source_step, local_step, condition_index=0):
    if local_step < 0 or source_step <= 0:
        raise ValueError('Invalid continuation position')
    return base_seed + 10000 * condition_index + 100 * (source_step + local_step)


def learner_position(source_step, local_step):
    return {'branch_local_optimizer_steps': local_step,
            'cumulative_learner_optimizer_steps': source_step + local_step,
            'micro_step': 4 * (source_step + local_step)}
