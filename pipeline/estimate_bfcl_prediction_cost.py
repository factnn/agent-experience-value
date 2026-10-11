"""Resource scenarios from completed cost summaries; no outcome scores or launches."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / 'rl/bfcl_intervention_dose_repeats_20261011'


def estimate():
    sources, eval_seconds, eval_tokens, train_rates = [], [], [], []
    for path in sorted(BATCH.glob('*/summary.json')):
        raw = path.read_bytes()
        row = json.loads(raw)
        if row.get('status') != 'complete':
            continue
        if path.parent.name.startswith('eval_'):
            assert row['completed_tasks'] == 22
            eval_seconds.append(row['wall_seconds'] / row['completed_tasks'])
            eval_tokens.append(row['sampled_tokens'] / row['completed_tasks'])
            facts = {k: row[k] for k in ['completed_tasks', 'sampled_tokens', 'wall_seconds']}
        elif path.parent.name.startswith('train_'):
            train_rates.append(row['wall_seconds'] / row['branch_sampled_tokens'])
            facts = {k: row[k] for k in ['branch_sampled_tokens', 'wall_seconds']}
        else:
            continue
        sources.append({'path': str(path.relative_to(ROOT)),
                        'sha256': hashlib.sha256(raw).hexdigest(), 'cost_facts': facts})
    assert len(train_rates) == 3 and eval_seconds
    rates = {'completed_evaluation_panels': len(eval_seconds),
             'completed_training_trajectories': len(train_rates),
             'evaluation_seconds_per_task_panel_mean': statistics.mean(eval_seconds),
             'evaluation_seconds_per_task_panel_mean_range': [min(eval_seconds), max(eval_seconds)],
             'evaluation_tokens_per_task_mean': statistics.mean(eval_tokens),
             'training_seconds_per_token_mean': statistics.mean(train_rates)}
    scenarios = []
    for groups, training_seeds in [(16, 3), (16, 2), (8, 3)]:
        stages, evaluation_seeds, tasks, budget = 2, 2, 32, 65536
        branches = groups * stages * training_seeds
        uniform = stages * training_seeds
        trained = branches + uniform
        policies = trained + stages  # One shared untrained baseline at each stage.
        episodes = policies * evaluation_seeds * tasks
        train_tokens = trained * budget
        train_hours = train_tokens * statistics.mean(train_rates) / 3600
        eval_hours = episodes * statistics.mean(eval_seconds) / 3600
        scenarios.append({'source_groups': groups, 'stages': stages,
            'training_seeds': training_seeds, 'evaluation_seeds': evaluation_seeds,
            'evaluation_tasks': tasks, 'training_budget_per_branch': budget,
            'source_group_branches': branches, 'full_pool_uniform_branches': uniform,
            'shared_stage_baselines': stages, 'unique_endpoint_or_baseline_policies': policies,
            'evaluation_episodes': episodes, 'nominal_training_tokens': train_tokens,
            'estimated_evaluation_tokens': episodes * statistics.mean(eval_tokens),
            'estimated_training_gpu_hours': train_hours,
            'estimated_evaluation_gpu_hours': eval_hours,
            'evaluation_gpu_hours_using_observed_panel_rate_range':
                [episodes * min(eval_seconds) / 3600, episodes * max(eval_seconds) / 3600],
            'estimated_total_gpu_hours_exclusions_apply': train_hours + eval_hours,
            'ideal_four_gpu_hours_exclusions_apply': (train_hours + eval_hours) / 4})
    return {'status': 'planning_only_not_frozen_or_launched',
        'created_at': datetime.now(timezone.utc).isoformat(), 'sources': sources,
        'rates': rates, 'scenarios': scenarios, 'new_model_calls': 0,
        'assumptions': ['Two stages, one endpoint per branch, shared baseline per stage and evaluation seed.',
            'All proposed trained policies evaluated on the same 32 tasks at two seeds; no aliases assumed.',
            'Training wall time prorated linearly from three completed full-pool trajectories; not a runtime guarantee.',
            'Panel mean range is descriptive across completed panels, not a confidence or prediction interval.',
            'Four-GPU division assumes perfect availability and balance; actual elapsed time can be longer.'],
        'excluded_costs': ['Token overshoot at generation-group boundaries.',
            'Additional balanced current-policy probes and their shared acquisition ledger.',
            'Engineering fork acceptance, failed attempts, restarts and new model downloads/acceptance.',
            'A separate 21-task target-generalization confirmation panel and any other model/environment validation.'],
        'interpretation': 'Resource tradeoffs only. Fewer groups reduce prediction conditions; fewer seeds reduce repetition. Neither option is selected or certified statistically identifiable.'}


if __name__ == '__main__':
    result = estimate()
    output = ROOT / 'rl/bfcl_prediction_cost_scenarios_20261011.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['rates', 'scenarios']}, indent=2))
