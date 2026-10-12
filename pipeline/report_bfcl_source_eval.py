"""Replay transfer panel provenance, rewards, token lineage, and prediction timing."""
import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean
from bfcl_source_eval_contract import digest, load_job
from report_bfcl_intervention import read, lines, audit_sampling
from supervise_bfcl_stage_fork import ROOT, write


def audit_panel(job, evaluation, summary, run, outcomes, rollouts, calls, cost):
    tasks = job['task_ids']
    assert summary['status'] == 'complete' and summary['weights_unchanged'] and summary['optimizer_updates'] == 0
    assert summary['policy_case'] == run['policy_case'] == job['policy_case']
    assert summary['role'] == run['role'] == job['role']
    assert summary['evaluation_repeat'] == run['evaluation_repeat'] == job['evaluation_repeat']
    assert summary['policy_fingerprint'] == run['policy_fingerprint'] == job['policy_fingerprint']
    assert run['prediction_seal'] == job.get('prediction_seal')
    assert summary['completed_tasks'] == len(tasks) == len(outcomes) == len(rollouts) == cost['rollouts']
    assert [row['task_id'] for row in outcomes] == tasks
    assert [row['seed'] for row in outcomes] == job['task_seeds']
    assert {row['group'] for row in calls} == set(range(len(tasks)))
    by_stratum = defaultdict(list)
    for index, row in enumerate(outcomes):
        task, seed = tasks[index], job['task_seeds'][index]
        assert row['split'] == 'composition_transfer_eval' and row['stratum'] == evaluation['registry'][task]
        assert len(row['rewards']) == 1 and row['rewards'][0] in (0, 1)
        actual = [x for x in rollouts if x['group'] == index]
        assert len(actual) == 1
        actual = actual[0]
        assert actual['rollout_index'] == 0 and actual['task_id'] == task and actual['seed'] == seed
        assert actual['phase'] == 'evaluation' and actual['policy_fingerprint'] == job['policy_fingerprint']
        assert [actual['reward']] == row['rewards'] and [actual['stop_reason']] == row['stop_reasons']
        raw = [x for x in calls if x['group'] == index]
        assert all(x['seed'] == seed and x['phase'] == 'evaluation' and x['rollout_index'] == 0 for x in raw)
        assert sum(len(x['generated_ids']) for x in raw) == row['sampled_tokens'] <= evaluation['whole_completion_budget']
        by_stratum[row['stratum']].append(row['rewards'][0])
    assert sum(row['sampled_tokens'] for row in outcomes) == summary['sampled_tokens'] == cost['new_model_tokens']
    assert set(by_stratum) == set(evaluation['registry'].values())
    strata = {key: mean(values) for key, values in by_stratum.items()}
    return {'macro_success': mean(strata.values()), 'stratum_success': strata,
            'task_weighted_success': mean(row['rewards'][0] for row in outcomes)}


def audit_job(root, path):
    job, master, _, _, seal = load_job(root, path, digest(path), check_remote=False)
    out = root / job['output']
    summary, run = read(out/'summary.json'), read(out/'run.json')
    assert run['job_sha256'] == digest(path)
    if seal:
        publication = read((root / job['prediction_seal']['path']).parent / 'publication.json')
        assert seal['sealed_unix'] <= publication['verified_remote_at_unix'] <= run['evaluation_started_unix']
        assert run['heldout_gate'] == 'verified'
    else:
        assert run['heldout_gate'] == 'not_applicable'
    cost = audit_sampling(out)
    scores = audit_panel(job, master['evaluation'], summary, run, lines(out/'outcomes.jsonl'),
                         lines(out/'rollouts.jsonl'), lines(out/'generation_calls.jsonl'), cost)
    return {'status': 'passed', 'policy_case': job['policy_case'], 'evaluation_repeat': job['evaluation_repeat'],
            'role': job['role'], 'job_sha256': digest(path), 'policy_fingerprint': job['policy_fingerprint'],
            'cost': cost, **scores, 'source_files': {name: digest(out/name) for name in
             ['summary.json', 'run.json', 'outcomes.jsonl', 'rollouts.jsonl', 'generation_calls.jsonl']},
            'prediction_publication_timing_verified': bool(seal),
            'scope': 'One complete paired-panel input; no independent-repeat or transfer prediction claim.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = read(args.job)
    try:
        result = audit_job(ROOT, args.job)
    except Exception as error:
        result = {'status': 'failed', 'type': type(error).__name__, 'message': str(error), 'partial_evidence_retained': True}
    write(ROOT / job['output'] / 'audit.json', result)
    print(__import__('json').dumps(result))
    raise SystemExit(0 if result['status'] == 'passed' else 1)


if __name__ == '__main__':
    main()
