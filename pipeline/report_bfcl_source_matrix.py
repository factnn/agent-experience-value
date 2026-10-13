"""Finalize the original 51-job audit after the registered control continuation.

No sampling, queue edits, Goal changes, or rewriting of prior failure receipts.
"""
import argparse
import json
from pathlib import Path
from bfcl_source_eval_contract import pinned
from supervise_bfcl_source_training import digest
from supervise_bfcl_stage_fork import ROOT, read, process_has_not_exited


def validate_partition(master_ref, master, continuation, original, resumed,
                       old_terminal, new_terminal, admission_failure):
    entries = master['jobs']
    names = [e['name'] for e in entries]
    recovery = continuation['execution_continuation']
    if len(names) != 51 or len(set(names)) != 51:
        raise ValueError('Require the original unique 51-job matrix')
    if recovery['original_scientific_protocol'] != master_ref:
        raise ValueError('Continuation is bound to another scientific matrix')
    kept, pending = original['complete'], original['pending']
    if (original['active'] or len(set(kept)) != len(kept) or len(set(pending)) != len(pending)
            or set(kept) & set(pending) or set(kept) | set(pending) != set(names)):
        raise ValueError('Original control state is incomplete or overlaps')
    if kept != recovery['completed_cases_retained'] or pending != recovery['pending_cases_only']:
        raise ValueError('Retained and unstarted cases changed')
    expected = [e for e in entries if e['name'] in pending]
    if continuation['jobs'] != expected:
        raise ValueError('Continuation changed, omitted, or repeated an original job')
    if (continuation['allowed_gpus'] != master['allowed_gpus']
            or continuation['max_simultaneous_gpus'] != 4 or continuation['automatic_retry'] is not False):
        raise ValueError('Continuation resource or retry contract changed')
    lanes = {gpu: [n for n in lane if n in pending] for gpu, lane in master['gpu_lanes'].items()}
    if continuation['gpu_lanes'] != lanes:
        raise ValueError('Original GPU lane order changed')
    if old_terminal.get('status') != 'incomplete_requires_review' or old_terminal['complete_jobs'] != len(kept):
        raise ValueError('Original interrupted execution has no matching terminal receipt')
    if not original['failures'] or any(
        f['case'] not in pending or f['type'] != 'RuntimeError'
        or f['message'] != 'Goal explicitly unavailable for work: blocked'
        for f in original['failures']
    ):
        raise ValueError('Only the recorded zero-sampling blocked admission is recoverable here')
    if (len(admission_failure) != 1 or admission_failure[0].get('gpu_children_launched') != 0
            or admission_failure[0].get('phase') != 'preflight'):
        raise ValueError('The failed initial continuation must have launched zero children')
    if (resumed['active'] or resumed['pending'] or resumed['failures']
            or len(resumed['complete']) != len(pending) or set(resumed['complete']) != set(pending)
            or new_terminal.get('status') != 'audited_complete'
            or new_terminal['complete_jobs'] != len(pending)):
        raise ValueError('Registered continuation is not completely audited')
    return names


def audit_source(root, job, expected_sha):
    import report_bfcl_source_training as auditor
    if root.resolve() != auditor.ROOT.resolve():
        raise ValueError('Actual training auditor is bound to this research repository')
    return auditor.audit_job(job, expected_sha)


def build_report(root, master_ref, continuation_ref):
    master, continuation = pinned(root, master_ref), pinned(root, continuation_ref)
    if len(continuation['execution_continuation'].get('previous_control_segments', [])) == 3:
        from bfcl_source_matrix_v4 import build_report_v4
        return build_report_v4(root, master_ref, continuation_ref, audit_source)
    if continuation['execution_continuation'].get('previous_control_segments'):
        from bfcl_source_matrix_v3 import build_report_v3
        return build_report_v3(root, master_ref, continuation_ref, audit_source)
    original_root, resumed_root = root/master['output_root'], root/continuation['output_root']
    failed = continuation['execution_continuation']['failed_initial_admission']
    failed_plan = pinned(root, failed['protocol'])
    failed_root = root/failed_plan['output_root']
    original, resumed = read(original_root/'queue.json'), read(resumed_root/'queue.json')
    old_terminal, new_terminal = read(original_root/'supervisor.json'), read(resumed_root/'supervisor.json')
    admission_failure = read(root/failed['failures_path'])
    validate_partition(master_ref, master, continuation, original, resumed,
                       old_terminal, new_terminal, admission_failure)
    if (failed_root/'launch.json').exists() or list(failed_root.glob('*.console.process.json')):
        raise ValueError('Failed continuation contains launched child evidence')
    for folder in [original_root, resumed_root]:
        if process_has_not_exited(read(folder/'launch.json')['supervisor_pid']):
            raise ValueError('Training supervisor must have exited before finalization')
    for plan in [master, continuation]:
        for name, sha in plan['input_sha256'].items():
            if digest(root/name) != sha:
                raise ValueError('Frozen execution dependency changed: '+name)
    resumed_batch = read(resumed_root/'audit_status.json')
    if resumed_batch['status'] != 'passed' or resumed_batch['complete_jobs'] != len(continuation['jobs']):
        raise ValueError('Continuation batch audit is incomplete')
    resumed_audits = {a['case_id']: a for a in resumed_batch['jobs']}
    if (set(resumed_audits) != {e['name'] for e in continuation['jobs']}
            or len(resumed_batch['jobs']) != len(resumed_audits)):
        raise ValueError('Duplicate or missing continuation audits')
    index = read(root/'experiments/index.json')
    audits, evidence = [], []
    for entry in master['jobs']:
        job = pinned(root, entry)
        out = root/job['output']
        fresh = audit_source(root, job, entry['sha256'])
        if fresh['status'] != 'passed' or fresh != read(out/'audit.json'):
            raise ValueError('Fresh complete raw audit differs from saved audit')
        if entry['name'] in resumed_audits and fresh != resumed_audits[entry['name']]:
            raise ValueError('Continuation audit differs from original policy evidence')
        manifest = read(out/'evidence_manifest.json')
        if manifest['case_id'] != entry['name'] or not manifest['fresh_full_raw_audit_matches_saved']:
            raise ValueError('Missing full manual evidence manifest')
        for name, record in manifest['files'].items():
            if (out/name).stat().st_size != record['bytes'] or digest(out/name) != record['sha256']:
                raise ValueError('Finished evidence changed: '+str(out/name))
        events = [e for e in index if e['source'] == str((out/'summary.json').relative_to(root))]
        if not events or read(root/'experiments/reviews'/(events[-1]['id']+'.json'))['status'] != 'reviewed':
            raise ValueError('Every completed training policy must have a manual review')
        audits.append(fresh)
        evidence.append({'job': entry, 'manifest_sha256': digest(out/'evidence_manifest.json'),
                         'review_event': events[-1]['id']})
    if (sum(a['cost']['new_model_tokens'] for a in resumed_audits.values())
            != resumed_batch['new_sampled_tokens']):
        raise ValueError('Continuation cost total is inconsistent')
    controls = {str(folder.relative_to(root)): {n: digest(folder/n) for n in files}
                for folder, files in [(original_root, ['queue.json', 'supervisor.json', 'failures.json']),
                                      (failed_root, ['failures.json']),
                                      (resumed_root, ['queue.json', 'supervisor.json', 'audit_status.json'])]}
    return {'status': 'passed', 'complete_jobs': 51,
            'new_sampled_tokens': sum(a['cost']['new_model_tokens'] for a in audits), 'jobs': audits,
            'execution_recovery': {'original_protocol': master_ref, 'continuation_protocol': continuation_ref,
                                   'failed_initial_continuation': failed['protocol'],
                                   'retained_jobs': len(original['complete']), 'continued_jobs': len(resumed['complete']),
                                   'original_control_failures': original['failures'],
                                   'zero_child_admission_failure': admission_failure, 'source_sha256': controls},
            'training_evidence': evidence, 'evaluation_performed': False, 'goal_complete': False,
            'scope': 'Original complete training matrix after registered execution recovery. All failures preserved; no transfer labels or Goal completion.'}


def finalize(root, master_ref, continuation_ref):
    master = pinned(root, master_ref)
    output = root/master['output_root']/'audit_status.json'
    if output.exists():
        raise FileExistsError('Do not overwrite a complete matrix audit')
    result = build_report(root, master_ref, continuation_ref)
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    for name in ['protocol', 'continuation']:
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    args = parser.parse_args()
    def reference(name):
        return {'path': str(getattr(args, name).resolve().relative_to(ROOT)),
                'sha256': getattr(args, name+'_sha256')}
    result = finalize(ROOT, reference('protocol'), reference('continuation'))
    print(json.dumps({'status': result['status'], 'complete_jobs': result['complete_jobs'],
                      'new_sampled_tokens': result['new_sampled_tokens'], 'goal_complete': False}))


if __name__ == '__main__':
    main()
