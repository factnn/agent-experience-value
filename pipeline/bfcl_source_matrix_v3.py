"""Merge the original 24, V2's 12, and V3's 15 without rewriting any control history."""
from bfcl_source_eval_contract import pinned
from supervise_bfcl_source_training import digest
from supervise_bfcl_stage_fork import read, process_has_not_exited


def checked_segment(root, plan, expected_complete, final=False):
    folder = root/plan['output_root']
    queue, terminal = read(folder/'queue.json'), read(folder/'supervisor.json')
    complete, pending = queue['complete'], queue['pending']
    names = [e['name'] for e in plan['jobs']]
    if (queue['active'] or len(set(complete)) != len(complete) or len(set(pending)) != len(pending)
            or set(complete) & set(pending) or set(complete) | set(pending) != set(names)
            or len(complete) != expected_complete or terminal['complete_jobs'] != len(complete)):
        raise ValueError('Control segment completion/partition differs from registration')
    expected_status = 'audited_complete' if final else 'incomplete_requires_review'
    if terminal['status'] != expected_status or terminal['pending_jobs'] != pending:
        raise ValueError('Control terminal does not match queue')
    if final:
        if pending or queue['failures']:
            raise ValueError('Final continuation is incomplete or failed')
    elif not queue['failures'] or any(f['case'] not in pending or f['type'] != 'RuntimeError'
            or f['message'] != 'Goal explicitly unavailable for work: blocked' for f in queue['failures']):
        raise ValueError('Prior failure is not the retained zero-child control admission failure')
    if not final and read(folder/'failures.json') != queue['failures']:
        raise ValueError('Failure receipts differ from queue')
    launch = read(folder/'launch.json')
    if process_has_not_exited(launch['supervisor_pid']):
        raise ValueError('Every training supervisor must have exited')
    for path, sha in plan['input_sha256'].items():
        if digest(root/path) != sha:
            raise ValueError('Frozen dependency changed: '+path)
    return folder, queue, launch


def checked_continuation(master_ref, master, plan, retained, pending):
    recovery = plan['execution_continuation']
    if (recovery['original_scientific_protocol'] != master_ref
            or recovery['completed_cases_retained'] != retained or recovery['pending_cases_only'] != pending):
        raise ValueError('Continuation lineage/retained/unstarted identities differ')
    if plan['jobs'] != [e for e in master['jobs'] if e['name'] in pending]:
        raise ValueError('Continuation changed or replaced original scientific jobs')
    if (plan['allowed_gpus'] != master['allowed_gpus'] or plan['max_simultaneous_gpus'] != 4
            or plan['automatic_retry'] is not False
            or plan['gpu_lanes'] != {g: [n for n in lane if n in pending] for g,lane in master['gpu_lanes'].items()}):
        raise ValueError('Continuation changed resources or GPU lane order')


def build_report_v3(root, master_ref, continuation_ref, audit_source):
    master, v3 = pinned(root, master_ref), pinned(root, continuation_ref)
    names = [e['name'] for e in master['jobs']]
    if len(names) != 51 or len(set(names)) != 51:
        raise ValueError('Require the original unique 51-job matrix')
    recovery = v3['execution_continuation']
    segments = recovery['previous_control_segments']
    if len(segments) != 2 or segments[0]['protocol'] != master_ref:
        raise ValueError('Require the registered original and V2 control lineage')
    v2_ref = segments[1]['protocol']
    v2 = pinned(root, v2_ref)
    if [s['output_root'] for s in segments] != [master['output_root'], v2['output_root']]:
        raise ValueError('Control root substituted')
    if [s['complete_jobs'] for s in segments] != [24, 12]:
        raise ValueError('Registered retained counts changed')
    original_root, original, _ = checked_segment(root, master, 24)
    v2_root, second, _ = checked_segment(root, v2, 12)
    checked_continuation(master_ref, master, v2, original['complete'], original['pending'])
    retained = original['complete'] + second['complete']
    if len(set(retained)) != 36:
        raise ValueError('Retained control segments overlap')
    checked_continuation(master_ref, master, v3, retained, second['pending'])
    final_root, final, launch = checked_segment(root, v3, 15, final=True)
    if set(retained) & set(final['complete']) or set(retained+final['complete']) != set(names):
        raise ValueError('Actual three execution roots do not uniquely cover 51 jobs')
    # SSH execution records the new owner; native-Goal execution remains supported.
    authority_ref = None
    if 'control_authority_sha256' in launch:
        authority_ref = {'path': 'rl/BFCL_SOURCE_STAGE0_SSH_AUTHORITY_20261013.json',
                         'sha256': launch['control_authority_sha256']}
        authority = pinned(root, authority_ref)
        if authority['training_protocol'] != continuation_ref or authority['mode'] != 'human_authorized_ssh_takeover':
            raise ValueError('SSH authority was bound to a different execution')
        for path, sha in authority['control_input_sha256'].items():
            if digest(root/path) != sha:
                raise ValueError('Registered SSH controller changed')
    failed = recovery['failed_initial_admission']
    if failed != v2['execution_continuation']['failed_initial_admission']:
        raise ValueError('The original zero-child failure lineage changed')
    failed_plan = pinned(root, failed['protocol'])
    failed_root = root/failed_plan['output_root']
    failures = read(root/failed['failures_path'])
    if (len(failures) != 1 or failures[0].get('gpu_children_launched') != 0
            or failures[0].get('phase') != 'preflight' or (failed_root/'launch.json').exists()
            or list(failed_root.glob('*.console.process.json'))):
        raise ValueError('Failed initial continuation must have launched zero children')
    batch = read(final_root/'audit_status.json')
    if batch['status'] != 'passed' or batch['complete_jobs'] != 15:
        raise ValueError('Final continuation batch is not completely audited')
    final_audits = {a['case_id']: a for a in batch['jobs']}
    if len(final_audits) != len(batch['jobs']) or set(final_audits) != set(final['complete']):
        raise ValueError('Final continuation batch substituted/duplicated an audit')
    index = read(root/'experiments/index.json')
    audits, evidence = [], []
    for entry in master['jobs']:
        job = pinned(root, entry)
        out = root/job['output']
        fresh = audit_source(root, job, entry['sha256'])
        if fresh['status'] != 'passed' or fresh != read(out/'audit.json'):
            raise ValueError('Fresh raw replay differs from saved endpoint audit')
        if entry['name'] in final_audits and fresh != final_audits[entry['name']]:
            raise ValueError('Final control batch audit differs from the actual policy')
        manifest = read(out/'evidence_manifest.json')
        if manifest['case_id'] != entry['name'] or not manifest['fresh_full_raw_audit_matches_saved']:
            raise ValueError('Complete raw evidence manifest required')
        for path, record in manifest['files'].items():
            if (out/path).stat().st_size != record['bytes'] or digest(out/path) != record['sha256']:
                raise ValueError('Completed endpoint evidence changed')
        events = [e for e in index if e['source'] == str((out/'summary.json').relative_to(root))]
        if not events or read(root/'experiments/reviews'/(events[-1]['id']+'.json'))['status'] != 'reviewed':
            raise ValueError('Every training endpoint must be reviewed by the execution agent')
        audits.append(fresh)
        evidence.append({'job': entry,'manifest_sha256':digest(out/'evidence_manifest.json'),'review_event':events[-1]['id']})
    if sum(a['cost']['new_model_tokens'] for a in final_audits.values()) != batch['new_sampled_tokens']:
        raise ValueError('Final continuation cost inconsistent')
    controls={str(folder.relative_to(root)):{n:digest(folder/n) for n in files}
        for folder,files in [(original_root,['queue.json','supervisor.json','failures.json']),
                             (v2_root,['queue.json','supervisor.json','failures.json']),
                             (failed_root,['failures.json']),
                             (final_root,['queue.json','supervisor.json','audit_status.json','launch.json'])]}
    return {'status':'passed','complete_jobs':51,
        'new_sampled_tokens':sum(a['cost']['new_model_tokens'] for a in audits),'jobs':audits,
        'execution_recovery':{'original_protocol':master_ref,'continuation_protocol':continuation_ref,
            'previous_control_segments':segments,'retained_jobs':36,'continued_jobs':15,
            'failed_initial_continuation':failed['protocol'],'zero_child_admission_failure':failures,
            'original_control_failures':original['failures'],'v2_control_failures':second['failures'],
            'source_sha256':controls,'ssh_control_authority':authority_ref},
        'training_evidence':evidence,'evaluation_performed':False,'goal_complete':False,
        'scope':'Full original matrix across 24+12+15 actual executions. Failures retained. No transfer labels.'}
