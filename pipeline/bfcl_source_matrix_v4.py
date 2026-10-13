"""Finalize the original 51 policies across four immutable execution segments.

The V3 handoff is an administrative admission stop, never a successful V3 batch.
"""
from bfcl_source_eval_contract import pinned
from bfcl_source_matrix_v3 import checked_segment, checked_continuation
from supervise_bfcl_source_training import digest
from supervise_bfcl_stage_fork import read, process_has_not_exited

STOP_MESSAGE = 'SSH owner stopped further job admission'


def check_ssh_authority(root, reference, protocol, launch):
    authority = pinned(root, reference)
    if (authority['training_protocol'] != protocol
            or authority['mode'] != 'human_authorized_ssh_takeover'
            or reference['sha256'] != launch['control_authority_sha256']):
        raise ValueError('SSH authority differs from launched control segment')
    for path, sha in authority['control_input_sha256'].items():
        if digest(root/path) != sha:
            raise ValueError('Registered SSH control dependency changed')
    return authority


def checked_handoff(root, master_ref, v3_ref, v3, v4):
    recovery = v4['execution_continuation']
    receipt = pinned(root, recovery['handoff_receipt'])
    if (receipt['status'] != 'authorized_transfer'
            or receipt['original_scientific_protocol'] != master_ref
            or receipt['source_protocol'] != v3_ref
            or receipt['source_output_root'] != v3['output_root']):
        raise ValueError('Handoff is bound to another scientific or control segment')
    snapshot = pinned(root, receipt['source_snapshot']['queue'])
    failures = pinned(root, receipt['source_snapshot']['failures'])
    transferred = receipt['transferred_pending']
    if (not transferred or len(set(transferred)) != len(transferred)
            or transferred != snapshot['pending'] or snapshot['failures'] != failures
            or receipt['admission_stop_failure'] not in failures
            or not failures or any(f['case'] not in transferred or f['type'] != 'RuntimeError'
                                   or f['message'] != STOP_MESSAGE for f in failures)):
        raise ValueError('Handoff lacks the actual stopped-admission pending partition')
    observed = snapshot['complete'] + list(snapshot['active'].values()) + transferred
    if len(set(observed)) != len(observed) or set(observed) != {e['name'] for e in v3['jobs']}:
        raise ValueError('Handoff snapshot duplicates or omits V3 identities')
    folder = root/v3['output_root']
    queue, terminal, launch = read(folder/'queue.json'), read(folder/'supervisor.json'), read(folder/'launch.json')
    expected_complete = set(snapshot['complete']) | set(snapshot['active'].values())
    if (queue['active'] or len(queue['complete']) != len(set(queue['complete']))
            or set(queue['complete']) != expected_complete or queue['pending'] != transferred
            or queue['failures'] != failures or read(folder/'failures.json') != failures
            or terminal['status'] != 'incomplete_requires_review'
            or terminal['complete_jobs'] != len(queue['complete']) or terminal['pending_jobs'] != transferred
            or process_has_not_exited(launch['supervisor_pid'])):
        raise ValueError('V3 has not finished precisely its retained live children')
    if (folder/'audit_status.json').exists():
        raise ValueError('An administratively partial V3 cannot claim a passed full batch')
    evidence = receipt['no_child_evidence']
    if set(evidence) != set(transferred):
        raise ValueError('Every transferred case needs contemporaneous no-child evidence')
    master_entries = {entry['name']: entry for entry in pinned(root, master_ref)['jobs']}
    source_entries = {entry['name']: entry for entry in v3['jobs']}
    for case in transferred:
        record = evidence[case]
        if (any(record[field] is not True for field in [
                'original_output_absent', 'v3_process_receipt_absent',
                'v3_run_receipt_absent', 'v3_console_absent'])
                or record['original_job'] != master_entries[case]
                or record['original_job'] != source_entries[case]
                or (folder/(case+'.console.process.json')).exists()):
            raise ValueError('Transferred job was already launched or differs from original identity')
    for path, sha in v3['input_sha256'].items():
        if digest(root/path) != sha:
            raise ValueError('Frozen V3 dependency changed')
    return folder, queue, launch, receipt


def check_execution_resources(root, plans_and_queues):
    """Check recorded actual device identity and overlapping owned-child intervals."""
    boundaries, device_intervals = [], {}
    for plan, queue in plans_and_queues:
        entries = {e['name']: e for e in plan['jobs']}
        lanes = {case: int(gpu) for gpu, names in plan['gpu_lanes'].items() for case in names}
        for case in queue['complete']:
            entry = entries[case]
            job = pinned(root, entry)
            output = root/job['output']
            run = read(output/'run.json')
            proc = read(root/plan['output_root']/(case+'.console.process.json'))
            terminal = read(output/'supervisor.json')
            if (run['case_id'] != case or run['job_sha256'] != entry['sha256']
                    or int(run['gpu']) != lanes[case] or int(run['gpu']) not in plan['allowed_gpus']
                    or run['pid'] != proc['pid'] or process_has_not_exited(proc['pid'])
                    or terminal['returncode'] != 0 or terminal['hard_timeout']):
                raise ValueError('Actual completed child differs from assigned device or original job')
            start, duration = proc['started_unix'], terminal['wall_seconds']
            if duration <= 0:
                raise ValueError('Missing actual execution duration')
            boundaries.extend([(start, 1), (start+duration, -1)])
            device_intervals.setdefault(int(run['gpu']), []).append((start, start+duration, case))
    for intervals in device_intervals.values():
        ordered = sorted(intervals)
        if any(a[1] > b[0] for a,b in zip(ordered, ordered[1:])):
            raise ValueError('Recorded child execution intervals overlap on one physical GPU')
    active = maximum = 0
    for _, delta in sorted(boundaries):
        active += delta
        maximum = max(maximum, active)
    if maximum > 6:
        raise ValueError('Actual V3/V4 child execution intervals exceed six concurrent workers')
    return {'actual_peak_owned_children': maximum, 'method': 'Recorded process start plus supervisor wall duration; V3 and V4 combined'}


def build_report_v4(root, master_ref, continuation_ref, audit_source):
    master, v4 = pinned(root, master_ref), pinned(root, continuation_ref)
    names = [e['name'] for e in master['jobs']]
    recovery = v4['execution_continuation']
    segments = recovery['previous_control_segments']
    if len(names) != 51 or len(set(names)) != 51 or len(segments) != 3 or segments[0]['protocol'] != master_ref:
        raise ValueError('Require original 51 jobs and original/V2/V3 control lineage')
    v2_ref, v3_ref = segments[1]['protocol'], segments[2]['protocol']
    v2, v3 = pinned(root, v2_ref), pinned(root, v3_ref)
    if [s['output_root'] for s in segments] != [p['output_root'] for p in [master, v2, v3]]:
        raise ValueError('Registered execution root substituted')
    original_root, original, _ = checked_segment(root, master, 24)
    v2_root, second, _ = checked_segment(root, v2, 12)
    checked_continuation(master_ref, master, v2, original['complete'], original['pending'])
    retained = original['complete'] + second['complete']
    if len(set(retained)) != 36:
        raise ValueError('Original and V2 overlap')
    checked_continuation(master_ref, master, v3, retained, second['pending'])
    v3_root, third, third_launch, handoff = checked_handoff(root, master_ref, v3_ref, v3, v4)
    transferred = handoff['transferred_pending']
    if (recovery['original_scientific_protocol'] != master_ref
            or recovery['completed_cases_retained'] != retained + pinned(root, handoff['source_snapshot']['queue'])['complete']
            or recovery['pending_cases_only'] != transferred
            or v4['input_sha256'] != master['input_sha256']
            or v4['jobs'] != [e for e in master['jobs'] if e['name'] in transferred]):
        raise ValueError('V4 changed or repeated original scientific jobs')
    if (not set(v4['allowed_gpus']) <= {0,1,4,5,6,7}
            or v4['max_simultaneous_gpus'] != len(v4['allowed_gpus'])
            or v4['automatic_retry'] is not False):
        raise ValueError('V4 resource/retry contract changed')
    lanes = [n for g, lane in v4['gpu_lanes'].items() for n in lane]
    if set(v4['gpu_lanes']) != {str(g) for g in v4['allowed_gpus']} or len(lanes) != len(set(lanes)) or set(lanes) != set(transferred):
        raise ValueError('V4 lanes do not uniquely assign transferred jobs to authorized devices')
    v4_root, fourth, fourth_launch = checked_segment(root, v4, len(transferred), final=True)
    completed = original['complete'] + second['complete'] + third['complete'] + fourth['complete']
    if len(completed) != 51 or len(set(completed)) != 51 or set(completed) != set(names):
        raise ValueError('Actual four segments do not cover original 51 jobs exactly once')
    v3_authority = {'path': 'rl/BFCL_SOURCE_STAGE0_SSH_AUTHORITY_20261013.json', 'sha256': third_launch['control_authority_sha256']}
    check_ssh_authority(root, v3_authority, v3_ref, third_launch)
    v4_authority = {'path': fourth_launch['control_authority_path'], 'sha256': fourth_launch['control_authority_sha256']}
    authority = check_ssh_authority(root, v4_authority, continuation_ref, fourth_launch)
    if (authority['allowed_gpus'] != v4['allowed_gpus']
            or authority['max_simultaneous_gpus'] != v4['max_simultaneous_gpus']
            or authority['aggregate_max_simultaneous_gpus'] != 6):
        raise ValueError('Actual V4 launch lacks explicit six-GPU aggregate authorization')
    resources = check_execution_resources(root, [(v3, third), (v4, fourth)])
    failed = recovery['failed_initial_admission']
    if failed != v2['execution_continuation']['failed_initial_admission'] or failed != v3['execution_continuation']['failed_initial_admission']:
        raise ValueError('Original failed admission lineage changed')
    failed_plan = pinned(root, failed['protocol'])
    failed_root = root/failed_plan['output_root']
    failures = read(root/failed['failures_path'])
    if (len(failures) != 1 or failures[0].get('gpu_children_launched') != 0
            or failures[0].get('phase') != 'preflight' or (failed_root/'launch.json').exists()
            or list(failed_root.glob('*.console.process.json'))):
        raise ValueError('Original failed continuation launched children')
    batch = read(v4_root/'audit_status.json')
    final_audits = {a['case_id']: a for a in batch['jobs']}
    if (batch['status'] != 'passed' or batch['complete_jobs'] != len(transferred)
            or len(final_audits) != len(batch['jobs']) or set(final_audits) != set(transferred)):
        raise ValueError('V4 full batch audit substituted or incomplete')
    index = read(root/'experiments/index.json')
    audits, evidence = [], []
    for entry in master['jobs']:
        job = pinned(root, entry)
        out = root/job['output']
        fresh = audit_source(root, job, entry['sha256'])
        if fresh['status'] != 'passed' or fresh != read(out/'audit.json'):
            raise ValueError('Fresh raw audit differs from saved endpoint')
        if entry['name'] in final_audits and fresh != final_audits[entry['name']]:
            raise ValueError('V4 batch audit differs from original policy')
        manifest = read(out/'evidence_manifest.json')
        if manifest['case_id'] != entry['name'] or not manifest['fresh_full_raw_audit_matches_saved']:
            raise ValueError('Complete raw evidence manifest required')
        for path, record in manifest['files'].items():
            if (out/path).stat().st_size != record['bytes'] or digest(out/path) != record['sha256']:
                raise ValueError('Completed endpoint evidence changed')
        events = [e for e in index if e['source'] == str((out/'summary.json').relative_to(root))]
        if not events or read(root/'experiments/reviews'/(events[-1]['id']+'.json'))['status'] != 'reviewed':
            raise ValueError('Every endpoint needs execution-agent review')
        audits.append(fresh)
        evidence.append({'job': entry, 'manifest_sha256': digest(out/'evidence_manifest.json'), 'review_event': events[-1]['id']})
    if sum(a['cost']['new_model_tokens'] for a in final_audits.values()) != batch['new_sampled_tokens']:
        raise ValueError('V4 cost inconsistent')
    controls = {str(folder.relative_to(root)): {n: digest(folder/n) for n in files}
        for folder, files in [(original_root, ['queue.json','supervisor.json','failures.json']),
            (v2_root, ['queue.json','supervisor.json','failures.json']), (failed_root, ['failures.json']),
            (v3_root, ['queue.json','supervisor.json','failures.json','launch.json']),
            (v4_root, ['queue.json','supervisor.json','audit_status.json','launch.json'])]}
    return {'status': 'passed', 'complete_jobs': 51,
        'new_sampled_tokens': sum(a['cost']['new_model_tokens'] for a in audits), 'jobs': audits,
        'execution_recovery': {'original_protocol': master_ref, 'continuation_protocol': continuation_ref,
            'previous_control_segments': segments, 'retained_jobs': 36+len(third['complete']),
            'continued_jobs': len(fourth['complete']), 'segment_complete_counts': [24,12,len(third['complete']),len(fourth['complete'])],
            'handoff_receipt': recovery['handoff_receipt'], 'v3_administrative_failures': third['failures'],
            'original_control_failures': original['failures'], 'v2_control_failures': second['failures'],
            'failed_initial_continuation': failed['protocol'], 'zero_child_admission_failure': failures,
            'source_sha256': controls, 'ssh_control_authorities': [v3_authority, v4_authority], 'resources': resources},
        'training_evidence': evidence, 'evaluation_performed': False, 'goal_complete': False,
        'scope': 'Full original 51-job matrix across four actual execution roots; administrative handoff and all failures retained; no transfer labels.'}
