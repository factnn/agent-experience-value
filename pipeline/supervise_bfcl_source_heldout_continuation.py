"""Reuse the existing eight-card supervisor for the original24 heldout handoff.

No original controller bytes change. Only the handoff phase check and the final
phase-certificate description/location differ from the existing fit adapter.
"""
from pathlib import Path
import supervise_bfcl_source_evaluation_eight as existing

ORIGINAL_SHA = '73cdc552862072169c6bfa5ca83a2dfd5d2b9178a5154bc0f35a0874b5e63b04'
_write = existing.write


def validate_handoff(root, plan):
    if (plan['phase'] != 'heldout' or len(plan['jobs']) != 24
            or plan['allowed_gpus'] != list(range(8)) or plan['max_simultaneous_gpus'] != 8
            or plan['original_manifest'] != {
                'path': 'rl/bfcl_source_stage0_heldout_jobs_20261015/manifest.json',
                'sha256': ORIGINAL_SHA}):
        raise ValueError('Only the original24 heldout eight-card continuation is supported')
    original = existing.pinned(root, plan['original_manifest'])
    for key in ('phase', 'execution_protocol', 'output_root', 'hard_seconds_per_panel',
                'prediction_seal', 'required_review_events', 'jobs'):
        if plan[key] != original[key]:
            raise ValueError('Handoff changed scientific field: '+key)
    if any(plan['input_sha256'].get(p) != sha for p, sha in original['input_sha256'].items()):
        raise ValueError('Original frozen dependency was changed')
    if plan['control_root'] == plan['output_root']:
        raise ValueError('Control records must preserve original history')
    receipt = existing.pinned(root, plan['handoff'])
    if receipt['original_manifest'] != plan['original_manifest'] or receipt['resource_authorization'] != plan['resource_authorization']:
        raise ValueError('Handoff binding differs')
    if existing.digest(root/receipt['old_stop_marker']['path']) != receipt['old_stop_marker']['sha256']:
        raise ValueError('Old admission latch disappeared or changed')
    names = {e['name'] for e in plan['jobs']}
    done, carry, pending = receipt['complete_at_latch'], receipt['carried'], receipt['pending_at_latch']
    allocated = done+[x['panel'] for x in carry]+pending
    if len(allocated) != len(names) or set(allocated) != names or len(set(allocated)) != len(allocated):
        raise ValueError('Original job identities omitted or duplicated')
    if len({x['gpu'] for x in carry}) != len(carry) or len({x['pid'] for x in carry}) != len(carry):
        raise ValueError('Carried process/GPU duplicated')
    for name in done:
        if not existing.completed_panel(root, plan, name):
            raise ValueError('Previously completed panel is incomplete: '+name)
    for info in carry:
        if info['panel'] not in plan['gpu_lanes'][str(info['gpu'])]:
            raise ValueError('Carried job assigned to a different GPU')
        entry = next(e for e in plan['jobs'] if e['name'] == info['panel'])
        process = existing.pinned(root, info['process_receipt'])
        if process['pid'] != info['pid'] or process['pgid'] != info['pid'] or process['hard_seconds'] != plan['hard_seconds_per_panel']:
            raise ValueError('Carried lifetime or hard budget differs')
        if '--job' not in process['command'] or process['command'][process['command'].index('--job')+1] != str(root/entry['path']):
            raise ValueError('Carried process belongs to another job')
        if '--job-sha256' not in process['command'] or process['command'][process['command'].index('--job-sha256')+1] != entry['sha256']:
            raise ValueError('Carried process uses different job bytes')
        if existing.process_state(info['pid'], info['start_ticks']) not in ('missing', 'reused', 'Z', 'X'):
            proc = Path('/proc')/str(info['pid'])
            actual = [x.decode() for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
            if actual != process['command']:
                raise ValueError('Live carried command differs')
            env = (proc/'environ').read_bytes().split(b'\0')
            cuda = next((x.split(b'=',1)[1].decode() for x in env if x.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
            if cuda != str(info['gpu']):
                raise ValueError('Live carried GPU differs')
    for name in pending:
        job = existing.pinned(root,next(e for e in plan['jobs'] if e['name'] == name))
        if (root/job['output']).exists():
            raise ValueError('Pending job has execution evidence; do not rerun: '+name)
    return receipt


def write_phase_completion(path, value):
    if path.name == 'audit_status.json' and value.get('phase') == 'heldout':
        if value.get('complete_panels') != 24 or len(value.get('jobs',[])) != 24:
            raise ValueError('Require all24 original heldout audits')
        original = existing.pinned(existing.ROOT, value['original_manifest'])
        canonical = existing.ROOT/original['output_root']/'audit_status.json'
        if canonical.exists():
            raise FileExistsError('Preserve any existing phase certificate')
        value = value | {'scope': 'All24 original heldout job identities completed once across original and continuation controls. Original seal and all failure histories preserved; no scientific retry.'}
        _write(path, value)
        with canonical.open('xb') as f:
            f.write(path.read_bytes())
        return
    _write(path, value)


def main():
    existing.validate_handoff = validate_handoff
    existing.write = write_phase_completion
    existing.main()


if __name__ == '__main__':
    main()
