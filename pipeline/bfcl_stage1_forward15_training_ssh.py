"""Adapt the published stage1 controller to the user-approved15-branch batch.

Only plan validation, phase ownership and batch certificate/count metadata change.
The original scientific trainer,auditor,single-attempt worker and process bounds
are called unchanged. Bindings are process-local and restored after CPU tests.
"""
import argparse
from contextlib import contextmanager

from bfcl_source_eval_contract import pinned
from bfcl_stage1_forward15_contract import MODE, SELF as CONTRACT, validate
import bfcl_stage1_training_ssh as core
from supervise_bfcl_stage_fork import ROOT, read

SELF = 'pipeline/bfcl_stage1_forward15_training_ssh.py'
SCOPE = 'stage1_fixed_parent_forward_training_15'
ORIGINAL_EXCLUSIVE = core.exclusive


def load_plan(root, protocol_ref):
    plan = pinned(root, protocol_ref)
    if plan.get('batch_mode') != MODE:
        raise ValueError('This entry accepts only the registered forward15 protocol')
    jobs = validate(root, plan)
    cap = pinned(root, plan['resource_authorization'])
    if plan['resource_authorization']['path'] != core.CAP or cap['max_simultaneous_gpus_for_subsequent_batches'] != 8:
        raise ValueError('Wrong current eight-free-card amendment')
    return plan, jobs


def audit_batch(root, protocol_ref, master, authority, control_root):
    if master['batch_mode'] != MODE or master['training_jobs'] != 15 or len(master['jobs']) != 15:
        raise ValueError('Wrong active15 training certificate scope')
    audits = []
    for entry in master['jobs']:
        job = pinned(root, entry)
        audit = read(root/job['output']/'audit.json')
        if (audit['status'] != 'passed' or audit['job_sha256'] != entry['sha256']
                or audit['cost']['new_model_tokens'] > job['token_budget']+32768):
            raise ValueError('Incomplete original15 raw training audits')
        audits.append(audit)
    actual = sum(audit['cost']['new_model_tokens'] for audit in audits)
    if actual > master['new_training_token_ceiling']:
        raise ValueError('Forward15 training token ceiling exceeded')
    certificate = {'status': 'passed', 'stage': 1, 'batch_mode': MODE, 'complete_jobs': 15,
                   'execution_protocol': protocol_ref, 'forward_bundle': authority['forward_bundle'],
                   'new_sampled_tokens': actual, 'jobs': audits, 'ancestor_tokens_recharged': 0,
                   'evaluation_performed': False, 'original_full51_complete': False,
                   'scope': 'All15 active forward-only branches audited;36 original branches deferred. Execution review and32 target panels remain.'}
    path = root/master['complete_training_audit_path']
    ORIGINAL_EXCLUSIVE(path, certificate)
    ORIGINAL_EXCLUSIVE(control_root/'training_audit_binding.json', {'complete_audit': core.ref(root, path)})


def write_metadata(path, value):
    if path.name == 'launch.json':
        value = value | {'scientific_jobs': 15, 'batch_mode': MODE}
    elif path.name == 'training_control_result.json':
        value = value | {'expected_jobs': 15, 'batch_mode': MODE}
    ORIGINAL_EXCLUSIVE(path, value)


@contextmanager
def integration():
    names = ('load_plan', 'audit_batch', 'exclusive', 'SCOPE', 'CONTROL_CODE')
    before = {name: getattr(core, name) for name in names}
    core.load_plan = load_plan
    core.audit_batch = audit_batch
    core.exclusive = write_metadata
    core.SCOPE = SCOPE
    core.CONTROL_CODE = core.CONTROL_CODE | {SELF, CONTRACT}
    try:
        yield core
    finally:
        for name, value in before.items(): setattr(core, name, value)


def run(root, protocol_ref, authority_ref, commit):
    with integration():
        try:
            return core.run(root, protocol_ref, authority_ref, commit)
        except RuntimeError as error:
            if str(error).startswith('Incomplete original51 batch;'):
                raise RuntimeError('Incomplete registered forward15 batch; preserve original evidence and register any continuation separately') from error
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'authority'):
        parser.add_argument('--'+name, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--publication-commit', required=True)
    args = parser.parse_args()
    run(ROOT, {'path': args.protocol, 'sha256': args.protocol_sha256},
        {'path': args.authority, 'sha256': args.authority_sha256}, args.publication_commit)
