"""CPU guard checks; synthetic metadata only, no native8B predictions or GPU work."""
import copy
import json
from pathlib import Path
import tempfile

from seal_bfcl_qwen8_cross_size_predictions import (
    REVISION, digest, native_target_record_check, pinned, require_complete_probe, verify_chronology)


def rejects(callback):
    try:
        callback()
    except ValueError:
        return
    raise AssertionError('Invalid input was accepted')


def main():
    names = ['cpu_fixture_' + str(i) for i in range(16)]
    plan = {'cases': [{'name': name} for name in names]}
    complete = {'status': 'passed', 'model_revision': REVISION, 'complete_cases': 16,
                'rows': [{'case': name} for name in names]}
    require_complete_probe(complete, plan)
    mutations = [lambda x: x.update(status='incomplete'), lambda x: x.update(model_revision='4B'),
                 lambda x: x.update(complete_cases=15), lambda x: x['rows'].reverse(),
                 lambda x: x['rows'].pop()]
    for change in mutations:
        bad = copy.deepcopy(complete)
        change(bad)
        rejects(lambda: require_complete_probe(bad, plan))
    receipt = {'sealed_unix': 10., 'publication_required_before_any_native8b_target_evaluation': True}
    verify_chronology(receipt, {'verified_remote_at_unix': 20.}, 30.)
    rejects(lambda: verify_chronology(receipt, {'verified_remote_at_unix': 5.}, 30.))
    rejects(lambda: verify_chronology(receipt, {'verified_remote_at_unix': 20.}, 15.))
    rejects(lambda: verify_chronology(receipt | {'publication_required_before_any_native8b_target_evaluation': False},
                                       {'verified_remote_at_unix': 20.}, 30.))
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        path = root / 'value.json'
        path.write_text('{"cpu_fixture":true}\n')
        reference = {'path': 'value.json', 'sha256': digest(path)}
        assert pinned(root, reference) == {'cpu_fixture': True}
        path.write_text('{"cpu_fixture":false}\n')
        rejects(lambda: pinned(root, reference))
        old = root / 'rl/old4B'
        old.mkdir(parents=True)
        (old / 'run.json').write_text(json.dumps({'model_revision': 'old4B', 'target_evaluation': True}))
        native = root / 'rl/native8B'
        native.mkdir()
        (native / 'run.json').write_text(json.dumps({'model_revision': REVISION, 'target_evaluation': False}))
        result = native_target_record_check(root)
        assert result['explicit_native8b_target_runs'] == 0 and len(result['tagged_native8b_run_records']) == 1
        (native / 'run.json').write_text(json.dumps({'model_revision': REVISION, 'target_evaluation': True}))
        rejects(lambda: native_target_record_check(root))
    print(json.dumps({'status': 'passed', 'complete_probe_mutations_rejected': 5,
                      'publication_chronology_mutations_rejected': 3, 'changed_pinned_bytes_rejected': True,
                      'native_target_record_rejected_old4B_record_allowed': True,
                      'fixtures': 'Synthetic guard metadata only; no actual8B feature values or point predictions',
                      'new_model_tokens': 0, 'gpu_jobs': 0}))


if __name__ == '__main__':
    main()
