"""Seal frozen4B mappings on complete native8B feature panels before target use.

CPU postprocessing only. This does not fit a model, sample a trajectory, admit
RL training or open a transfer panel. Future8B target controllers must verify
the published receipt, common-state identity and publication chronology.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

from bfcl_cross_model_prediction_values import feature_panels, predictions
from bfcl_qwen8_initial_probe import REVISION, audit_probe, check_binding
from seal_bfcl_transfer_predictions import record_publication

KIND = 'frozen4b_to_native8b_actual_point_prediction_seal'
FILES = {'original_five_models.json', 'h1_early_model.json', 'structural_early_models.json',
         'native8b_features.json', 'predictions.json', 'complete_probe_review.json'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def pinned(root, reference):
    path = root / reference['path']
    if digest(path) != reference['sha256']:
        raise ValueError('Pinned bytes changed: ' + reference['path'])
    return read(path)


def ref(root, path):
    return {'path': str(path.relative_to(root)), 'sha256': digest(path)}


def exclusive(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def require_complete_probe(batch, plan):
    names = [case['name'] for case in plan['cases']]
    if (batch.get('status') != 'passed' or batch.get('model_revision') != REVISION
            or batch.get('complete_cases') != 16 or len(names) != 16
            or len(set(names)) != 16 or [x['case'] for x in batch.get('rows', [])] != names):
        raise ValueError('Require the complete original16 native8B probe audits')


def source_models(root, protocol):
    early = pinned(root, protocol['original_five_receipt'])
    original_path = (root / protocol['original_five_receipt']['path']).parent / 'models.json'
    if digest(original_path) != early['files']['models.json']:
        raise ValueError('Original five model bytes differ from their original seal')
    h1_receipt = pinned(root, protocol['h1_early_receipt'])
    h1 = pinned(root, h1_receipt['model'])
    structural = pinned(root, protocol['structural_models'])
    structural_protocol = pinned(root, protocol['structural_protocol'])
    if structural['source_analysis_contract'] != protocol['analysis_contract']:
        raise ValueError('Structural model analysis differs')
    if early['analysis_contract_sha256'] != protocol['analysis_contract']['sha256']:
        raise ValueError('Original analysis differs')
    return read(original_path), h1, structural, {
        'original_five_models.json': original_path,
        'h1_early_model.json': root / h1_receipt['model']['path'],
        'structural_early_models.json': root / protocol['structural_models']['path']}


def native_target_record_check(root):
    """Check explicit project-native metadata; admission chronology is also mandatory.

    This is a scoped check of this project's tagged run records, not a claim
    about private processes or activities outside the registered project.
    """
    rows = []
    for path in sorted((root / 'rl').rglob('run.json')):
        value = read(path)
        if value.get('model_revision') != REVISION:
            continue
        if value.get('target_evaluation') is True:
            raise ValueError('Native8B target run already exists: ' + str(path.relative_to(root)))
        rows.append({'run': ref(root, path), 'explicit_target_evaluation': value.get('target_evaluation')})
    return {'tagged_native8b_run_records': rows, 'explicit_native8b_target_runs': 0,
            'scope': 'Project run.json records explicitly tagged with this8B revision; future target admission additionally requires verified publication before its actual start.'}


def seal(root, protocol_ref):
    protocol = pinned(root, protocol_ref)
    if protocol['kind'] != 'native8b_cross_size_frozen_mapping_seal_protocol':
        raise ValueError('Wrong point-seal protocol')
    for name, sha in protocol['implementation_sha256'].items():
        if digest(root / name) != sha:
            raise ValueError('Sealing implementation changed: ' + name)
    plan = pinned(root, protocol['probe_plan'])
    if plan['model_revision'] != REVISION or plan['stage'] != 0:
        raise ValueError('Require own native8B step0 features')
    batch_path = root / protocol['complete_probe_audit_path']
    if not batch_path.exists():
        raise ValueError('Require the complete original16 native8B probe audits')
    batch = read(batch_path)
    require_complete_probe(batch, plan)
    check_binding(plan)
    rows = []
    for case in plan['cases']:
        actual = audit_probe(plan, case, protocol['probe_plan']['sha256'])
        saved = read(root / plan['output_root'] / case['name'] / 'audit.json')
        if actual != saved:
            raise ValueError('Fresh raw replay differs from saved case audit: ' + case['name'])
        rows.append(actual)
    if rows != batch['rows']:
        raise ValueError('Batch and individual audit rows differ')
    access = native_target_record_check(root)
    analysis = pinned(root, protocol['analysis_contract'])
    five, h1, structural, model_paths = source_models(root, protocol)
    features = feature_panels(root, plan, protocol['probe_plan'], analysis,
                              protocol['analysis_contract']['sha256'])
    values = predictions(five, h1, structural, features, analysis)
    review = {'status': 'passed', 'model_revision': REVISION, 'complete_cases': 16,
              'raw_replay_equals_saved_audits': True, 'batch_audit': ref(root, batch_path),
              'case_audits': [ref(root, root / plan['output_root'] / c['name'] / 'audit.json') for c in plan['cases']],
              'validation_labels_audited_for_integrity': True,
              'validation_labels_used_to_modify_features_or_models': False,
              'target_labels_used': False, 'target_record_check': access}
    output = root / protocol['output_root']
    # All dependencies and full raw audits pass before any seal directory is made.
    output.mkdir(exist_ok=False)
    for name, source in model_paths.items():
        with (output / name).open('xb') as stream:
            stream.write(source.read_bytes())
    exclusive(output / 'native8b_features.json', features)
    exclusive(output / 'predictions.json', values)
    exclusive(output / 'complete_probe_review.json', review)
    receipt = {'kind': KIND, 'sealed_unix': time.time(), 'protocol': protocol_ref,
               'stage': 0, 'target_model_revision': REVISION,
               'native_common_binding': plan['common_binding'], 'probe_plan': protocol['probe_plan'],
               'analysis_contract': protocol['analysis_contract'],
               'original_five_receipt': protocol['original_five_receipt'],
               'h1_early_receipt': protocol['h1_early_receipt'],
               'structural_protocol': protocol['structural_protocol'], 'structural_models': protocol['structural_models'],
               'files': {name: digest(output / name) for name in sorted(FILES)},
               'implementation_sha256': protocol['implementation_sha256'],
               'all_source_groups': analysis['source_split']['all_groups'],
               'original_4b_fit_groups': analysis['source_split']['fit'],
               'original_4b_heldout_groups': analysis['source_split']['heldout'],
               'actual_point_prediction_count': 128, 'predicted_endpoint': 'G',
               'prediction_blocks': ['original_five', 'separate_H1', 'separate_structural'],
               'new_model_tokens': 0, 'optimizer_updates': 0, 'model_refitted': False,
               'standardization_refitted': False, 'prediction_clipping': False,
               'native8b_target_labels_used': False, 'reserved21_accessed': False,
               'publication_required_before_any_native8b_target_evaluation': True,
               'scope': protocol['scope'], 'goal_complete': False}
    exclusive(output / 'receipt.json', receipt)
    return ref(root, output / 'receipt.json')


def verify_chronology(receipt, publication, evaluation_started_unix=None):
    if receipt.get('publication_required_before_any_native8b_target_evaluation') is not True:
        raise ValueError('Cross-size publication must precede every native8B target role')
    if not receipt['sealed_unix'] <= publication['verified_remote_at_unix']:
        raise ValueError('Publication precedes local sealing')
    if evaluation_started_unix is not None and publication['verified_remote_at_unix'] > evaluation_started_unix:
        raise ValueError('Native8B target started before verified cross-size publication')


def verify_cross_size_seal(root, reference, common_binding_ref, analysis_ref,
                           *, publication_ref, check_remote=True, evaluation_started_unix=None):
    receipt = pinned(root, reference)
    if (receipt['kind'] != KIND or receipt['stage'] != 0 or receipt['target_model_revision'] != REVISION
            or receipt['native8b_target_labels_used'] is not False
            or receipt['actual_point_prediction_count'] != 128):
        raise ValueError('Wrong or incomplete native8B actual forecast seal')
    if receipt['native_common_binding'] != common_binding_ref or receipt['analysis_contract'] != analysis_ref:
        raise ValueError('Wrong native common state or analysis contract')
    protocol = pinned(root, receipt['protocol'])
    if receipt['probe_plan'] != protocol['probe_plan'] or receipt['analysis_contract'] != protocol['analysis_contract']:
        raise ValueError('Sealed provenance differs from the registered inputs')
    if receipt['implementation_sha256'] != protocol['implementation_sha256']:
        raise ValueError('Sealed implementation differs from registered code')
    for name, sha in receipt['implementation_sha256'].items():
        if digest(root / name) != sha:
            raise ValueError('Sealed implementation bytes changed: ' + name)
    directory = (root / reference['path']).parent
    if set(receipt['files']) != FILES:
        raise ValueError('Incomplete cross-size seal files')
    for name, sha in receipt['files'].items():
        if digest(directory / name) != sha:
            raise ValueError('Sealed point forecast bytes changed: ' + name)
    five, h1, structural, original_paths = source_models(root, protocol)
    for name, path in original_paths.items():
        if digest(directory / name) != digest(path):
            raise ValueError('Copied frozen model bytes changed: ' + name)
    plan = pinned(root, receipt['probe_plan'])
    if plan['common_binding'] != common_binding_ref:
        raise ValueError('Probe and seal native state differ')
    binding = pinned(root, common_binding_ref)
    if binding['model_revision'] != REVISION or binding['policy_fingerprint'] != plan['policy_fingerprint']:
        raise ValueError('Not the expected native8B common state')
    features = read(directory / 'native8b_features.json')
    if (features['model_revision'] != REVISION or features['stage'] != 0
            or features['source_protocol'] != receipt['probe_plan']
            or features['policy_fingerprint'] != plan['policy_fingerprint']):
        raise ValueError('Wrong native feature provenance')
    analysis = pinned(root, analysis_ref)
    if (receipt['all_source_groups'] != analysis['source_split']['all_groups']
            or receipt['original_4b_fit_groups'] != analysis['source_split']['fit']
            or receipt['original_4b_heldout_groups'] != analysis['source_split']['heldout']):
        raise ValueError('Source identities or historical split changed')
    expected = predictions(five, h1, structural, features, analysis)
    if expected != read(directory / 'predictions.json'):
        raise ValueError('Sealed actual points differ from the original fixed functions')
    if root / publication_ref['path'] != directory / 'publication.json':
        raise ValueError('Publication record belongs to another seal')
    publication = pinned(root, publication_ref)
    if publication['receipt_sha256'] != reference['sha256']:
        raise ValueError('Publication receipt SHA differs')
    verify_chronology(receipt, publication, evaluation_started_unix)
    files = {str((directory / name).relative_to(root)): sha for name, sha in receipt['files'].items()}
    files[reference['path']] = reference['sha256']
    for name, sha in files.items():
        result = subprocess.run(['git', 'show', publication['commit'] + ':' + name],
                                cwd=root, capture_output=True, timeout=30)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != sha:
            raise ValueError('Published commit does not contain sealed bytes: ' + name)
    if check_remote:
        from supervise_bfcl_stage1_probe import verify_publication
        verify_publication(root, files, publication['commit'])
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('seal')
    create.add_argument('--protocol', required=True)
    create.add_argument('--protocol-sha256', required=True)
    publish = commands.add_parser('record-publication')
    publish.add_argument('--receipt', required=True)
    publish.add_argument('--receipt-sha256', required=True)
    publish.add_argument('--commit', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == 'seal':
        result = seal(root, {'path': args.protocol, 'sha256': args.protocol_sha256})
    else:
        path = root / args.receipt
        if read(path)['kind'] != KIND:
            raise ValueError('Not a cross-size prediction seal')
        result = record_publication(root, path, args.receipt_sha256, args.commit)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
