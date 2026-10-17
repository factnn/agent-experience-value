"""Fit the two E368 exploratory4B rules and seal prospective native8B points."""
import argparse
import hashlib
import math
from pathlib import Path
import subprocess
import time

from bfcl_model_conditioned_contract import SIGNALS, digest, exclusive, identity_ref, load_execution, pinned, read, ref
from fit_bfcl_structural_candidates import fit_one
from seal_bfcl_qwen8_cross_size_predictions import native_target_record_check, verify_chronology
from seal_bfcl_transfer_predictions import record_publication

KIND = 'E368_frozen4b_to_native8b_actual_auxiliary_prediction_seal'
FILES = {'frozen4b_models.json', 'features.json', 'predictions.json'}


def fit_and_predict(design, features, analysis, old_report):
    if (features['feature_phase_only'] is not True or features['target_labels_used'] is not False
            or features['reward_or_grade_filtering'] is not False or features['model_updates'] != 0):
        raise ValueError('Require the complete presampling-only forward features')
    if old_report['status'] != 'passed' or old_report['labels']['panel_count'] != 104:
        raise ValueError('Require the previously complete original4B transfer report')
    groups, fit = analysis['source_split']['all_groups'], analysis['source_split']['fit']
    held = analysis['source_split']['heldout']
    if len(fit) != 12 or len(held) != 4 or len(groups) != 16:
        raise ValueError('Original source split changed')
    for profile in ['qwen4b', 'qwen8b']:
        if set(features['groups'][profile]) != set(groups):
            raise ValueError('Incomplete/changed source profile')
        if any(set(features['groups'][profile][g]) != set(SIGNALS) for g in groups):
            raise ValueError('Missing/selected scientific feature')
    if features['profiles'] != design['profiles']:
        raise ValueError('Wrong learner/ancestor feature identities')
    labels = {group: old_report['labels']['G_labels'][group] for group in fit}
    models = {signal: fit_one([features['groups']['qwen4b'][g][signal] for g in fit],
                              [labels[g] for g in fit]) for signal in SIGNALS}
    registry = {'kind': 'E368_exploratory4b_fitted_rules_for_prospective8b', 'models': models,
        'fit_groups': fit, 'historical_heldout_groups': held, 'fit_G_labels': labels,
        'fit_policy_fingerprint': design['profiles']['qwen4b']['policy_fingerprint'],
        'fit_stage': 0, 'lambda': 1., 'feature_selection': False, 'hyperparameter_search': False,
        'prediction_clipping': False, 'sign_constraint': False, 'native8b_normalization_refitted': False,
        'old4B_labels_were_already_open': True, 'historical4b_scores_are_prospective_validation': False}
    points, standardized = {}, []
    for group in groups:
        points[group] = {}
        for signal in SIGNALS:
            model, value = models[signal], features['groups']['qwen8b'][group][signal]
            scale = model['fit_population_std']
            z = (value - model['fit_mean']) / scale if scale else 0.
            prediction = model['intercept'] + model['slope'] * z
            if not math.isfinite(prediction):
                raise ValueError('Nonfinite actual point forecast')
            points[group][signal] = prediction
            standardized.append({'source_group': group, 'signal': signal, 'raw_native8b_feature': value,
                'frozen4b_fit_mean': model['fit_mean'], 'frozen4b_population_std': scale, 'z': z,
                'intercept': model['intercept'], 'slope': model['slope'], 'lambda': 1.,
                'predicted_G': prediction, 'historical4b_source_role': 'fit' if group in fit else 'heldout'})
    predictions = {'kind': 'E368_actual_auxiliary_cross_size_G_points', 'target': 'G', 'points': points,
        'standardized_source_rows': standardized, 'all_source_groups': groups,
        'target_model_revision': design['profiles']['qwen8b']['base_revision'],
        'target_policy_fingerprint': design['profiles']['qwen8b']['policy_fingerprint'],
        'model_refitted_on8b_labels': False, 'standardization_refitted_on8b_features': False,
        'prediction_clipping': False, 'native8b_target_labels_used': False,
        'scope': 'Separate two-proxy auxiliary block. Cross-size tests on original4B fit12 and heldout4 must be distinguished; not16 previously unseen sources or new target-task generalization.'}
    return registry, predictions


def seal(root, execution_ref, output):
    execution, design = load_execution(root, execution_ref)
    execution_ref = identity_ref(execution_ref)
    audit_path = root / design['output_root'] / 'audit_status.json'
    if not audit_path.exists():
        raise ValueError('Require all32 completed E368 forward-measurement audits')
    audit = read(audit_path)
    if (audit['status'] != 'passed' or audit['complete_cases'] != 32 or audit['execution'] != execution_ref
            or [r['case'] for r in audit['rows']] != [c['name'] for c in design['cases']]):
        raise ValueError('Wrong/incomplete E368 raw probability audit')
    features = pinned(root, audit['features'])
    if features['execution'] != execution_ref or features['design'] != execution['design']:
        raise ValueError('Feature execution provenance changed')
    analysis_ref = design['fit_rule']['analysis_contract']
    report_ref = design['fit_rule']['audited4b_report']
    models, values = fit_and_predict(design, features, pinned(root, analysis_ref), pinned(root, report_ref))
    access = native_target_record_check(root)
    output.mkdir(exist_ok=False)
    with (output / 'features.json').open('xb') as stream:
        stream.write((root / audit['features']['path']).read_bytes())
    exclusive(output / 'frozen4b_models.json', models)
    exclusive(output / 'predictions.json', values)
    receipt = {'kind': KIND, 'sealed_unix': time.time(), 'design': execution['design'],
        'execution': execution_ref, 'complete_measurement_audit': ref(root, audit_path),
        'native_common_binding': design['profiles']['qwen8b']['native_common_binding'],
        'target_model_revision': design['profiles']['qwen8b']['base_revision'],
        'analysis_contract': analysis_ref, 'audited4b_fit_label_report': report_ref,
        'files': {name: digest(output / name) for name in sorted(FILES)},
        'actual_point_prediction_count': 32, 'native8b_target_labels_used': False,
        'publication_required_before_any_native8b_target_evaluation': True,
        'new_model_tokens': 0, 'optimizer_updates': 0, 'reserved21_accessed': False,
        'target_record_check': access,
        'scope': 'E368 two-model-conditioned-proxy forecasts, kept separate from E367 original128 and any future within8B fits; old4B association is exploratory.',
        'goal_complete': False}
    exclusive(output / 'receipt.json', receipt)
    return ref(root, output / 'receipt.json')


def verify_auxiliary_seal(root, receipt_ref, publication_ref, common_binding_ref, analysis_ref,
                          *, check_remote=True, evaluation_started_unix=None):
    receipt = pinned(root, receipt_ref)
    if (receipt['kind'] != KIND or receipt['actual_point_prediction_count'] != 32
            or receipt['native8b_target_labels_used'] is not False
            or receipt['native_common_binding'] != common_binding_ref or receipt['analysis_contract'] != analysis_ref):
        raise ValueError('Wrong actual auxiliary prediction seal or native parent/split')
    execution, design = load_execution(root, receipt['execution'])
    if receipt['design'] != execution['design'] or receipt['analysis_contract'] != design['fit_rule']['analysis_contract']:
        raise ValueError('Wrong registered auxiliary definition')
    if (receipt['native_common_binding'] != design['profiles']['qwen8b']['native_common_binding']
            or receipt['target_model_revision'] != design['profiles']['qwen8b']['base_revision']
            or receipt['audited4b_fit_label_report'] != design['fit_rule']['audited4b_report']):
        raise ValueError('Auxiliary model/label lineage differs from its registered definition')
    directory = (root / receipt_ref['path']).parent
    if set(receipt['files']) != FILES:
        raise ValueError('Incomplete auxiliary point seal')
    for name, sha in receipt['files'].items():
        if digest(directory / name) != sha:
            raise ValueError('Sealed auxiliary bytes changed: ' + name)
    features = read(directory / 'features.json')
    audit = pinned(root, receipt['complete_measurement_audit'])
    if (audit['status'] != 'passed' or audit['complete_cases'] != 32
            or audit['features']['sha256'] != receipt['files']['features.json']):
        raise ValueError('Sealed features differ from complete raw measurement audit')
    models, values = fit_and_predict(design, features, pinned(root, analysis_ref),
                                    pinned(root, receipt['audited4b_fit_label_report']))
    if models != read(directory / 'frozen4b_models.json') or values != read(directory / 'predictions.json'):
        raise ValueError('Auxiliary frozen coefficients/standardization/actual points differ')
    if root / publication_ref['path'] != directory / 'publication.json':
        raise ValueError('Wrong auxiliary publication record')
    publication = pinned(root, publication_ref)
    if publication['receipt_sha256'] != receipt_ref['sha256']:
        raise ValueError('Publication refers to different actual auxiliary points')
    verify_chronology(receipt, publication, evaluation_started_unix)
    files = {str((directory / name).relative_to(root)): sha for name, sha in receipt['files'].items()}
    files[receipt_ref['path']] = receipt_ref['sha256']
    for name, sha in files.items():
        result = subprocess.run(['git', 'show', publication['commit'] + ':' + name],
                                 cwd=root, capture_output=True, timeout=30)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != sha:
            raise ValueError('Published commit lacks exact auxiliary bytes: ' + name)
    if check_remote:
        from supervise_bfcl_stage1_probe import verify_publication
        verify_publication(root, files, publication['commit'])
    return receipt


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('seal')
    create.add_argument('--execution', required=True)
    create.add_argument('--execution-sha256', required=True)
    create.add_argument('--output', required=True)
    publish = commands.add_parser('record-publication')
    publish.add_argument('--receipt', required=True)
    publish.add_argument('--receipt-sha256', required=True)
    publish.add_argument('--commit', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == 'seal':
        value = seal(root, {'path': args.execution, 'sha256': args.execution_sha256}, root / args.output)
    else:
        path = root / args.receipt
        if read(path)['kind'] != KIND:
            raise ValueError('Not an E368 auxiliary actual-point seal')
        value = record_publication(root, path, args.receipt_sha256, args.commit)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    main()
