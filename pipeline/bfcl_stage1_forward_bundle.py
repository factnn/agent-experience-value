"""Seal original early-five and separate H1 late forecasts before stage1 RL acquisition.

This is a versioned adapter, not a replacement for the running probe dependencies.
The target access component applies to common, fit, control AND heldout panels;
training completion/current evaluation authority/within-late heldout seals remain
additional requirements of the eventual stage1 target execution contract.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from bfcl_source_eval_contract import digest, pinned, verify_seal
from fit_bfcl_h1_source_success import source_success, predict as h1_predict
from seal_bfcl_forward_predictions import extract_features, feature_row, forward_artifact, verify_forward_seal
from supervise_bfcl_stage_fork import ROOT, read

SELF = 'pipeline/bfcl_stage1_forward_bundle.py'
FIVE_FILES = ('early_models.json', 'late_features.json', 'predictions.json')
BUNDLE_FILES = {'original_five/'+name for name in (*FIVE_FILES, 'receipt.json')} | {
    'h1_early_model.json', 'h1_late_features.json', 'h1_predictions.json'}


def reference(root, path):
    return {'path': str(path.resolve().relative_to(root.resolve())), 'sha256': digest(path)}


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()


def check_stop(root, definition, phase):
    names = ['global_stop_file', 'training_stop_file' if phase == 'acquisition' else 'evaluation_stop_file']
    for name in names:
        path = (root/definition[name]).resolve()
        if path.parent != (root/'experiments/.runtime').resolve():
            raise ValueError('Stop marker outside current project runtime')
        if path.exists():
            raise RuntimeError('Current SSH stop blocks stage1 '+phase+' admission')


def load_definition(root, definition_ref):
    definition = pinned(root, definition_ref)
    if definition['kind'] != 'stage1_original_five_and_separate_H1_forward_binding':
        raise ValueError('Wrong forward definition')
    if definition['all_late_target_roles_require_forward'] is not True:
        raise ValueError('All late target roles must require published forecasts')
    for name, value in definition['implementation_sha256'].items():
        if digest(root/name) != value:
            raise ValueError('Frozen forward implementation changed: '+name)
    if SELF not in definition['implementation_sha256']:
        raise ValueError('Unpinned forward adapter')
    master = pinned(root, definition['stage1_execution_protocol'])
    probe = pinned(root, definition['stage1_probe_protocol'])
    if (master['kind'] != 'source_stage1_training_and_transfer_execution' or master['stage'] != 1
            or master['stage1_probe_protocol'] != definition['stage1_probe_protocol']
            or master['parent_binding'] != probe['parent_binding']):
        raise ValueError('Wrong fixed stage1 lineage')
    parent = pinned(root, master['parent_binding'])
    early = pinned(root, master['stage0_execution_protocol'])
    analysis = pinned(root, master['analysis_contract'])
    if parent['parent_case'] != 's0_uniform_r0' or parent['source_step'] != 7:
        raise ValueError('Wrong prespecified learner state')
    if early['analysis_contract'] != master['analysis_contract'] or parent['stage0_execution_protocol'] != master['stage0_execution_protocol']:
        raise ValueError('Early/late analysis or ancestor changed')
    report = pinned(root, probe['stage0_transfer_report'])
    if definition['early_five_receipt'] != report['evidence']['prediction_seal']:
        raise ValueError('Require the original audited early-five seal')
    return definition, master, probe, parent, early, analysis


def early_inputs(root, definition, master, probe, early, analysis):
    five = verify_seal(root, definition['early_five_receipt'], master['stage0_execution_protocol']['sha256'],
                       master['analysis_contract']['sha256'], analysis['source_split']['heldout'], check_remote=False)
    startup = pinned(root, definition['probe_startup_receipt'])
    if startup['runtime_protocol'] != definition['stage1_probe_protocol']:
        raise ValueError('Wrong actual probe startup receipt')
    observed = pinned(root, startup['publication'])
    old_publication = read((root/definition['early_five_receipt']['path']).parent/'publication.json')
    # The actual observed remote commit is immutable historical evidence. A new
    # network query is required for NEW forecasts, not to re-prove this old event.
    result = subprocess.run(['git', 'merge-base', '--is-ancestor', old_publication['commit'], observed['remote_main']],
                            cwd=root, capture_output=True, timeout=30)
    if result.returncode or old_publication['verified_remote_at_unix'] > observed['verified_unix']:
        raise ValueError('Early-five publication not anchored before actual late probe admission')
    h1_receipt = pinned(root, definition['early_H1_receipt'])
    h1 = pinned(root, h1_receipt['model'])
    protocol = pinned(root, definition['H1_protocol'])
    if h1_receipt['protocol'] != definition['H1_protocol'] or h1['protocol'] != definition['H1_protocol']:
        raise ValueError('Wrong frozen auxiliary H1 model')
    if (protocol['analysis_contract'] != master['analysis_contract']
            or protocol['fit_groups'] != analysis['source_split']['fit']
            or protocol['future_test_groups'] != analysis['source_split']['heldout']):
        raise ValueError('H1 source split changed')
    model = h1['model']
    for name in ('intercept', 'slope', 'early_fit_mean', 'early_fit_population_std', 'lambda'):
        if type(model[name]) not in (int, float) or not math.isfinite(model[name]):
            raise ValueError('Nonfinite H1 model')
    if (model['lambda'] != 1 or model['early_fit_population_std'] < 0
            or model['fit_groups'] != protocol['fit_groups'] or model['future_test_groups'] != protocol['future_test_groups']
            or any(model[name] is not False for name in ('sign_constraint', 'feature_selection', 'prediction_clipping', 'standardization_refitted_on_late_features'))
            or h1['late_target_labels_used'] is not False or h1['local_validation_rewards_used'] is not False):
        raise ValueError('H1 fitting/normalization/label rules changed')
    if digest(root/h1_receipt['code_used_at_fit']['path']) != h1_receipt['code_used_at_fit']['sha256']:
        raise ValueError('Missing H1 fit implementation')
    return five, h1_receipt, h1


def require_complete_probe(root, definition, probe, review_ref):
    batch_path = root/probe['output_root']/'audit_status.json'
    if not batch_path.exists():
        raise ValueError('All16 late probe cases must be complete before sealing')
    batch = read(batch_path)
    if (batch['status'] != 'passed' or batch['stage'] != 1 or batch['complete_cases'] != 16
            or batch['all_16_groups_complete'] is not True or len(batch['rows']) != 16):
        raise ValueError('All16 late probe cases must pass before sealing')
    if not review_ref:
        raise ValueError('Require the completed probe execution-agent review')
    review = pinned(root, review_ref)
    if (review['kind'] != 'complete_stage1_probe_raw_execution_review' or review['status'] != 'reviewed'
            or review['probe_protocol'] != definition['stage1_probe_protocol']
            or review['complete_audit'] != reference(root, batch_path)
            or review['fresh_raw_audits_match_saved'] is not True):
        raise ValueError('Probe review does not bind the complete raw audit')
    return reference(root, batch_path)


def h1_features(root, plan, probe_ref):
    rows = []
    for case in plan['cases']:
        path = root/plan['output_root']/case['name']/'feature.jsonl'
        raw = [json.loads(line) for line in path.read_text().splitlines()]
        # Reuse all original task/seed/stage/policy/group identity checks. Never
        # read validation in feature construction or drop all-success/failure tasks.
        feature_row(raw, case, plan['policy_fingerprint'])
        rows.append({'source_group': case['source_group'], 'task_count': len(raw),
                     'source_success_rate': source_success(raw), 'feature_sha256': digest(path)})
    return {'stage': 1, 'source_protocol': probe_ref, 'parent_binding': plan['parent_binding'],
            'policy_fingerprint': plan['policy_fingerprint'], 'rows': rows,
            'validation_labels_used': False, 'late_target_labels_used': False}


def h1_predictions(h1, features, analysis):
    groups = analysis['source_split']['all_groups']
    rows = features['rows']
    if (features['stage'] != 1 or features['validation_labels_used'] is not False
            or features['late_target_labels_used'] is not False or len(rows) != 16
            or len({r['source_group'] for r in rows}) != 16 or set(r['source_group'] for r in rows) != set(groups)
            or any(not 0 <= r['source_success_rate'] <= 1 for r in rows)):
        raise ValueError('Require all16 feature-only source success measurements')
    values = {r['source_group']: r['source_success_rate'] for r in rows}
    return {'kind': 'separate_early_H1_to_late_G_predictions', 'stage': 1, 'fit_stage': 0, 'target': 'G',
            'heldout_groups': analysis['source_split']['heldout'],
            'point_predictions': {g: h1_predict(h1['model'], values[g]) for g in analysis['source_split']['heldout']},
            'late_target_labels_used': False, 'refit': False, 'renormalized': False,
            'feature_selection': False, 'prediction_clipping': False,
            'scope': 'Separately registered auxiliary H1; never a sixth original stage0 confirmatory model.'}


def seal(root, definition_ref, review_ref, output):
    root, output = root.resolve(), output.resolve()
    definition, master, probe, parent, early, analysis = load_definition(root, definition_ref)
    check_stop(root, definition, 'acquisition')
    if output != root/definition['output_root'] or output.exists():
        raise ValueError('Require the unused registered forward output path')
    # Refuse partial panels before any output directory or heavy parent replay.
    batch_ref = require_complete_probe(root, definition, probe, review_ref)
    if (root/master['output_root']).exists() or any((root/name).exists() for name in definition['late_target_output_roots']):
        raise ValueError('Forward sealing must precede new late acquisition and all target output')
    from bfcl_stage1_probe_ssh import load_protocol
    plan, binding = load_protocol(root, root/definition['stage1_probe_protocol']['path'], definition['stage1_probe_protocol']['sha256'])
    five, h1_receipt, h1 = early_inputs(root, definition, master, probe, early, analysis)
    late = extract_features(root, plan, definition['stage1_probe_protocol'], master['analysis_contract'])
    early_dir = (root/definition['early_five_receipt']['path']).parent
    five_prediction = forward_artifact(read(early_dir/'models.json'), late, analysis, early['source_stage0']['policy_fingerprint'])
    h1_late = h1_features(root, plan, definition['stage1_probe_protocol'])
    h1_prediction = h1_predictions(h1, h1_late, analysis)
    now = time.time()
    blobs = {'original_five/early_models.json': (early_dir/'models.json').read_bytes(),
             'original_five/late_features.json': encode(late), 'original_five/predictions.json': encode(five_prediction),
             'h1_early_model.json': (root/h1_receipt['model']['path']).read_bytes(),
             'h1_late_features.json': encode(h1_late), 'h1_predictions.json': encode(h1_prediction)}
    hashes = lambda: {name: hashlib.sha256(value).hexdigest() for name, value in blobs.items()}
    five_receipt = {'kind': 'early_to_late_prediction_seal', 'stage': 1, 'sealed_unix': now,
        'early_prediction_seal': definition['early_five_receipt'], 'stage0_execution_protocol': binding['stage0_execution_protocol'],
        'stage1_probe_protocol': definition['stage1_probe_protocol'], 'analysis_contract': master['analysis_contract'],
        'parent_binding': plan['parent_binding'], 'heldout_groups': analysis['source_split']['heldout'],
        'files': {name: hashes()['original_five/'+name] for name in FIVE_FILES},
        'implementation_sha256': definition['implementation_sha256'],
        'publication_required_before_any_stage1_target_evaluation': True, 'late_target_labels_used': False, 'new_sampled_tokens': 0}
    blobs['original_five/receipt.json'] = encode(five_receipt)
    receipt = {'kind': 'stage1_original_five_and_separate_H1_forward_seal', 'stage': 1, 'sealed_unix': now,
        'definition': definition_ref, 'stage1_execution_protocol': definition['stage1_execution_protocol'],
        'stage1_probe_protocol': definition['stage1_probe_protocol'], 'complete_probe_audit': batch_ref, 'complete_probe_review': review_ref,
        'early_five_receipt': definition['early_five_receipt'], 'early_H1_receipt': definition['early_H1_receipt'],
        'original_five_receipt': {'path': str((output/'original_five/receipt.json').relative_to(root)), 'sha256': hashes()['original_five/receipt.json']},
        'files': hashes(), 'late_target_labels_used': False, 'stage1_training_experience_acquired_before_seal': False,
        'publication_required_before_any_stage1_target_evaluation': True, 'publication_required_before_stage1_acquisition': True,
        'new_scientific_tokens': 0, 'goal_complete': False}
    blobs['receipt.json'] = encode(receipt)
    check_stop(root, definition, 'acquisition')
    output.mkdir(exist_ok=False)
    (output/'original_five').mkdir()
    for name, value in blobs.items():
        with (output/name).open('xb') as handle:
            handle.write(value)
    return reference(root, output/'receipt.json')


def verify_contents(root, receipt_ref):
    receipt = pinned(root, receipt_ref)
    if (receipt['kind'] != 'stage1_original_five_and_separate_H1_forward_seal' or receipt['stage'] != 1
            or receipt['late_target_labels_used'] is not False or receipt['stage1_training_experience_acquired_before_seal'] is not False
            or receipt['publication_required_before_any_stage1_target_evaluation'] is not True
            or receipt['publication_required_before_stage1_acquisition'] is not True or set(receipt['files']) != BUNDLE_FILES):
        raise ValueError('Incomplete or wrong-stage forward bundle')
    definition, master, probe, parent, early, analysis = load_definition(root, receipt['definition'])
    for key in ('stage1_execution_protocol', 'stage1_probe_protocol', 'early_five_receipt', 'early_H1_receipt'):
        if receipt[key] != definition[key]: raise ValueError('Forward bundle binding changed: '+key)
    if require_complete_probe(root, definition, probe, receipt['complete_probe_review']) != receipt['complete_probe_audit']:
        raise ValueError('Forward complete-probe provenance changed')
    five, h1_receipt, h1 = early_inputs(root, definition, master, probe, early, analysis)
    directory = (root/receipt_ref['path']).parent
    if directory != root/definition['output_root']:
        raise ValueError('Forward bundle moved from its registered path')
    for name, value in receipt['files'].items():
        if digest(directory/name) != value: raise ValueError('Forward sealed bytes changed: '+name)
    if reference(root, directory/'original_five/receipt.json') != receipt['original_five_receipt']:
        raise ValueError('Original-five nested receipt mismatch')
    late = read(directory/'original_five/late_features.json')
    h1_late = read(directory/'h1_late_features.json')
    for item in (late, h1_late):
        if (item['source_protocol'] != definition['stage1_probe_protocol'] or item['parent_binding'] != master['parent_binding']
                or item['policy_fingerprint'] != parent['policy_fingerprint']):
            raise ValueError('Late feature source/ancestor changed')
    if (late['analysis_contract_sha256'] != master['analysis_contract']['sha256']
            or receipt['files']['original_five/early_models.json'] != five['files']['models.json']
            or receipt['files']['h1_early_model.json'] != h1_receipt['model']['sha256']):
        raise ValueError('Early model bytes or analysis changed')
    if (forward_artifact(read(directory/'original_five/early_models.json'), late, analysis, early['source_stage0']['policy_fingerprint'])
            != read(directory/'original_five/predictions.json') or h1_predictions(h1, h1_late, analysis) != read(directory/'h1_predictions.json')):
        raise ValueError('Forward predictions differ from original fixed early mappings')
    five_features = {r['source_group']: (r['task_count'], r['feature_sha256']) for r in late['rows']}
    if five_features != {r['source_group']: (r['task_count'], r['feature_sha256']) for r in h1_late['rows']}:
        raise ValueError('Original-five and H1 must use identical complete feature task panels')
    fit_time = datetime.datetime.fromisoformat(h1['fitted_at']).timestamp()
    review = pinned(root, receipt['complete_probe_review'])
    if not five['sealed_unix'] <= fit_time <= receipt['sealed_unix'] or datetime.datetime.fromisoformat(review['reviewed_at']).timestamp() > receipt['sealed_unix']:
        raise ValueError('Forward model/probe review chronology changed')
    return receipt, definition, master


def publication_files(root, receipt_ref, receipt, definition):
    directory = (root/receipt_ref['path']).parent
    files = {str((directory/name).relative_to(root)): value for name, value in receipt['files'].items()}
    refs = [receipt_ref, receipt['definition'], receipt['complete_probe_audit'], receipt['complete_probe_review'],
            *[definition[k] for k in ('stage1_execution_protocol', 'stage1_probe_protocol', 'early_five_receipt', 'early_H1_receipt', 'H1_protocol', 'probe_startup_receipt')]]
    h1_receipt = pinned(root, definition['early_H1_receipt'])
    refs += [h1_receipt['model'], h1_receipt['code_used_at_fit']]
    files.update({ref['path']: ref['sha256'] for ref in refs})
    files.update(definition['implementation_sha256'])
    return files


def record_publication(root, receipt_ref, commit):
    receipt, definition, master = verify_contents(root, receipt_ref)
    check = lambda: check_stop(root, definition, 'acquisition')
    check()
    directory = (root/receipt_ref['path']).parent
    paths = [directory/'publication.json', directory/'original_five/publication.json']
    if any(path.exists() for path in paths):
        raise FileExistsError('Preserve existing publication receipts')
    from bfcl_stage1_probe_ssh import verify_publication
    proof = verify_publication(root, publication_files(root, receipt_ref, receipt, definition), commit, check)
    publication = proof | {'receipt_sha256': receipt_ref['sha256'], 'verified_remote_at_unix': proof['verified_unix'],
                           'kind': 'stage1_forward_bundle_actual_GitHub_publication'}
    if proof['verified_unix'] < receipt['sealed_unix']:
        raise ValueError('Publication preceded sealing')
    nested = publication | {'receipt_sha256': receipt['original_five_receipt']['sha256']}
    for path, value in zip(paths, (publication, nested)):
        with path.open('xb') as handle: handle.write(encode(value))
    return publication


def verify_bundle(root, receipt_ref, expected_protocol, use_started_unix=None):
    receipt, definition, master = verify_contents(root, receipt_ref)
    if receipt['stage1_execution_protocol'] != expected_protocol:
        raise ValueError('Forward bundle belongs to another intervention matrix')
    directory = (root/receipt_ref['path']).parent
    publication = read(directory/'publication.json')
    files = publication_files(root, receipt_ref, receipt, definition)
    usage = time.time() if use_started_unix is None else use_started_unix
    validate_publication(publication, receipt_ref, receipt, files, usage)
    for name, value in files.items():
        result = subprocess.run(['git', 'show', publication['commit']+':'+name], cwd=root, capture_output=True, timeout=30)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != value:
            raise ValueError('Published forward commit does not contain exact bytes: '+name)
    verify_forward_seal(root, receipt['original_five_receipt'], definition['stage1_probe_protocol'], master['parent_binding'],
                        master['analysis_contract'], check_remote=False, evaluation_started_unix=usage)
    return {'status': 'published_forward_five_and_separate_H1_verified', 'receipt': receipt_ref,
            'publication_commit': publication['commit'], 'verified_remote_at_unix': publication['verified_remote_at_unix']}


def validate_publication(publication, receipt_ref, receipt, files, usage):
    if (publication['kind'] != 'stage1_forward_bundle_actual_GitHub_publication'
            or publication['receipt_sha256'] != receipt_ref['sha256'] or publication['files'] != files
            or not all(type(v) in (int, float) and math.isfinite(v) for v in
                       (receipt['sealed_unix'], publication['verified_remote_at_unix'], publication['verified_unix'], usage))
            or not receipt['sealed_unix'] <= publication['verified_remote_at_unix'] == publication['verified_unix'] <= usage):
        raise ValueError('Forward actual publication missing, changed or later than scientific use')


def verify_forward_target_gate(root, job, master, evaluation_started_unix=None):
    # No role exemption. This function must be called before target output creation.
    if master['stage'] != 1 or master['kind'] != 'source_stage1_training_and_transfer_execution':
        raise ValueError('Require the actual stage1 target protocol')
    if not job.get('forward_bundle'):
        raise ValueError('Every late target panel, including common/fit, requires published original-five and H1 forecasts')
    proof = verify_bundle(root, job['forward_bundle'], job['execution_protocol'], evaluation_started_unix)
    receipt = pinned(root, job['forward_bundle'])
    definition = pinned(root, receipt['definition'])
    if pinned(root, job['execution_protocol']) != master:
        raise ValueError('Target job protocol differs from its actual registered master')
    check_stop(root, definition, 'evaluation')
    return proof | {'target_role': job['role'], 'all_roles_checked': True,
                    'scope': 'Forward access prerequisite only. Complete51/current eval authority and within-late heldout seal remain required.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('seal')
    for name in ('definition', 'probe-review'):
        p.add_argument('--'+name, required=True)
        p.add_argument('--'+name+'-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('record-publication')
    p.add_argument('--receipt', required=True); p.add_argument('--receipt-sha256', required=True); p.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.command == 'seal':
        result = seal(ROOT, {'path': args.definition, 'sha256': args.definition_sha256},
                      {'path': args.probe_review, 'sha256': args.probe_review_sha256}, args.output)
    else:
        result = record_publication(ROOT, {'path': args.receipt, 'sha256': args.receipt_sha256}, args.commit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
