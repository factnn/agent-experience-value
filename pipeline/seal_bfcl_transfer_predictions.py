"""Re-audit the complete fit panel, then seal actual fixed-model predictions.

No GPU work and no network mutation. Publish the resulting four files through
Git, then use record-publication to verify that exact commit on remote main.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from bfcl_source_eval_contract import digest, pinned
from bfcl_transfer_labels import source_registry, aggregate_labels
from bfcl_transfer_predictor import fit_models, predict_models
from report_bfcl_source_eval import audit_job
from supervise_bfcl_stage_fork import ROOT, read, write


def fit_artifacts(master, analysis, features, labels):
    if labels['phase'] != 'fit' or labels['panel_count'] != 80 or labels['stage'] != 0:
        raise ValueError('Require all 80 pre-seal panels')
    if features['stage'] != 0 or features['policy_fingerprint'] != master['source_stage0']['policy_fingerprint']:
        raise ValueError('Feature policy differs from the common stage0 ancestor')
    if features['analysis_contract_sha256'] != master['analysis_contract']['sha256']:
        raise ValueError('Features belong to a different analysis contract')
    models = fit_models(features, labels['G_labels'], analysis)
    predictions = predict_models(models, features, analysis)
    return models, {'stage': 0, 'heldout_groups': analysis['source_split']['heldout'],
                    'point_predictions': {g: predictions[g] for g in analysis['source_split']['heldout']},
                    'target': 'G', 'fit_labels_used': analysis['source_split']['fit'],
                    'heldout_labels_used': False, 'feature_selection': False}


def seal(root, manifest_path, manifest_sha, output):
    manifest = pinned(root, {'path': str(manifest_path.relative_to(root)), 'sha256': manifest_sha})
    if manifest['phase'] != 'fit':
        raise ValueError('Only the fit phase can produce the initial prediction seal')
    master = pinned(root, manifest['execution_protocol'])
    analysis = pinned(root, master['analysis_contract'])
    features = pinned(root, master['features'])
    training = [pinned(root, entry) for entry in master['jobs']]
    registry = source_registry(master, training, analysis)
    # Check the entire manifest before opening any target outcome, including accidental heldout entries.
    jobs, seen = [], set()
    for entry in manifest['jobs']:
        job = pinned(root, entry)
        case = job['policy_case']
        if job['execution_protocol'] != manifest['execution_protocol'] or case not in registry:
            raise ValueError('Evaluation execution identity changed')
        if registry[case]['role'] == 'heldout' or job['role'] != registry[case]['role']:
            raise ValueError('Heldout target labels cannot enter the fit seal')
        key = case, job['evaluation_repeat']
        if type(key[1]) is not int or key[1] not in (0, 1) or key in seen:
            raise ValueError('Duplicate or invalid evaluation panel')
        seen.add(key)
        jobs.append((entry, job))
    expected = {(c, e) for c, row in registry.items() if row['role'] != 'heldout' for e in range(2)}
    if seen != expected:
        raise ValueError('Fit manifest must contain all 80 planned panels')
    panels, evidence = [], []
    for entry, job in jobs:
        panel = audit_job(root, root / entry['path'])
        saved = read(root / job['output'] / 'audit.json')
        if panel != saved:
            raise ValueError('Fresh raw-panel audit differs from saved audit')
        panels.append(panel)
        evidence.append({'job': entry, 'audit_sha256': digest(root / job['output'] / 'audit.json'),
                         'source_files': panel['source_files'], 'cost': panel['cost']})
    labels = aggregate_labels(panels, registry, analysis, 'fit')
    labels['evidence'] = evidence
    models, predictions = fit_artifacts(master, analysis, features, labels)
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('models.json', models), ('predictions.json', predictions), ('fit_labels.json', labels)]:
        write(output / name, value)
    receipt = {'stage': 0, 'sealed_unix': time.time(), 'execution_protocol_sha256': manifest['execution_protocol']['sha256'],
               'analysis_contract_sha256': master['analysis_contract']['sha256'], 'features': master['features'],
               'fit_evaluation_manifest': {'path': str(manifest_path.relative_to(root)), 'sha256': manifest_sha},
               'heldout_groups': analysis['source_split']['heldout'],
               'files': {n: digest(output / n) for n in ['models.json', 'predictions.json', 'fit_labels.json']},
               'implementation_sha256': {str(p.relative_to(root)): digest(p) for p in [
                   root / 'pipeline/bfcl_transfer_labels.py', root / 'pipeline/bfcl_transfer_predictor.py',
                   root / 'pipeline/seal_bfcl_transfer_predictions.py']},
               'publication_required_before_heldout_evaluation': True}
    write(output / 'receipt.json', receipt)
    return receipt


def record_publication(root, receipt_path, receipt_sha, commit):
    if digest(receipt_path) != receipt_sha:
        raise ValueError('Receipt changed')
    receipt = read(receipt_path)
    directory = receipt_path.parent
    if (directory / 'publication.json').exists():
        raise ValueError('Publication receipt already exists; do not overwrite evidence')
    for name, sha in (receipt['files'] | {'receipt.json': receipt_sha}).items():
        path = directory / name
        if digest(path) != sha:
            raise ValueError('Sealed bytes changed')
        result = subprocess.run(['git', 'show', commit + ':' + str(path.relative_to(root))], cwd=root, capture_output=True)
        if result.returncode or hashlib.sha256(result.stdout).hexdigest() != sha:
            raise ValueError('Specified Git commit does not contain the exact sealed bytes')
    result = subprocess.run(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=root, capture_output=True, timeout=30)
    if result.returncode or not result.stdout.strip():
        raise RuntimeError('Remote main verification unavailable')
    remote = result.stdout.decode().split()[0]
    result = subprocess.run(['git', 'merge-base', '--is-ancestor', commit, remote], cwd=root, capture_output=True)
    if result.returncode:
        raise ValueError('Sealed commit is not a locally verifiable ancestor of remote main')
    publication = {'receipt_sha256': receipt_sha, 'commit': commit, 'remote_main': remote,
                   'verified_remote_at_unix': time.time()}
    if publication['verified_remote_at_unix'] < receipt['sealed_unix']:
        raise ValueError('Publication clock precedes sealing')
    write(directory / 'publication.json', publication)
    return publication


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    sealing = commands.add_parser('seal')
    sealing.add_argument('--manifest', type=Path, required=True)
    sealing.add_argument('--manifest-sha256', required=True)
    sealing.add_argument('--output', type=Path, required=True)
    publication = commands.add_parser('record-publication')
    publication.add_argument('--receipt', type=Path, required=True)
    publication.add_argument('--receipt-sha256', required=True)
    publication.add_argument('--commit', required=True)
    args = parser.parse_args()
    if args.command == 'seal':
        result = seal(ROOT, args.manifest.resolve(), args.manifest_sha256, args.output.resolve())
    else:
        result = record_publication(ROOT, args.receipt.resolve(), args.receipt_sha256, args.commit)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
