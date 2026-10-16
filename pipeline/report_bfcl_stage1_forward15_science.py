"""Re-audit all15/32 raw evidence and report frozen early-to-late predictions.

This program performs no inference/fitting and never writes original audits.
It refuses incomplete matrices and creates a separate final report exclusively.
"""
import argparse
from pathlib import Path
import datetime
import json

from bfcl_source_eval_contract import digest, pinned
from bfcl_stage1_forward32_ssh import load_manifest
from bfcl_stage1_forward_bundle import verify_bundle
from bfcl_stage1_forward15_analysis import registry, labels, score, stability
from bfcl_transfer_predictor import SIGNALS
from report_bfcl_stage1_forward32 import audit_job as audit_target
from report_bfcl_source_training import audit_job as audit_training
from supervise_bfcl_stage_fork import ROOT, read


def ref(root, path):
    return {'path': str(path.relative_to(root)), 'sha256': digest(path)}


def build(root, manifest_ref):
    manifest, master, binding, jobs = load_manifest(root, manifest_ref)
    target_certificate = read(root/manifest['output_root']/'audit_status.json')
    training_certificate = pinned(root, binding['complete_training_audit'])
    if (target_certificate['status'] != 'passed' or target_certificate['complete_panels'] != 32
            or target_certificate['manifest'] != manifest_ref or target_certificate['original_full104_complete'] is not False):
        raise ValueError('All original32 target panels must be complete first')
    training_jobs = [pinned(root, entry) for entry in master['jobs']]
    training_audits = []
    for entry, job in zip(master['jobs'], training_jobs):
        fresh = audit_training(job, entry['sha256'])
        if fresh != read(root/job['output']/'audit.json'):
            raise ValueError('Fresh original training raw replay changed: '+entry['name'])
        training_audits.append(fresh)
    if training_certificate['jobs'] != training_audits:
        raise ValueError('Complete15 raw training certificate differs from replay')
    panels = []
    for entry, job in zip(manifest['jobs'], jobs):
        fresh = audit_target(root, root/entry['path'])
        if fresh != read(root/job['output']/'audit.json'):
            raise ValueError('Fresh original target raw replay changed: '+entry['name'])
        panels.append(fresh)
    if target_certificate['rows'] != panels:
        raise ValueError('Complete32 raw target certificate differs from replay')
    analysis = pinned(root, master['analysis_contract'])
    measured = labels(panels, registry(master, training_jobs, analysis), analysis)
    forward = pinned(root, binding['forward_bundle'])
    directory = (root/binding['forward_bundle']['path']).parent
    publication = verify_bundle(root, binding['forward_bundle'], manifest['execution_protocol'])
    five = read(directory/'original_five/predictions.json')
    h1 = read(directory/'h1_predictions.json')
    predictions = score(five['point_predictions'], h1['point_predictions'], measured, analysis)
    models = read(directory/'original_five/early_models.json')
    features = read(directory/'original_five/late_features.json')
    feature_rows = {r['source_group']: r for r in features['rows']}
    standardized = {}
    for group in measured['source_groups']:
        standardized[group] = {}
        for signal in SIGNALS:
            model = models['models'][signal]
            x = feature_rows[group][signal]; scale = model['fit_population_std']
            standardized[group][signal] = {'late_signal': x, 'frozen_early_mean': model['fit_mean'],
                'frozen_early_population_std': scale, 'z': (x-model['fit_mean'])/scale if scale else 0.0,
                'frozen_intercept': model['intercept'], 'frozen_slope': model['slope'],
                'frozen_lambda': model['lambda'], 'predicted_G': five['point_predictions'][group][signal]}
    h1_model_artifact = read(directory/'h1_early_model.json')
    h1_features = read(directory/'h1_late_features.json')
    h1_model = h1_model_artifact['model']
    h1_rows = {r['source_group']: r for r in h1_features['rows']}
    for group in measured['source_groups']:
        x = h1_rows[group]['source_success_rate'];scale = h1_model['early_fit_population_std']
        standardized[group]['separate_H1'] = {'late_signal': x, 'frozen_early_mean': h1_model['early_fit_mean'],
            'frozen_early_population_std': scale, 'z': (x-h1_model['early_fit_mean'])/scale if scale else 0.0,
            'frozen_intercept': h1_model['intercept'], 'frozen_slope': h1_model['slope'],
            'frozen_lambda': h1_model['lambda'], 'predicted_G': h1['point_predictions'][group]}
    probe = pinned(root, master['stage1_probe_protocol'])
    early = pinned(root, probe['stage0_transfer_report'])
    if early['status'] != 'passed' or early['stage'] != 0:
        raise ValueError('Require the original complete early51/104 report')
    stage_comparison = {g: {'early': early['labels']['groups'][g], 'late': measured['groups'][g]}
                        for g in measured['source_groups']}
    train_tokens = sum(a['cost']['new_model_tokens'] for a in training_audits)
    eval_tokens = sum(a['cost']['new_model_tokens'] for a in panels)
    if (train_tokens != training_certificate['new_sampled_tokens'] or eval_tokens != target_certificate['new_sampled_tokens']
            or train_tokens > master['new_training_token_ceiling'] or eval_tokens > manifest['new_token_ceiling']):
        raise ValueError('Actual branch/panel costs differ or exceed registered bounds')
    probe_audit = pinned(root, forward['complete_probe_audit'])
    control_roots = list(binding['training_control_roots'])
    for job in jobs:
        run = read(root/job['output']/'run.json')
        authority = pinned(root, run['current_eval_authority'])
        if authority['control_root'] not in control_roots:control_roots.append(authority['control_root'])
    execution_failures = {name: read(root/name/'failures.json') if (root/name/'failures.json').exists() else []
                          for name in control_roots}
    training_processes = {j['case_id']: read(root/j['output']/'supervisor.json') for j in training_jobs}
    target_processes = {e['name']: read(root/j['output']/'supervisor.json') for e, j in zip(manifest['jobs'], jobs)}
    if any(p['returncode'] != 0 or p['hard_timeout'] for p in [*training_processes.values(), *target_processes.values()]):
        raise ValueError('Completed certificates conflict with original scientific process termination')
    costs = {'training_new_tokens': train_tokens, 'evaluation_new_tokens': eval_tokens,
        'shared_late_probe_tokens_once': probe_audit['new_sampled_tokens'],
        'late_total_new_model_tokens_once': train_tokens+eval_tokens+probe_audit['new_sampled_tokens'],
        'inherited_ancestor_tokens_recharged': 0,
        'training_nominal_tokens': master['nominal_new_training_tokens'],
        'training_complete_group_overshoot_tokens': train_tokens-master['nominal_new_training_tokens'],
        'training_rows': training_audits, 'evaluation_panels': panels,
        'optimizer_steps': sum(a['optimizer_steps'] for a in training_audits),
        'fresh_nonzero_gradient_steps': sum(len(a['fresh_nonzero_gradient_steps']) for a in training_audits),
        'changed_parameter_steps': sum(len(a['changed_parameter_steps']) for a in training_audits),
        'training_processes': training_processes, 'evaluation_processes': target_processes,
        'training_worker_wall_seconds': sum(p['wall_seconds'] for p in training_processes.values()),
        'evaluation_worker_wall_seconds': sum(p['wall_seconds'] for p in target_processes.values()),
        'execution_failure_records': execution_failures,
        'scope': 'Actual new model tokens, not equal FLOPs or GPU-active time. Training/probe/evaluation distinct; inherited parent and shared probe charged once. Failure records are not scientific negative outcomes; separately registered retries require explicit additional cost review.'}
    return {'status': 'passed', 'stage': 1, 'batch_mode': 'forward_only_15',
        'reported_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'raw_replay': {'training_jobs': 15, 'target_panels': 32, 'saved_audits_equal_fresh': True},
        'labels': measured, 'prediction_results': predictions, 'repeat_stability': stability(measured),
        'sealed_models': models, 'sealed_separate_H1_model': h1_model_artifact,
        'standardized_frozen_predictions': standardized, 'sealed_predictions': five,
        'sealed_separate_H1_predictions': h1, 'sealed_separate_H1_features': h1_features,
        'same_four_source_stage_comparison': stage_comparison,
        'early_four_source_original_five_results': early['prediction_results'],
        'stage_comparison_scope': 'Same four sources and same32 tasks, new registered evaluation seeds. Stage1 ancestor is the prespecified full stage0 Uniformr0 endpoint, shared with earlier results; changed learner state does not establish stronger capability. No late16-source ranking, late refit or independent replication claim.',
        'costs': costs, 'evidence': {'execution_protocol': manifest['execution_protocol'], 'manifest': manifest_ref,
            'complete_training_audit': binding['complete_training_audit'], 'complete_training_review': binding['complete_training_review'],
            'complete_target_audit': ref(root, root/manifest['output_root']/'audit_status.json'),
            'forward_bundle': binding['forward_bundle'], 'forward_publication': publication,
            'early_report': probe['stage0_transfer_report']},
        'implementation_sha256': {name: digest(root/name) for name in [
            'pipeline/report_bfcl_stage1_forward15_science.py', 'pipeline/bfcl_stage1_forward15_analysis.py',
            'pipeline/bfcl_transfer_labels.py', 'pipeline/bfcl_transfer_predictor.py',
            'pipeline/report_bfcl_stage1_forward32.py', 'pipeline/report_bfcl_source_training.py']},
        'reserved21_accessed': False, 'within_late_fitting_performed': False,
        'original_full_late51_104_complete': False, 'goal_complete': False,
        'interpretation_limits': ['G, V and prediction error answer different questions.',
            'A lower MSE does not demonstrate a monotone benefit from selecting higher-signal sources.',
            'Source, task, evaluation seed, training repeat and shared ancestor create dependence.',
            'Successful training audits and changed parameters do not establish positive transfer.',
            'Partial model failures do not reject all learnability/value mechanisms.',
            'Full late matrix, additional mechanisms and necessary model/environment extrapolation remain outside this approved batch.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', default='rl/bfcl_source_stage1_forward_report_20261016')
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT/'rl') or output.exists():
        raise ValueError('Require a new separate report directory within project rl')
    result = build(ROOT, {'path': args.manifest, 'sha256': args.manifest_sha256})
    output.mkdir()
    with (output/'audit_status.json').open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False);handle.write('\n')
    print(json.dumps({'status': 'passed', 'report': ref(ROOT, output/'audit_status.json'),
                      'training_jobs': 15, 'target_panels': 32, 'goal_complete': False}))


if __name__ == '__main__': main()
