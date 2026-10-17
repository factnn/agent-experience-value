"""Full native8B15/32 raw replay and unchanged prospective prediction report.

No generation, fitting, selection, correction, admission or overwrite. The
complete matrix is required before reading outcomes for scientific analysis.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from bfcl_qwen8_forward15_analysis import (registry, labels, prediction_views,
    paired_resampling, stability)
from bfcl_qwen8_source_execution import exclusive, ref
from bfcl_qwen8_transfer_contract import forecasts
from bfcl_source_eval_contract import pinned
from report_bfcl_intervention import lines, read
from report_bfcl_qwen8_source_training import audit_job as audit_training
from report_bfcl_qwen8_source_transfer import audit_job as audit_target
from supervise_bfcl_qwen8_source_transfer import load_manifest


def point_registry(root, master, binding, design, core, auxiliary, analysis):
    directory = (root / binding['cross_size_seal']['path']).parent
    original = read(directory / 'predictions.json')
    groups = analysis['source_split']['all_groups']
    points = {g: {'original_five/' + n: v for n, v in original['original_five'][g].items()} for g in groups}
    standardized = list(original['standardized_source_rows'])
    for g in groups:
        points[g]['separate_H1'] = original['separate_H1'][g]
        points[g].update({'separate_structural/' + n: v for n, v in original['separate_structural'][g].items()})
        points[g]['zero_gain_reference'] = design['zero_gain_reference']['actual_point_predictions'][g]
        standardized.append({'source_group': g, 'model': 'zero_gain_reference', 'raw_signal': None,
            'frozen_fit_mean': None, 'frozen_population_std': None, 'z': None, 'intercept': 0.,
            'slope': 0., 'lambda': None, 'predicted_G': points[g]['zero_gain_reference']})
    disposition = binding['auxiliary_forecast_disposition']
    models = {'original_five': read(directory / 'original_five_models.json'),
        'separate_H1': read(directory / 'h1_early_model.json'),
        'separate_structural': read(directory / 'structural_early_models.json'),
        'zero_gain_reference': design['zero_gain_reference']}
    if disposition['status'] == 'sealed':
        auxdir = (root / disposition['receipt']['path']).parent
        extra = read(auxdir / 'predictions.json')
        models['separate_model_conditioned'] = read(auxdir / 'frozen4b_models.json')
        for g in groups:
            points[g].update({'separate_model_conditioned/' + n: v for n, v in extra['points'][g].items()})
        standardized.extend(dict(r, model='separate_model_conditioned/' + r['signal'])
                            for r in extra['standardized_source_rows'])
    else:
        models['separate_model_conditioned'] = {'status': disposition['status'],
            'failure_disposition': disposition, 'unavailable_points_imputed': False}
    expected = 176 if disposition['status'] == 'sealed' else 144
    if (sum(len(v) for v in points.values()) != expected or len(standardized) != expected
            or len({tuple((r['source_group'], r['model'])) for r in standardized}) != expected):
        raise ValueError('Prospective point/model/source registry is incomplete or repeated')
    tested = analysis['source_split']['heldout']
    return points, standardized, models, {g: points[g] for g in tested}


def relationship_description(standardized, measured):
    groups = measured['source_groups']
    selected = [r for r in standardized if r['source_group'] in groups]
    by_model = defaultdict(dict)
    for row in selected:
        by_model[row['model']][row['source_group']] = row
    result = {}
    for name, rows in by_model.items():
        x = [rows[g].get('raw_signal', rows[g].get('raw_native8b_feature')) for g in groups]
        y = [measured['G_labels'][g] for g in groups]
        correlation = None
        if all(v is not None for v in x) and np.std(x) > 0 and np.std(y) > 0:
            correlation = float(np.corrcoef(x, y)[0, 1])
        result[name] = {'frozen_source_rows': [rows[g] for g in groups],
            'observed_four_source_raw_signal_G_Pearson': correlation,
            'scope': 'Four fixed sources, descriptive direction only. The signed frozen slope and point errors remain unchanged; neither association nor lower MSE proves a high-signal sampling policy improves transfer.'}
    return result


def numeric_cost(rows):
    keys = {k for row in rows for k, value in row['cost'].items() if type(value) in (int, float)}
    return {k: sum(row['cost'].get(k, 0) for row in rows) for k in sorted(keys)}


def build(root, manifest_ref):
    manifest, master, binding, jobs = load_manifest(root, manifest_ref)
    design = pinned(root, master['design'])
    target_path = root / manifest['output_root'] / 'audit_status.json'
    target = read(target_path)
    training = pinned(root, binding['complete_training_audit'])
    if (target['status'] != 'passed' or target['complete_panels'] != 32
            or target['manifest'] != manifest_ref or target['admission_binding'] != manifest['admission_binding']):
        raise ValueError('The full actual native8B32 certificate is required before analysis')
    source_jobs = [pinned(root, entry) for entry in master['jobs']]
    source_rows = [audit_training(root, root / entry['path'], manifest['execution_protocol']) for entry in master['jobs']]
    for job, actual in zip(source_jobs, source_rows):
        if actual != read(root / job['output'] / 'audit.json'):
            raise ValueError('Complete native8B training raw replay changed: ' + job['case_id'])
    if training['jobs'] != source_rows:
        raise ValueError('Native8B15 certificate differs from original raw replay')
    panels = [audit_target(root, root / entry['path']) for entry in manifest['jobs']]
    if (target['rows'] != panels or any(row != read(root / job['output'] / 'audit.json')
            for row, job in zip(panels, jobs))):
        raise ValueError('Native8B32 certificate differs from original raw replay')
    analysis = pinned(root, master['analysis_contract'])
    source_registry = registry(master, source_jobs, analysis)
    measured = labels(panels, source_registry, analysis)
    earliest_target = min(read(root / j['output'] / 'run.json')['evaluation_started_unix'] for j in jobs)
    core, auxiliary = forecasts(root, binding, master, earliest_target)
    all_points, standardized, models, points = point_registry(root, master, binding, design, core, auxiliary, analysis)
    outcomes = {(j['policy_case'], j['evaluation_repeat']): lines(root / j['output'] / 'outcomes.jsonl') for j in jobs}
    uncertainty = paired_resampling(outcomes, source_registry, master['evaluation'], measured,
                                    points, design['analysis']['descriptive_resampling'])
    common = pinned(root, master['native_common_binding'])
    common_audit = pinned(root, common['common_audit'])
    acquisition = pinned(root, master['source_prediction_binding'])
    probe_audit = pinned(root, acquisition['complete_native_probe_audit'])
    shared_cost = numeric_cost([common_audit, *probe_audit['rows']])
    train_cost, eval_cost = numeric_cost(source_rows), numeric_cost(panels)
    if (train_cost['new_model_tokens'] != training['new_sampled_tokens']
            or eval_cost['new_model_tokens'] != target['new_sampled_tokens']):
        raise ValueError('Full actual costs differ from their raw branch/panel certificates')
    auxiliary_design = pinned(root, master['auxiliary_candidate_design'])
    if binding['auxiliary_forecast_disposition']['status'] == 'sealed':
        auxiliary_cost = pinned(root, auxiliary['complete_measurement_audit'])['cost']
    else:
        auxiliary_cost = {'bounded_failure_receipt': binding['auxiliary_forecast_disposition']['receipt'],
                          'actual_partial_evidence': pinned(root, auxiliary['partial_cost_and_terminal_evidence'])}
    summaries = {j['case_id']: read(root / j['output'] / 'summary.json') for j in source_jobs}
    clusters = defaultdict(list)
    for case, summary in summaries.items():
        clusters[summary['final_fingerprint']].append(case)
    controls = {master['control_root'], auxiliary_design['control_root']}
    for job in jobs:
        run = read(root / job['output'] / 'run.json')
        controls.add(pinned(root, run['current_eval_authority'])['control_root'])
    native_terminal = pinned(root, acquisition['prior_native_probe_terminal_proof'])
    controls.update(str(Path(c['process_record']['path']).parent) for c in native_terminal['children'])
    controls.add(str(Path(common_audit['prior_control_failure']['path']).parent))
    failure_records = []
    for control in sorted(controls):
        for name in ['failures.json', 'control_failure.json', 'admission_failure.json',
                     'source_control_result.json', 'target_control_result.json']:
            path = root / control / name
            if path.exists():
                failure_records.append({'record': ref(root, path), 'value': read(path)})
    prior_admission = ((root / master['source_prediction_binding']['path']).parent
                       / 'source_admission_script_failure_20261017.json')
    if prior_admission.exists():
        failure_records.append({'record': ref(root, prior_admission), 'value': read(prior_admission)})
    old = pinned(root, auxiliary_design['fit_rule']['audited4b_report'])
    source_rows_with_points = [{'source_group': g, 'G': measured['G_labels'][g], 'V': measured['V_labels'][g],
        'frozen_point_predictions_G': points[g]} for g in measured['source_groups']]
    implementation = [Path(__file__).resolve(), root / 'pipeline/bfcl_qwen8_forward15_analysis.py',
        root / 'pipeline/bfcl_transfer_labels.py', root / 'pipeline/bfcl_stage1_forward15_analysis.py',
        root / 'pipeline/report_bfcl_qwen8_source_training.py', root / 'pipeline/report_bfcl_qwen8_source_transfer.py']
    return {'status': 'passed', 'kind': 'complete_native8b_forward15_32_science_report',
        'reported_at': datetime.now(timezone.utc).isoformat(), 'model_revision': master['model_revision'],
        'stage': 0, 'raw_replay': {'training_jobs': 15, 'target_panels': 32, 'saved_audits_equal_fresh': True},
        'labels': measured, 'source_predictions_and_observations': source_rows_with_points,
        'prediction_results': prediction_views(points, measured), 'repeat_stability': stability(measured),
        'paired_fixed_set_sensitivity': uncertainty, 'sealed_models': models,
        'all16_frozen_predictions_G': all_points, 'all16_standardized_frozen_rows': standardized,
        'signal_direction_description': relationship_description(standardized, measured),
        'tested_sources': measured['source_groups'], 'untested_forecast_sources': analysis['source_split']['fit'],
        'same_four_source_4b_8b': {g: {'original4b_stage0': old['labels']['groups'][g],
            'native8b_stage0': measured['groups'][g]} for g in measured['source_groups']},
        'costs': {'native_common_and_probe_once': shared_cost, 'source_training': train_cost,
            'target_evaluation': eval_cost, 'auxiliary_forward_measurement': auxiliary_cost,
            'native_route_generated_tokens_excluding_separate_capacity_case':
                shared_cost['new_model_tokens'] + train_cost['new_model_tokens'] + eval_cost['new_model_tokens'],
            'separate_E361_capacity_tokens': 5200, 'capacity_already_recorded_and_not_rerun': True,
            'training_nominal_tokens': design['budgets']['nominal_training_new_tokens'],
            'training_whole_group_overshoot_tokens': train_cost['new_model_tokens'] - design['budgets']['nominal_training_new_tokens'],
            'optimizer_steps': sum(a['optimizer_steps'] for a in source_rows),
            'fresh_nonzero_gradient_steps': sum(len(a['fresh_nonzero_gradient_steps']) for a in source_rows),
            'changed_parameter_steps': sum(len(a['changed_parameter_steps']) for a in source_rows),
            'training_worker_wall_seconds': sum(a['actual_supervised_worker_wall_seconds'] for a in source_rows),
            'evaluation_worker_wall_seconds': sum(a['actual_supervised_worker_wall_seconds'] for a in panels),
            'registered_budget_bounds': design['budgets'], 'failure_and_control_records': failure_records,
            'scope': 'Actual generated/input/external tokens and supervised wall; not strict equal FLOPs. Common/probe/shared forward work counted once. Prior4B study costs are not charged again. Control/publication failures are distinct from scientific negative gains.'},
        'training_weight_fingerprint_clusters': dict(clusters), 'distinct_training_endpoint_weights': len(clusters),
        'training_audits': source_rows, 'target_audits': panels,
        'evidence': {'scientific_design': master['design'], 'execution_protocol': manifest['execution_protocol'],
            'manifest': manifest_ref, 'admission_binding': manifest['admission_binding'],
            'complete_training_audit': binding['complete_training_audit'],
            'complete_training_review': binding['complete_training_review'], 'complete_target_audit': ref(root, target_path),
            'cross_size_seal': binding['cross_size_seal'], 'cross_size_publication': binding['cross_size_publication'],
            'auxiliary_disposition': binding['auxiliary_forecast_disposition'],
            'common_audit': common['common_audit'], 'complete_probe_audit': acquisition['complete_native_probe_audit']},
        'implementation': [ref(root, path) for path in implementation],
        'new_model_tokens_from_reporting': 0, 'within_native8b_fitting_performed': False,
        'reserved21_accessed': False, 'goal_complete': False,
        'scope': 'Frozen4B mappings tested at one native8B state on four original heldout sources and the same32 target tasks. Same-family/model-state/template/context and shared-seed dependencies preclude a pure parameter-count causal claim. Twelve forecasts remain untested. A lower prediction error does not prove high-signal acquisition helps; other mechanisms, ID, family/environment and reserved21 validation remain unfinished.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', default='rl/qwen3_8b_stage0_forward_report_20261017')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output).resolve()
    if not output.is_relative_to(root / 'rl') or output.exists():
        raise FileExistsError('Require a new separate report directory; preserve original evidence')
    result = build(root, {'path': args.manifest, 'sha256': args.manifest_sha256})
    output.mkdir(exist_ok=False)
    exclusive(output / 'audit_status.json', result)
    print(json.dumps({'report': ref(root, output / 'audit_status.json'), 'status': 'passed', 'goal_complete': False}))
