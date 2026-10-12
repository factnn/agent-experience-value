"""Seal stage0 coefficients applied to stage1 features, without any late labels.

Late target runners must call verify_forward_seal before *every* target panel,
including common/fit controls, not just the four heldout source endpoints.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean
import time
from bfcl_source_eval_contract import digest, pinned, verify_seal
from bfcl_transfer_predictor import SIGNALS, feature_rows, predict_models
from bfcl_probe_forecast import forecast
from supervise_bfcl_stage_fork import ROOT, read, write


def feature_row(rows, case, fingerprint):
    if [r['task_id'] for r in rows] != case['task_ids'] or [r['seed'] for r in rows] != case['seeds']['feature']:
        raise ValueError('Late features must use the entire ordered feature panel')
    if any(r['phase'] != 'feature' or r['stage'] != 1 or r['policy_fingerprint'] != fingerprint or
           r['source_group'] != case['source_group'] for r in rows):
        raise ValueError('Late features contain validation rewards or the wrong stage/policy/source')
    f = forecast(rows, case['historical_signal'])
    values = list(f['task_predictions'].values())
    return {'source_group':case['source_group'], 'task_count':len(rows),
            'primary_task_beta':mean(v['task_beta'] for v in values),
            'current_pooled_beta':values[0]['current_pooled_beta'],
            'historical_frontier':case['historical_signal']['frontier'],
            'historical_coverage':case['historical_signal']['coverage_deficit']}, f


def forward_artifact(fitted, late_features, analysis, early_fingerprint):
    if fitted['fit_stage'] != 0 or fitted['fit_policy_fingerprint'] != early_fingerprint:
        raise ValueError('Require the original stage0 fitted model')
    if fitted['feature_selection'] is not False or fitted['prediction_clipping'] is not False:
        raise ValueError('No feature selection or clipping in stage extrapolation')
    if set(fitted['models']) != {'fit_mean', *SIGNALS}:
        raise ValueError('All five original predictors are required')
    for signal, model in fitted['models'].items():
        fields = {'intercept'} if signal == 'fit_mean' else {'intercept','slope','fit_mean','fit_population_std','lambda'}
        if set(model) != fields or any(type(v) not in (int,float) or not math.isfinite(v) for v in model.values()):
            raise ValueError('Invalid frozen model coefficients')
        if signal != 'fit_mean' and (model['lambda'] != 1 or model['fit_population_std'] < 0):
            raise ValueError('Frozen regularization/normalization changed')
    if late_features['stage'] != 1 or late_features['validation_labels_used'] is not False:
        raise ValueError('Require stage1 feature-only measurements')
    rows = feature_rows(late_features, analysis)
    if any(not 0 <= r[s] <= 1 for r in rows.values() for s in SIGNALS):
        raise ValueError('Late probability/coverage feature outside [0,1]')
    predictions = predict_models(fitted, late_features, analysis)
    return {'kind':'early_to_late_transfer_predictions', 'stage':1, 'fit_stage':0, 'target':'G',
            'heldout_groups':analysis['source_split']['heldout'],
            'point_predictions':{g:predictions[g] for g in analysis['source_split']['heldout']},
            'late_target_labels_used':False, 'refit':False, 'renormalized':False,
            'feature_selection':False, 'prediction_clipping':False,
            'scope':'Frozen early models on the four late heldout sources; within-late fitting is a separate analysis.'}


def extract_features(root, plan, protocol_ref, analysis_ref):
    # Heavy imports are needed only for an actual complete raw-probe replay.
    import report_bfcl_stage1_probe as auditor
    if auditor.ROOT.resolve() != root.resolve():
        raise ValueError('Raw auditor root does not match the evidence root')
    analysis = pinned(root, analysis_ref)
    batch = read(root/plan['output_root']/'audit_status.json')
    if batch['status'] != 'passed' or batch['stage'] != 1 or batch['complete_cases'] != 16 or not batch['all_16_groups_complete']:
        raise ValueError('Require all sixteen audited late probes')
    rows, evidence = [], []
    for case in plan['cases']:
        out = root/plan['output_root']/case['name']
        fresh = auditor.audit_case(plan, case, protocol_ref['sha256'])
        saved = read(out/'audit.json')
        if saved['status'] != 'passed' or saved['rows'] != [fresh]:
            raise ValueError('Fresh late raw audit differs from saved case audit')
        raw = [json.loads(line) for line in (out/'feature.jsonl').read_text().splitlines()]
        row, prediction = feature_row(raw, case, plan['policy_fingerprint'])
        if prediction != read(out/'predictions.json'):
            raise ValueError('Late feature forecast mismatch')
        rows.append(row | {'feature_sha256':digest(out/'feature.jsonl'), 'prediction_sha256':digest(out/'predictions.json')})
        evidence.append({'case':case['name'], 'audit':{'path':str((out/'audit.json').relative_to(root)), 'sha256':digest(out/'audit.json')},
            'source_files':{str((out/n).relative_to(root)):digest(out/n) for n in ['run.json','summary.json','feature.jsonl','validation.jsonl',
                'generation_calls.jsonl','rollouts.jsonl','predictions.json','prediction_receipt.json','validation_start.json']}, 'raw_audit':fresh})
    if [e['raw_audit'] for e in evidence] != batch['rows']:
        raise ValueError('Fresh late raw audits differ from complete batch audit')
    result = {'stage':1, 'policy_fingerprint':plan['policy_fingerprint'], 'source_protocol':protocol_ref,
              'parent_binding':plan['parent_binding'], 'analysis_contract_sha256':analysis_ref['sha256'],
              'rows':rows, 'validation_labels_used':False, 'local_validation_read_for_raw_audit_only':True, 'evidence':evidence}
    feature_rows(result, analysis)
    return result


def seal(root, protocol_ref, early_ref, output):
    from bfcl_stage1_probe_contract import load_protocol
    plan, binding = load_protocol(root, root/protocol_ref['path'], protocol_ref['sha256'])
    report = pinned(root, plan['stage0_transfer_report'])
    if report['evidence']['prediction_seal'] != early_ref:
        raise ValueError('Use the early model already audited in the stage0 report')
    master = pinned(root, binding['stage0_execution_protocol'])
    analysis_ref = master['analysis_contract']; analysis = pinned(root, analysis_ref)
    early = verify_seal(root, early_ref, binding['stage0_execution_protocol']['sha256'], analysis_ref['sha256'], analysis['source_split']['heldout'])
    if early['implementation_sha256']['pipeline/bfcl_transfer_predictor.py'] != digest(root/'pipeline/bfcl_transfer_predictor.py'):
        raise ValueError('Predictor implementation differs from the early seal')
    early_dir = (root/early_ref['path']).parent
    fitted = read(early_dir/'models.json')
    late = extract_features(root, plan, protocol_ref, analysis_ref)
    prediction = forward_artifact(fitted, late, analysis, master['source_stage0']['policy_fingerprint'])
    output.mkdir(parents=True, exist_ok=False)
    # Copy the published model bytes verbatim. No late normalization/refit path.
    (output/'early_models.json').write_bytes((early_dir/'models.json').read_bytes())
    if digest(output/'early_models.json') != early['files']['models.json']:
        raise ValueError('Early model changed during copying')
    write(output/'late_features.json', late); write(output/'predictions.json', prediction)
    receipt = {'kind':'early_to_late_prediction_seal', 'stage':1, 'sealed_unix':time.time(),
        'early_prediction_seal':early_ref, 'stage0_execution_protocol':binding['stage0_execution_protocol'],
        'stage1_probe_protocol':protocol_ref, 'analysis_contract':analysis_ref, 'parent_binding':plan['parent_binding'],
        'heldout_groups':analysis['source_split']['heldout'],
        'files':{n:digest(output/n) for n in ['early_models.json','late_features.json','predictions.json']},
        'implementation_sha256':{p:digest(root/p) for p in ['pipeline/seal_bfcl_forward_predictions.py','pipeline/bfcl_transfer_predictor.py','pipeline/report_bfcl_stage1_probe.py']},
        'publication_required_before_any_stage1_target_evaluation':True,
        'late_target_labels_used':False, 'new_sampled_tokens':0}
    write(output/'receipt.json', receipt)
    return receipt


def verify_forward_seal(root, reference, probe_ref, parent_ref, analysis_ref, *, check_remote=True, evaluation_started_unix=None):
    """Future stage1 target runners/auditors must bind and verify these inputs."""
    from supervise_bfcl_stage1_probe import verify_publication
    receipt = pinned(root, reference)
    if receipt['kind'] != 'early_to_late_prediction_seal' or receipt['stage'] != 1 or receipt['late_target_labels_used'] is not False:
        raise ValueError('Require an early-to-late seal without late target labels')
    if receipt['stage1_probe_protocol'] != probe_ref or receipt['parent_binding'] != parent_ref or receipt['analysis_contract'] != analysis_ref:
        raise ValueError('Forward seal stage/parent/analysis binding mismatch')
    if receipt['publication_required_before_any_stage1_target_evaluation'] is not True:
        raise ValueError('Forward seal must precede every late target panel')
    plan = pinned(root, probe_ref); binding = pinned(root, parent_ref); analysis = pinned(root, analysis_ref)
    if plan['stage'] != 1 or plan['parent_binding'] != parent_ref or receipt['stage0_execution_protocol'] != binding['stage0_execution_protocol']:
        raise ValueError('Forward seal does not bind the fixed ancestor')
    master = pinned(root, binding['stage0_execution_protocol'])
    if master['analysis_contract'] != analysis_ref:
        raise ValueError('Forward analysis differs from early protocol')
    report = pinned(root, plan['stage0_transfer_report'])
    early_ref = receipt['early_prediction_seal']
    if early_ref != report['evidence']['prediction_seal']:
        raise ValueError('Forward model is not the audited early model')
    early = verify_seal(root, early_ref, binding['stage0_execution_protocol']['sha256'], analysis_ref['sha256'], analysis['source_split']['heldout'], check_remote=check_remote)
    if early['implementation_sha256']['pipeline/bfcl_transfer_predictor.py'] != digest(root/'pipeline/bfcl_transfer_predictor.py'):
        raise ValueError('Predictor implementation differs from the early seal')
    directory = (root/reference['path']).parent
    if set(receipt['files']) != {'early_models.json','late_features.json','predictions.json'}:
        raise ValueError('Incomplete forward seal')
    for name,sha in receipt['files'].items():
        if digest(directory/name) != sha:raise ValueError('Forward sealed bytes changed')
    if receipt['files']['early_models.json'] != early['files']['models.json']:
        raise ValueError('Forward model bytes differ from the published early model')
    late = read(directory/'late_features.json')
    if late['source_protocol'] != probe_ref or late['parent_binding'] != parent_ref or late['analysis_contract_sha256'] != analysis_ref['sha256'] or late['policy_fingerprint'] != binding['policy_fingerprint']:
        raise ValueError('Late feature provenance changed')
    predicted = forward_artifact(read(directory/'early_models.json'), late, analysis, master['source_stage0']['policy_fingerprint'])
    if predicted != read(directory/'predictions.json') or receipt['heldout_groups'] != analysis['source_split']['heldout']:
        raise ValueError('Forward prediction is not the original fixed model on late features')
    publication = read(directory/'publication.json')
    if publication['receipt_sha256'] != reference['sha256'] or not early['sealed_unix'] <= receipt['sealed_unix'] <= publication['verified_remote_at_unix']:
        raise ValueError('Invalid forward publication chronology')
    if evaluation_started_unix is not None and publication['verified_remote_at_unix'] > evaluation_started_unix:
        raise ValueError('Late target evaluation preceded forward publication')
    files = {str((directory/name).relative_to(root)):sha for name,sha in receipt['files'].items()}
    files[reference['path']] = reference['sha256']
    if check_remote:
        verify_publication(root, files, publication['commit'])
    else:
        import hashlib
        import subprocess
        for name,sha in files.items():
            result = subprocess.run(['git','show',publication['commit']+':'+name],cwd=root,capture_output=True)
            if result.returncode or hashlib.sha256(result.stdout).hexdigest() != sha:
                raise ValueError('Forward publication commit does not contain the sealed bytes')
    return receipt


def main():
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest='command',required=True)
    p = commands.add_parser('seal')
    for flag in ['probe-protocol','early-receipt']:
        p.add_argument('--'+flag,type=Path,required=True);p.add_argument('--'+flag+'-sha256',required=True)
    p.add_argument('--output',type=Path,required=True)
    p = commands.add_parser('record-publication');p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--receipt-sha256',required=True);p.add_argument('--commit',required=True)
    args = parser.parse_args()
    if args.command == 'seal':
        def ref(name):return {'path':str(getattr(args,name).resolve().relative_to(ROOT)), 'sha256':getattr(args,name+'_sha256')}
        result = seal(ROOT,ref('probe_protocol'),ref('early_receipt'),args.output.resolve())
    else:
        from seal_bfcl_transfer_predictions import record_publication
        if read(args.receipt)['kind'] != 'early_to_late_prediction_seal':raise ValueError('Not a forward seal')
        result = record_publication(ROOT,args.receipt.resolve(),args.receipt_sha256,args.commit)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':main()
