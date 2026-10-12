"""Replay the complete stage0 matrix and score only the previously sealed predictions."""
import argparse
import math
from pathlib import Path
from bfcl_source_eval_contract import digest, pinned, verify_seal
from bfcl_transfer_labels import source_registry, aggregate_labels
from bfcl_transfer_predictor import SIGNALS, predict_models, score_heldout
from seal_bfcl_transfer_predictions import fit_artifacts
from report_bfcl_source_eval import audit_job as audit_evaluation
from supervise_bfcl_source_evaluation import validate_manifest
from supervise_bfcl_stage_fork import ROOT, read, write


def prediction_results(labels, predictions, analysis):
    heldout = analysis['source_split']['heldout']
    models = analysis['features']['allowed_models']
    if labels['phase'] != 'all' or labels['panel_count'] != 104 or labels['stage'] != 0:
        raise ValueError('Require the entire stage0 104-panel matrix')
    if set(labels['groups']) != set(analysis['source_split']['all_groups']):
        raise ValueError('Missing source groups')
    if set(predictions) != set(heldout) or any(set(p) != set(models) for p in predictions.values()):
        raise ValueError('Require all four sealed conditions and all prespecified models')
    if any(not math.isfinite(p) for row in predictions.values() for p in row.values()):
        raise ValueError('Nonfinite prediction')
    primary = score_heldout(predictions, {g: labels['groups'][g]['G'] for g in heldout}, analysis)
    repeats = []
    for r in range(3):
        gains = {}
        for g in heldout:
            rows = labels['groups'][g]['training_repeats']
            if len(rows) != 3 or [row['training_repeat'] for row in rows] != list(range(3)):
                raise ValueError('Incomplete training repeats')
            gains[g] = rows[r]['G']
        repeats.append({'training_repeat': r, **score_heldout(predictions, gains, analysis)})
    signs = {}
    for g, row in labels['groups'].items():
        values = [r['G'] for r in row['training_repeats']]
        signs[g] = {'G_mean': row['G'], 'V_mean': row['V'], 'positive_G_mean': row['G'] > 0,
                    'positive_V_mean': row['V'] > 0, 'G_by_training_repeat': values,
                    'G_training_repeat_range': [min(values), max(values)],
                    'positive_G_training_repeats': sum(v > 0 for v in values),
                    'V_by_training_repeat': [r['V'] for r in row['training_repeats']]}
    return {'primary': primary, 'same_sealed_predictions_by_training_repeat': repeats,
            'primary_contrast_negative': primary['primary_difference'] < 0,
            'primary_contrast_negative_training_repeats': sum(r['primary_difference'] < 0 for r in repeats),
            'gains_separate_from_prediction_accuracy': signs,
            'inference': 'Finite heldout-source comparison; repeated scores share ancestors and targets. No population CI, automatic generality claim, or winning-model selection.'}


def coefficient_rows(fitted):
    """Describe the signed, sealed fit; accuracy does not set the slope's sign."""
    models = fitted['models']
    if set(models) != {'fit_mean', *SIGNALS}:
        raise ValueError('Require all fixed models for coefficient presentation')
    rows = []
    for name in ['fit_mean', *SIGNALS]:
        model = models[name]
        values = [model['intercept']]
        if name != 'fit_mean':
            values += [model[k] for k in ['slope', 'fit_mean', 'fit_population_std', 'lambda']]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError('Nonfinite sealed coefficient')
        if name == 'fit_mean':
            rows.append({'model': name, 'intercept': model['intercept'], 'slope': None,
                         'direction': 'constant', 'fit_mean': None, 'fit_population_std': None})
            continue
        scale, slope = model['fit_population_std'], model['slope']
        if scale < 0 or model['lambda'] != 1 or (scale == 0 and slope != 0):
            raise ValueError('Invalid frozen standardization or ridge coefficient')
        rows.append({'model': name, 'intercept': model['intercept'], 'slope': slope,
                     'direction': 'positive' if slope > 0 else 'negative' if slope < 0 else 'zero',
                     'fit_mean': model['fit_mean'], 'fit_population_std': scale})
    return rows


def source_prediction_rows(labels, fitted, features, analysis, sealed_predictions):
    """Display every source using the original fit, and bind heldout rows to its seal."""
    coefficient_rows(fitted)
    predictions = predict_models(fitted, features, analysis)
    heldout = analysis['source_split']['heldout']
    if {g: predictions[g] for g in heldout} != sealed_predictions:
        raise ValueError('Displayed heldout predictions differ from the original seal')
    if set(labels['groups']) != set(predictions):
        raise ValueError('Require every registered source in the prediction display')
    return [{'source_group': g, 'split': 'heldout' if g in heldout else 'fit',
             'observed_G': labels['groups'][g]['G'], 'predictions': predictions[g],
             'squared_errors': {m: (p-labels['groups'][g]['G'])**2 for m, p in predictions[g].items()}}
            for g in analysis['source_split']['all_groups']]


def cost_accounting(master, training_rows, panels):
    expected = {e['name'] for e in master['jobs']}
    if len(training_rows) != 51 or {r['case_id'] for r in training_rows} != expected:
        raise ValueError('Require every training branch once')
    panel_keys = [(p['policy_case'], p['evaluation_repeat']) for p in panels]
    if len(panel_keys) != 104 or set(panel_keys) != {(c, e) for c in ['common', *expected] for e in range(2)}:
        raise ValueError('Require every evaluation panel exactly once')
    controls = {r['training_repeat']: r for r in training_rows if r['role'] == 'uniform_control'}
    rows = []
    for row in training_rows:
        tokens, nominal = row['new_sampled_tokens'], row['nominal_token_budget']
        if type(tokens) is not int or tokens < nominal or tokens > nominal + 32768:
            raise ValueError('Invalid realized training cost')
        rows.append(row | {'overshoot_tokens': tokens-nominal,
                          'token_ratio_to_same_repeat_uniform': tokens/controls[row['training_repeat']]['new_sampled_tokens']})
    training = sum(r['new_sampled_tokens'] for r in rows)
    evaluation = sum(p['cost']['new_model_tokens'] for p in panels)
    ancestor = master['costs']['ancestor_probe_tokens_once']
    probe = master['costs']['stage0_feature_and_validation_probe_tokens_once']
    return {'training_new_tokens': training, 'evaluation_new_tokens': evaluation,
            'shared_ancestor_probe_tokens_once': ancestor, 'stage0_feature_and_validation_probe_tokens_once': probe,
            'stage0_total_with_shared_acquisition_once': training+evaluation+ancestor+probe,
            'training_rows': rows, 'evaluation_panels': len(panels),
            'scope': 'This stage0 matrix plus shared acquisition only, not total project cost. Historical failures/engineering/dose runs remain in their ledgers. Nominal budgets have whole-group overshoot; this is not exactly matched FLOPs or a deployment-efficiency claim.'}


def replay_training(root, master):
    # Use the actual learner-state/allocator/gradient audit, loaded lazily so pure
    # statistical checks do not import Torch. Production CLI uses .venv-rl.
    import report_bfcl_source_training as training_audit
    if root.resolve() != training_audit.ROOT.resolve():
        raise ValueError('Training auditor is bound to this research repository')
    rows, evidence = [], []
    for entry in master['jobs']:
        job = pinned(root, entry)
        out = root/job['output']
        fresh = training_audit.audit_job(job, entry['sha256'])
        if fresh != read(out/'audit.json'):
            raise ValueError('Fresh training audit differs from saved audit')
        summary = read(out/'summary.json')
        rows.append({'case_id': job['case_id'], 'role': job['role'], 'source_group': job['source_group'],
                     'training_repeat': job['training_repeat'], 'nominal_token_budget': job['token_budget'],
                     'new_sampled_tokens': fresh['cost']['new_model_tokens'], 'optimizer_steps': fresh['optimizer_steps'],
                     'fresh_nonzero_gradient_steps': fresh['fresh_nonzero_gradient_steps'],
                     'changed_parameter_steps': fresh['changed_parameter_steps']})
        evidence.append({'job': entry, 'audit_sha256': digest(out/'audit.json'), 'source_files': {
            n: digest(out/n) for n in ['summary.json', 'allocation.jsonl', 'groups.jsonl', 'updates.jsonl',
                                      'microsteps.jsonl', 'metrics.jsonl', 'rollouts.jsonl', 'generation_calls.jsonl',
                                      'components/manifest.json']}, 'final_payload_sha256': fresh['final_payload_sha256']})
    return rows, evidence


def build_report(root, fit_ref, heldout_ref, seal_ref):
    fit_plan, heldout_plan = pinned(root, fit_ref), pinned(root, heldout_ref)
    if fit_plan['phase'] != 'fit' or heldout_plan['phase'] != 'heldout':
        raise ValueError('Both complete evaluation phases are required')
    if fit_plan['execution_protocol'] != heldout_plan['execution_protocol'] or heldout_plan['prediction_seal'] != seal_ref:
        raise ValueError('Phase execution or prediction identity differs')
    if fit_plan['prediction_seal'] is not None:
        raise ValueError('Fit phase cannot depend on heldout predictions')
    master = validate_manifest(root, fit_plan)
    validate_manifest(root, heldout_plan)
    analysis = pinned(root, master['analysis_contract'])
    receipt = verify_seal(root, seal_ref, fit_plan['execution_protocol']['sha256'],
                          master['analysis_contract']['sha256'], analysis['source_split']['heldout'], check_remote=False)
    if receipt['fit_evaluation_manifest'] != fit_ref or receipt['features'] != master['features']:
        raise ValueError('Sealed fit manifest or features differ')
    directory = (root/seal_ref['path']).parent
    features = pinned(root, master['features'])
    registry = source_registry(master, [pinned(root, e) for e in master['jobs']], analysis)
    phases, evidence = {}, {}
    for phase, plan in [('fit', fit_plan), ('heldout', heldout_plan)]:
        batch = read(root/plan['output_root']/'audit_status.json')
        if batch['status'] != 'passed' or batch['complete_panels'] != len(plan['jobs']):
            raise ValueError('Evaluation phase has not completed its audit')
        panels, phase_evidence = [], []
        for entry in plan['jobs']:
            job = pinned(root, entry)
            fresh = audit_evaluation(root, root/entry['path'])
            if fresh != read(root/job['output']/'audit.json'):
                raise ValueError('Fresh evaluation audit differs from saved audit')
            panels.append(fresh)
            phase_evidence.append({'job': entry, 'audit_sha256': digest(root/job['output']/'audit.json'),
                                   'source_files': fresh['source_files'], 'cost': fresh['cost']})
        phases[phase], evidence[phase] = panels, phase_evidence
    fit_labels = aggregate_labels(phases['fit'], registry, analysis, 'fit')
    sealed_labels = read(directory/'fit_labels.json')
    if (fit_labels | {'evidence': evidence['fit']}) != sealed_labels:
        raise ValueError('Sealed fit labels differ from raw replay')
    # Replay checks the original fit; it does not replace or tune the sealed model.
    models, prediction_artifact = fit_artifacts(master, analysis, features, fit_labels)
    if models != read(directory/'models.json') or prediction_artifact != read(directory/'predictions.json'):
        raise ValueError('Sealed coefficients or predictions are not the registered fit')
    panels = phases['fit']+phases['heldout']
    labels = aggregate_labels(panels, registry, analysis, 'all')
    result = prediction_results(labels, prediction_artifact['point_predictions'], analysis)
    training_rows, training_evidence = replay_training(root, master)
    costs = cost_accounting(master, training_rows, panels)
    return {'status': 'passed', 'stage': 0, 'labels': labels, 'prediction_results': result,
            'sealed_predictions': prediction_artifact, 'sealed_models': models,
            'coefficient_directions': coefficient_rows(models),
            'source_predictions': source_prediction_rows(labels, models, features, analysis,
                                                         prediction_artifact['point_predictions']),
            'costs': costs,
            'evidence': {'execution_protocol': fit_plan['execution_protocol'], 'fit_manifest': fit_ref,
                         'heldout_manifest': heldout_ref, 'prediction_seal': seal_ref,
                         'evaluation': evidence, 'training': training_evidence},
            'implementation_sha256': {str(p.relative_to(root)): digest(p) for p in [Path(__file__).resolve(),
                root/'pipeline/bfcl_transfer_labels.py', root/'pipeline/bfcl_transfer_predictor.py']},
            'goal_complete': False, 'scope': 'Stage0 finite-source prediction only; no new ID comparison, late-stage result, scale/family/environment extension, or comprehensive mechanism comparison.'}


def markdown(report):
    scores, labels = report['prediction_results'], report['labels']
    result = ['# Stage0迁移收益与封存预测检验', '',
              '全部51训练分支与104评测面板已重新审计；使用评测前发布的固定预测，未选择最佳模型或seed。', '',
              '## 四留出来源的预测误差', '', '| 预设模型 | MSE |', '|---|---:|']
    for name, value in scores['primary']['MSE'].items():
        result.append(f'| {name} | {value:.8f} |')
    result += ['', f"主差（任务级−fit均值）：{scores['primary']['primary_difference']:.8f}。负值有利于任务级信号。",
               f"机制次要差（任务级−组合级）：{scores['primary']['mechanism_secondary_difference']:.8f}。",
               '这是四个固定条件上的误差差，不是总体显著性检验，也不等于RL带来正收益。', '',
               '| 训练重复 | 同一封存预测的主误差差 |', '|---|---:|']
    for row in scores['same_sealed_predictions_by_training_repeat']:
        result.append(f"| {row['training_repeat']} | {row['primary_difference']:.8f} |")
    result += ['', '## 封存系数与拟合方向', '',
               '| 预设模型 | 截距 | 每个fit标准差的斜率 | 方向 | fit特征均值 | fit总体标准差 |',
               '|---|---:|---:|---|---:|---:|']
    direction = {'constant': '常数基线', 'positive': '正斜率', 'negative': '负斜率', 'zero': '零斜率'}
    def number(value):
        return '—' if value is None else f'{value:.10g}'
    for row in report['coefficient_directions']:
        result.append(f"| {row['model']} | {number(row['intercept'])} | {number(row['slope'])} | "
                      f"{direction[row['direction']]} | {number(row['fit_mean'])} | {number(row['fit_population_std'])} |")
    result += ['', '系数与标准化来自评测前封存的12个fit来源；斜率没有正号约束。MSE改善与斜率方向分别解释：负斜率模型即使预测更准，也不支持高分来源更值得学习。正斜率也只是有限fit条件中的关系，仍需留出/重复/阶段证据。', '',
               '## 固定预测与各来源实测', '',
               'fit行是拟合内描述，heldout行才是预设来源留出检验；全部使用同一原封存模型，未重新拟合或标准化。', '',
               '| 来源 | 预测划分 | 实测平均G | ' + ' | '.join(['fit_mean', *SIGNALS]) + ' |',
               '|---|---|---:|' + '---:|'*5]
    for row in report['source_predictions']:
        values = ' | '.join(f"{row['predictions'][m]:.8f}" for m in ['fit_mean', *SIGNALS])
        result.append(f"| {row['source_group']} | {row['split']} | {row['observed_G']:.8f} | {values} |")
    result += ['', '## 来源收益及重复差异', '', '| 来源 | 预测划分 | 平均G | 平均V | 三训练重复G |', '|---|---|---:|---:|---|']
    heldout = set(report['sealed_predictions']['heldout_groups'])
    for group, row in labels['groups'].items():
        repeats = ', '.join(f"{r['G']:.6f}" for r in row['training_repeats'])
        result.append(f"| {group} | {'heldout' if group in heldout else 'fit'} | {row['G']:.6f} | {row['V']:.6f} | {repeats} |")
    c = report['costs']
    result += ['', 'G相对同阶段起点；V相对同训练重复Uniform。V为正而G不为正，只说明相对Uniform表现较好。逐评测seed配对行与全部预测值保存在audit_status.json。', '',
               '## 成本与结论边界', '',
               f"本轮训练新增{c['training_new_tokens']:,} token，评测新增{c['evaluation_new_tokens']:,}；共同祖先{c['shared_ancestor_probe_tokens_once']:,}与阶段0探测{c['stage0_feature_and_validation_probe_tokens_once']:,}分别只计一次。",
               '逐分支实际超额、相对Uniform实际token比、更新与非零梯度信息均保留在JSON。名义同预算不等于实际token/FLOPs完全相等。历史实验、失败和工程成本另见原账本。',
               '当前候选主要属于可学习性；不代表多种经验价值机制全面比较。共同祖先和目标题带来相关性；不报告常规独立行置信区间。无新ID对照，不据此声称ID—迁移脱节。晚阶段、规模、家族/环境与21保留目标题仍需独立证据。', '']
    return '\n'.join(result)


def main():
    parser = argparse.ArgumentParser()
    for name in ['fit-manifest', 'heldout-manifest', 'prediction-receipt']:
        parser.add_argument('--'+name, type=Path, required=True)
        parser.add_argument('--'+name+'-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def reference(name):
        path = getattr(args, name).resolve()
        return {'path': str(path.relative_to(ROOT)), 'sha256': getattr(args, name+'_sha256')}
    if args.output.exists():
        raise FileExistsError('Do not overwrite a previous report')
    report = build_report(ROOT, reference('fit_manifest'), reference('heldout_manifest'), reference('prediction_receipt'))
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'audit_status.json', report)
    (args.output/'README.md').write_text(markdown(report))
    print(__import__('json').dumps({'status': report['status'], 'primary': report['prediction_results']['primary'], 'goal_complete': False}))


if __name__ == '__main__':
    main()
