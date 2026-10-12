"""Complete, paired stage0 labels; never treat repeats as new source groups."""
import math
from statistics import mean
from bfcl_source_eval_contract import policy_role


def source_registry(master, training_jobs, analysis):
    if master['training_repeats'] != 3 or len(training_jobs) != 51:
        raise ValueError('Require the frozen stage0 three-repeat, 51-branch matrix')
    registered = [entry['name'] for entry in master['jobs']]
    if len(set(registered)) != 51 or {j['case_id'] for j in training_jobs} != set(registered):
        raise ValueError('Training identities do not match the execution protocol')
    result = {'common': {'role': 'baseline', 'training_repeat': None, 'source_group': None}}
    cells = set()
    for job in training_jobs:
        repeat = job['training_repeat']
        if job['stage'] != 0 or type(repeat) is not int or repeat not in range(3):
            raise ValueError('Different stage or training-repeat design')
        role = policy_role(job['case_id'], job, analysis)
        group = job['source_group']
        if role == 'control' and group is not None:
            raise ValueError('Uniform control cannot alias a source group')
        cell = group, repeat
        if cell in cells:
            raise ValueError('Duplicate training condition')
        cells.add(cell)
        result[job['case_id']] = {'role': role, 'training_repeat': repeat, 'source_group': group}
    expected = {(g, r) for g in [None, *analysis['source_split']['all_groups']] for r in range(3)}
    if cells != expected:
        raise ValueError('Missing source or control training condition')
    return result


def aggregate_labels(panels, registry, analysis, phase):
    """Input scores must come from re-audited panels, not unaudited summaries."""
    if phase not in ('fit', 'all'):
        raise ValueError('Unsupported analysis phase')
    groups = analysis['source_split']['fit' if phase == 'fit' else 'all_groups']
    selected = {case: row for case, row in registry.items()
                if row['role'] in ('baseline', 'control') or row['source_group'] in groups}
    expected = {(case, e) for case in selected for e in range(2)}
    values = {}
    for panel in panels:
        key = panel['policy_case'], panel['evaluation_repeat']
        if type(key[1]) is not int or key not in expected or key in values:
            raise ValueError('Extra, duplicate, or unregistered panel; heldout labels cannot enter the fit phase')
        if panel['status'] != 'passed' or panel['role'] != selected[key[0]]['role']:
            raise ValueError('Panel failed or source role changed')
        score = panel['macro_success']
        if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid audited success rate')
        strata = panel['stratum_success']
        if set(strata) != set(analysis['target_panel']['registry'].values()):
            raise ValueError('Missing target combination')
        if any(not math.isfinite(s) or not 0 <= s <= 1 for s in strata.values()) or abs(score-mean(strata.values())) > 1e-12:
            raise ValueError('Score must use the registered macro weighting')
        values[key] = panel
    if set(values) != expected:
        raise ValueError('Incomplete paired matrix; no source dropping or imputation')
    controls = {row['training_repeat']: case for case, row in selected.items() if row['role'] == 'control'}
    baseline = [values['common', e]['macro_success'] for e in range(2)]
    uniform = {r: [values[controls[r], e]['macro_success'] - baseline[e] for e in range(2)] for r in range(3)}
    result = {}
    for group in groups:
        repeats = []
        for r in range(3):
            case = next(c for c, row in selected.items() if row['source_group'] == group and row['training_repeat'] == r)
            cells = []
            for e in range(2):
                endpoint = values[case, e]['macro_success']
                gain = endpoint - baseline[e]
                cells.append({'evaluation_repeat': e, 'endpoint_success': endpoint, 'baseline_success': baseline[e],
                              'G': gain, 'G_uniform': uniform[r][e], 'V': gain-uniform[r][e]})
            repeats.append({'training_repeat': r, 'policy_case': case, 'panels': cells,
                            'G': mean(x['G'] for x in cells), 'V': mean(x['V'] for x in cells)})
        result[group] = {'G': mean(row['G'] for row in repeats), 'V': mean(row['V'] for row in repeats),
                         'training_repeats': repeats}
    return {'stage': 0, 'phase': phase, 'panel_count': len(values), 'source_groups': list(groups),
            'baseline_success_by_evaluation_repeat': baseline,
            'uniform_G_by_training_repeat': {str(r): {'panels': uniform[r], 'G': mean(uniform[r])} for r in range(3)},
            'groups': result, 'G_labels': {g: row['G'] for g, row in result.items()},
            'V_labels': {g: row['V'] for g, row in result.items()},
            'scope': 'Finite source conditions; shared baseline/targets. G, V, and predictive accuracy are distinct; no independent-row CI.'}
