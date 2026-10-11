"""Prespecified local-opportunity forecasts, fitted only to feature-panel rewards."""
from math import prod
from bfcl_task_probe_statistics import validate_panel, source_statistics


def beta_mixed(successes, observations):
    if observations < 0 or not 0 <= successes <= observations:
        raise ValueError('Invalid Bernoulli counts')
    a, b = 1 + successes, 1 + observations - successes
    return 1 - prod((a+i)/(a+b+i) for i in range(4)) - prod((b+i)/(a+b+i) for i in range(4))


def forecast(rows, historical):
    tasks, mapping = validate_panel(rows)
    if len(set(mapping.values())) != 1:
        raise ValueError('One source group per forecast receipt')
    total = sum(len(v) for v in tasks.values());successes = sum(sum(v) for v in tasks.values())
    current_pooled = beta_mixed(successes, total)
    old = beta_mixed(historical['successes_in_window'], historical['observed_episodes'])
    p = successes/total
    return {'task_predictions': {t: {'task_beta':beta_mixed(sum(v),len(v)),
             'current_pooled_beta':current_pooled, 'historical_pooled_beta':old,
             'current_pooled_plugin':1-p**4-(1-p)**4, 'prior_beta':beta_mixed(0,0)} for t,v in tasks.items()},
            'task_level_statistics':source_statistics(tasks),
            'historical':historical,'feature_draw_ids':[r['draw_id'] for r in rows],
            'primary_contrast':'task_beta_minus_current_pooled_beta_Brier',
            'prior':{'alpha':1,'beta':1},'four_rollouts_per_task':True}


def score(predictions, validation_rows):
    tasks,_=validate_panel(validation_rows)
    if set(tasks)!=set(predictions['task_predictions']):raise ValueError('Task panel mismatch')
    if set(predictions['feature_draw_ids']) & {r['draw_id'] for r in validation_rows}:
        raise ValueError('Prediction/validation draw overlap')
    if any(len(v)!=4 for v in tasks.values()):raise ValueError('Exactly one validation group per task')
    observed={t:int(0<sum(v)<4) for t,v in tasks.items()}
    keys=list(next(iter(predictions['task_predictions'].values())))
    losses={k:sum((predictions['task_predictions'][t][k]-y)**2 for t,y in observed.items())/len(observed) for k in keys}
    return {'task_count':len(tasks),'observed_mixed':observed,'Brier':losses,
            'primary_difference':losses['task_beta']-losses['current_pooled_beta'],
            'interpretation':'Negative primary difference favors task-level forecast; conditional local proxy only.'}
