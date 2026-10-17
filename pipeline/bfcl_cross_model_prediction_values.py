"""Feature-only values for frozen original-five, H1 and structural mappings.

Pure calculation helpers. These do not seal, publish or authorize training/eval;
callers must bind complete raw audits and publish actual forecasts before use.
"""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from bfcl_probe_forecast import forecast
from bfcl_transfer_predictor import SIGNALS,feature_rows,predict_models
from fit_bfcl_h1_source_success import source_success,predict as h1_predict
from bfcl_structural_value_features import SIGNALS as STRUCTURAL,source_features


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def feature_case(root,plan,case):
    out=root/plan['output_root']/case['name']
    rows=[json.loads(line) for line in (out/'feature.jsonl').open()]
    assert [r['task_id'] for r in rows]==case['task_ids'] and [r['seed'] for r in rows]==case['seeds']['feature']
    assert all(r['phase']=='feature' and r['stage']==0 and r['source_group']==case['source_group'] and r['policy_fingerprint']==plan['policy_fingerprint'] for r in rows)
    expected=forecast(rows,case['historical_signal'])
    assert expected==json.loads((out/'predictions.json').read_text())
    values=list(expected['task_predictions'].values());by_task={r['task_id']:r for r in rows}
    structural_rows=[]
    for line in (out/'rollouts.jsonl').open():
        row=json.loads(line)
        if row['phase']!='feature':continue
        assert row['task_id'] in by_task and row['policy_fingerprint']==plan['policy_fingerprint']
        match=by_task[row['task_id']]
        assert row['seed']==match['seed'] and row['draw_id']==match['draw_id']
        assert row['group']==case['task_ids'].index(row['task_id'])
        structural_rows.append(row)
    structural=source_features(structural_rows,case['task_ids'])
    return {'source_group':case['source_group'],'task_count':len(rows),
        # Preserve the original Stage0 source-aggregation floating-point path,
        # including averaging the repeated pooled value over all source tasks.
        'primary_task_beta':sum(x['task_beta'] for x in values)/len(values),
        'current_pooled_beta':sum(x['current_pooled_beta'] for x in values)/len(values),
        'historical_frontier':case['historical_signal']['frontier'],
        'historical_coverage':case['historical_signal']['coverage_deficit'],
        'source_success_rate':source_success(rows),**{key:structural[key] for key in STRUCTURAL},
        'feature_sha256':digest(out/'feature.jsonl'),'prediction_sha256':digest(out/'predictions.json'),
        'structural_rollout_source_sha256':digest(out/'rollouts.jsonl')}


def feature_panels(root,plan,plan_ref,analysis,analysis_sha):
    rows=[feature_case(root,plan,c) for c in plan['cases']]
    assert len(rows)==16 and {r['source_group'] for r in rows}==set(analysis['source_split']['all_groups'])
    result={'stage':0,'policy_fingerprint':plan['policy_fingerprint'],'source_protocol':plan_ref,
        'analysis_contract_sha256':analysis_sha,'rows':rows,'validation_labels_used':False,
        'target_labels_used':False,'model_revision':plan.get('model_revision'),
        'scope':'All source-training feature draws only. Structural values remain separate new candidates.'}
    feature_rows(result,analysis)
    assert all(0<=r[k]<=1 for r in rows for k in [*SIGNALS,'source_success_rate',*STRUCTURAL])
    return result


def model_values(model,location='fit_mean',scale_key='fit_population_std'):
    assert model['lambda']==1.0
    values=[model[k] for k in ['intercept','slope',location,scale_key,'lambda']]
    assert all(type(v) in (int,float) and math.isfinite(v) for v in values) and model[scale_key]>=0


def predictions(five,h1,structural,features,analysis):
    assert five['fit_stage']==0 and five['feature_selection'] is False and five['prediction_clipping'] is False
    assert set(five['models'])=={'fit_mean',*SIGNALS}
    assert five['fit_groups']==structural['fit_groups']==analysis['source_split']['fit']
    assert five['fit_policy_fingerprint']==structural['fit_policy_fingerprint']
    assert five['heldout_groups']==structural['historical_heldout_groups']==analysis['source_split']['heldout']
    for signal in SIGNALS:model_values(five['models'][signal])
    assert all(h1['model'][k] is False for k in ['feature_selection','prediction_clipping','standardization_refitted_on_late_features','sign_constraint'])
    assert h1['model']['fit_groups']==analysis['source_split']['fit'] and h1['model']['future_test_groups']==analysis['source_split']['heldout']
    model_values(h1['model'],'early_fit_mean','early_fit_population_std')
    assert set(structural['models'])==set(STRUCTURAL)
    assert structural['feature_selection'] is False and structural['hyperparameter_search'] is False and structural['prediction_clipping'] is False
    for model in structural['models'].values():model_values(model)
    original=predict_models(five,features,analysis);rows=feature_rows(features,analysis)
    h1_points={};structural_points={};standardized=[]
    for group in analysis['source_split']['all_groups']:
        row=rows[group];h1_points[group]=h1_predict(h1['model'],row['source_success_rate']);structural_points[group]={}
        standardized.append({'source_group':group,'model':'original_five/fit_mean','raw_signal':None,
            'frozen_fit_mean':None,'frozen_population_std':None,'z':None,'intercept':five['models']['fit_mean']['intercept'],
            'slope':0.0,'lambda':None,'predicted_G':original[group]['fit_mean']})
        mappings=[('original_five/'+s,row[s],five['models'][s],'fit_mean','fit_population_std') for s in SIGNALS]
        mappings.append(('separate_H1',row['source_success_rate'],h1['model'],'early_fit_mean','early_fit_population_std'))
        mappings.extend(('separate_structural/'+s,row[s],structural['models'][s],'fit_mean','fit_population_std') for s in STRUCTURAL)
        for name,x,model,location,scale_key in mappings:
            scale=model[scale_key];z=(x-model[location])/scale if scale else 0.0;value=model['intercept']+model['slope']*z
            if name.startswith('original_five/'):assert value==original[group][name.split('/',1)[1]]
            elif name=='separate_H1':assert value==h1_points[group]
            else:structural_points[group][name.split('/',1)[1]]=value
            standardized.append({'source_group':group,'model':name,'raw_signal':x,'frozen_fit_mean':model[location],
                'frozen_population_std':scale,'z':z,'intercept':model['intercept'],'slope':model['slope'],
                'lambda':model['lambda'],'predicted_G':value})
    assert len(standardized)==128 and all(math.isfinite(x['predicted_G']) for x in standardized)
    return {'kind':'frozen_model_point_values_requiring_separate_publication_seal','target_model_revision':features['model_revision'],
        'target_policy_fingerprint':features['policy_fingerprint'],'target':'G','all_source_groups':analysis['source_split']['all_groups'],
        'original_five':original,'separate_H1':h1_points,'separate_structural':structural_points,
        'standardized_source_rows':standardized,'model_refitted':False,'standardization_refitted':False,
        'prediction_clipping':False,'target_labels_used':False,'publication_verified_by_this_helper':False,
        'scope':'Point calculations only; caller must seal/publish exact model, feature and prediction bytes. These three model blocks remain separate.'}
