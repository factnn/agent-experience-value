"""Exploratory4B fits for separately registered prospective structural candidates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from bfcl_structural_value_features import SIGNALS


def fit_one(x,y):
    assert len(x)==len(y)>1 and all(math.isfinite(v) for v in [*x,*y])
    location=mean(x);scale=math.sqrt(mean((v-location)**2 for v in x));intercept=mean(y)
    z=[(v-location)/scale for v in x] if scale else [0.0]*len(x)
    slope=sum(a*(b-intercept) for a,b in zip(z,y))/(sum(a*a for a in z)+1.0)
    return {'intercept':intercept,'slope':slope,'fit_mean':location,'fit_population_std':scale,'lambda':1.0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--features',required=True);parser.add_argument('--early-report',required=True)
    parser.add_argument('--contract',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    def read(name):return json.loads((args.root/name).read_text())
    def ref(name):return {'path':name,'sha256':hashlib.sha256((args.root/name).read_bytes()).hexdigest()}
    features=read(args.features);report=read(args.early_report);contract=read(args.contract)
    assert features['validation_rewards_used'] is False and features['target_labels_used_for_extraction'] is False
    assert report['status']=='passed' and report['labels']['panel_count']==104
    assert ref(args.early_report)['sha256']=='c26424eee244968470b478a32ad6bf4b96d5563197972bc04331d6bc753bb992'
    fit_groups=contract['source_split']['fit'];heldout=contract['source_split']['heldout']
    assert len(fit_groups)==12 and len(heldout)==4 and set(features['groups'])==set(contract['source_split']['all_groups'])
    labels={g:report['labels']['G_labels'][g] for g in fit_groups}
    models={signal:fit_one([features['groups'][g][signal] for g in fit_groups],[labels[g] for g in fit_groups]) for signal in SIGNALS}
    result={'kind':'exploratory4B_fitted_models_for_prospective_external_validation','models':models,
        'fit_groups':fit_groups,'historical_heldout_groups':heldout,'fit_training_labels_only':labels,
        'fit_stage':0,'fit_policy_fingerprint':features['policy_fingerprint'],
        'source_features':ref(args.features),'source_early_report':ref(args.early_report),'source_analysis_contract':ref(args.contract),
        'prediction_rule':'intercept + slope * ((new_feature - frozen4B_fit_mean)/frozen4B_population_std); z=0 when scale=0',
        'feature_selection':False,'hyperparameter_search':False,'prediction_clipping':False,
        'late4B_or8B_label_refitting':False,'old4B_target_labels_were_already_open':True,
        'old4B_scores_are_prospective_validation':False,'8B_point_predictions_created':False,
        'scope':'New candidates remain separate from the original five and H1. No historical4B confirmatory success is claimed. External labels must remain closed until actual point forecasts are published.'}
    with args.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'models':models,'fit_source_count':12,'new_model_tokens':0,'external_point_predictions_not_yet_created':True}),flush=True)


if __name__=='__main__':main()
