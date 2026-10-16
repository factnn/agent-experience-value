"""Describe a complete, already scored Stage1 batch without fitting or sampling."""
import argparse,csv,datetime,hashlib,itertools,json,math,os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MODELS=['fit_mean','primary_task_beta','current_pooled_beta','historical_frontier','historical_coverage']
NAMES={'fit_mean':'冻结均值','primary_task_beta':'任务级Beta','current_pooled_beta':'组合级Beta','historical_frontier':'Frontier','historical_coverage':'Coverage','separate_H1':'H1成功率（辅助）'}
def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def reference(root,p):return {'path':str(p.relative_to(root)),'sha256':digest(p)}
def mean(xs):
 xs=list(xs);return math.fsum(xs)/len(xs)
def sign(x):return (x>0)-(x<0)
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows]])
def number(x):return f'{x:.6g}'
def pp(x):return f'{100*x:.4f}'

def build(root,report_path,report_sha,description_path,description_sha):
 assert digest(report_path)==report_sha and digest(description_path)==description_sha
 d=read(report_path);e=read(description_path)
 assert d['status']=='passed' and d['raw_replay']=={'training_jobs':15,'target_panels':32,'saved_audits_equal_fresh':True}
 assert d['reserved21_accessed'] is False and d['within_late_fitting_performed'] is False
 assert e['original_main_scores_unchanged'] is True and e['recalibrated_predictions_created'] is False
 groups=d['labels']['source_groups'];assert len(groups)==4
 early=read(root/d['evidence']['early_report']['path'])
 master=read(root/d['evidence']['execution_protocol']['path'])
 bundle=read(root/d['evidence']['forward_bundle']['path'])
 probe=read(root/bundle['complete_probe_audit']['path'])
 phases={'probe':probe['rows'],'training':d['costs']['training_rows'],'evaluation':d['costs']['evaluation_panels']}
 costs={}
 for name,rows in phases.items():
  costs[name]={k:sum(x['cost'][k] for x in rows) if all(k in x.get('cost',{}) for x in rows) else None for k in ['new_model_tokens','input_token_positions_excluding_padding','external_tokens','rollouts']}
 assert sum(x['new_model_tokens'] for x in costs.values())==d['costs']['late_total_new_model_tokens_once']
 gains=[];training=[];fingerprints={}
 for g in groups:
  for row in d['labels']['groups'][g]['training_repeats']:
   panels={p['evaluation_repeat']:p for p in row['panels']};repeat=row['training_repeat']
   gains.append({'source_group':g,'training_repeat':repeat,'G':row['G'],'V':row['V'],'evaluation0_G':panels[0]['G'],'evaluation1_G':panels[1]['G']})
 for entry,audit in zip(master['jobs'],d['costs']['training_rows']):
  job=read(root/entry['path']);summary=read(root/job['output']/'summary.json');fp=summary['final_fingerprint']
  fingerprints.setdefault(fp,[]).append(job['case_id'])
  training.append({'case_id':job['case_id'],'source_group':job['source_group'],'training_repeat':job['training_repeat'],'new_model_tokens':audit['cost']['new_model_tokens'],'nominal_tokens':job['token_budget'],'complete_group_overshoot':audit['cost']['new_model_tokens']-job['token_budget'],'optimizer_steps':audit['optimizer_steps'],'fresh_nonzero_gradient_steps':len(audit['fresh_nonzero_gradient_steps']),'changed_parameter_steps':len(audit['changed_parameter_steps']),'policy_fingerprint':fp})
 duplicates=[];panels={(p['policy_case'],p['evaluation_repeat']):p for p in d['costs']['evaluation_panels']}
 for fp,cases in fingerprints.items():
  for a,b in itertools.combinations(cases,2):
   for seed in range(2):
    left,right=panels[(a,seed)],panels[(b,seed)]
    compared={leaf:{'left_sha256':left['source_files'][leaf],'right_sha256':right['source_files'][leaf],'byte_identical':left['source_files'][leaf]==right['source_files'][leaf]} for leaf in ['outcomes.jsonl','rollouts.jsonl','generation_calls.jsonl']}
    duplicates.append({'left':a,'right':b,'evaluation_repeat':seed,'policy_fingerprint':fp,'files':compared,'all_three_behavioral_files_byte_identical':all(v['byte_identical'] for v in compared.values())})
 pairs=[]
 for a,b in itertools.combinations(groups,2):
  left={x['training_repeat']:x for x in gains if x['source_group']==a};right={x['training_repeat']:x for x in gains if x['source_group']==b}
  diffs=[left[i]['G']-right[i]['G'] for i in range(3)]
  seedmeans=[mean(left[i][f'evaluation{s}_G']-right[i][f'evaluation{s}_G'] for i in range(3)) for s in range(2)]
  direction=sign(diffs[0]);stable=direction!=0 and all(sign(v)==direction for v in diffs)
  pairs.append({'left':a,'right':b,'paired_differences_by_training_repeat':diffs,'differences_averaged_over_training_by_evaluation_repeat':seedmeans,'strict_same_direction_all_three_training_repeats':stable,'also_same_direction_both_evaluation_repeat_means':stable and all(sign(v)==direction for v in seedmeans)})
 predictions=[]
 for g in groups:
  for model in [*MODELS,'separate_H1']:
   pred=d['sealed_separate_H1_predictions']['point_predictions'][g] if model=='separate_H1' else d['sealed_predictions']['point_predictions'][g][model]
   params=d['standardized_frozen_predictions'][g].get(model,{})
   predictions.append({'source_group':g,'model':model,'registry':'separate_auxiliary_H1' if model=='separate_H1' else 'original_five','late_signal':params.get('late_signal'),'frozen_early_mean':params.get('frozen_early_mean'),'frozen_early_population_std':params.get('frozen_early_population_std'),'z':params.get('z'),'frozen_intercept':params.get('frozen_intercept',pred),'frozen_slope':params.get('frozen_slope',0),'frozen_lambda':params.get('frozen_lambda'),'predicted_G':pred,'observed_G':d['labels']['G_labels'][g],'observed_V':d['labels']['V_labels'][g],'signed_error':pred-d['labels']['G_labels'][g],'squared_error':(pred-d['labels']['G_labels'][g])**2})
 primary=d['prediction_results']['original_five']['MSE'];base=primary['fit_mean'];task=primary['primary_task_beta'];pool=primary['current_pooled_beta']
 task_decomp=e['late_original_five']['primary_task_beta'];delta=task_decomp['differences_from_frozen_mean_baseline']
 facts={'mean_G':mean(d['labels']['G_labels'].values()),'mean_V':mean(d['labels']['V_labels'].values()),'mean_early_same_four_G':mean(early['labels']['G_labels'][g] for g in groups),'mean_uniform_G':mean(v['G'] for v in d['labels']['uniform_G_by_training_repeat'].values()),'task_MSE_reduction_against_frozen_mean_fraction':(base-task)/base,'task_MSE_reduction_against_pooled_fraction':(pool-task)/pool,'task_RMSE_percentage_points':100*math.sqrt(task),'mean_baseline_RMSE_percentage_points':100*math.sqrt(base),'task_MSE_fraction_overall_squared_bias':task_decomp['overall_squared_bias']/task,'task_advantage_fraction_from_squared_bias_difference':delta['overall_squared_bias']/delta['MSE_original'],'task_beats_mean_sources':sum(x['original_five_squared_errors']['primary_task_beta']<x['original_five_squared_errors']['fit_mean'] for x in d['prediction_results']['source_rows']),'task_beats_mean_training_repeat_views':sum(x['MSE']['primary_task_beta']<x['MSE']['fit_mean'] for x in d['prediction_results']['by_training_repeat'].values()),'task_beats_mean_evaluation_repeat_views':sum(x['MSE']['primary_task_beta']<x['MSE']['fit_mean'] for x in d['prediction_results']['by_evaluation_repeat'].values()),'strict_source_pairs_all_training_repeats':sum(p['strict_same_direction_all_three_training_repeats'] for p in pairs),'source_pairs_also_both_evaluation_means':sum(p['also_same_direction_both_evaluation_repeat_means'] for p in pairs),'distinct_trainable_weight_fingerprints':len(fingerprints),'training_policy_instances':15}
 plot={'source_groups':groups,'gains':gains,'G':d['labels']['G_labels'],'V':d['labels']['V_labels'],'models':MODELS,'early_MSE':{m:e['early_original_five'][m]['MSE_original'] for m in MODELS},'late_MSE':primary,'late_components':{m:{k:e['late_original_five'][m][k] for k in ['overall_squared_bias','centered_between_source_error']} for m in MODELS},'separate_H1_MSE':d['prediction_results']['separate_H1']['MSE'],'mean_uniform_G':facts['mean_uniform_G'],'source_report':reference(root,report_path),'source_error_description':reference(root,description_path)}
 claims=[
  ('真实在线RL干预和迁移评测已完成','E321、原15训练证书、32评测证书及逐面板E325–E357（E326为后处理恢复）','更新/梯度/参数变化与迁移收益分别报告；审计通过不是正收益。'),
  ('四来源晚阶段平均G为正','主报告labels与gains_by_training_repeat.csv','仅MathAPI+TradingBot在三个训练重复均G>0；评测seed仍有波动。'),
  ('四来源平均V均为正','主报告labels与对应Uniform','晚阶段Uniform平均G为负；V>0不推出G>0，也不去除剂量和动量影响。'),
  ('任务级Beta在本次前向检验的点MSE低于冻结均值','原封存预测、完整主报告prediction_results','4个来源、同32题、新eval seeds；不是稳定采样规则或整类方法胜利。'),
  ('局部机会预测优势在两个学习状态出现','原Stage0报告、E303完整晚阶段探测','共享模型家族、任务与祖先；局部Brier优势不是迁移收益证明。'),
  ('误差同时包含整体水平和来源关系变化','error_description.json的原MSE恒等分解及同四来源变化','没有修正截距、重新拟合、创造新预测或把六个对比当独立复制。'),
  ('部分零新梯度分支仍有参数变化和正G','E313、E316、固定父优化器上下文及本报告','与继承动量解释一致；未做数值反事实重放，不能归因于新经验奖励梯度。'),
  ('两个训练重复的政策和部分评测证据重复','post_batch_description.json的SHA比较','保留原三重复和实际成本；不能把相同政策/seed下的副本当独立条件。'),
  ('已有可追溯的原始证据与检查点取得方式','docs/stage1_forward15_reproduction.md与原evidence_manifest','Git不含模型二进制；当前环境元数据不是补写的历史安装连续性证明。'),
  ('整个研究Goal尚未完成','原scope、NEXT与SCOPE_REVIEW','机制广度、ID、规模/家族/环境外推及21题另行协议仍有缺口；不自动补回完整晚51/104。')]
 return d,e,{'kind':'post_batch_descriptive_evidence','computed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_report':reference(root,report_path),'source_error_description':reference(root,description_path),'facts':facts,'costs_by_phase':costs,'training_rows':training,'gains_by_training_repeat':gains,'frozen_prediction_rows':predictions,'duplicate_policy_behavioral_files':duplicates,'matched_source_pairs':pairs,'claims':[dict(claim=a,evidence=b,boundary=c) for a,b,c in claims],'new_scientific_tokens':0,'new_gpu_jobs':0,'model_refitted':False,'original_primary_results_unchanged':True,'reserved21_accessed':False,'goal_complete':False},plot

def write_csv(path,rows):
 with path.open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def render(d,e,s):
 f=s['facts'];groups=d['labels']['source_groups'];scores=d['prediction_results']['original_five']['MSE']
 parts=['# Stage1：冻结早期预测的15训练／32面板前向验证',
 f"完整原始重放通过：15条在线RL训练、32个目标面板。四来源平均G为＋{pp(f['mean_G'])}个百分点，平均V为＋{pp(f['mean_V'])}个百分点。原任务级Beta点预测MSE比冻结均值低{100*f['task_MSE_reduction_against_frozen_mean_fraction']:.2f}%，但绝对误差以整体高估为主，来源排序的重复稳定性有限。以下是本固定父状态、同四来源与同32题条件下的结果；不等于整个研究Goal完成。",
 '## 研究问题、协议与实际执行',
 '研究对象是在线经验获取—完整学习状态下的RL更新—训练后迁移收益。共同起点固定为原Stage0 Uniform repeat0完整7步终点，未按分数换祖先。原五个早期预测器的模型、标准化与20个实际晚阶段点预测，以及独立H1的4个预测，均在晚训练和全部目标标签之前封存并核验发布。本批按用户事前决定只做原四个留出来源×3训练重复＋3 Uniform；没有晚阶段内拟合，原完整晚51/104延后。',
 'G是终点相对同阶段共同起点的收益；V再减去对应训练重复、对应评测seed的Uniform收益。先配对相同任务和seed，再平均2个评测重复及3个训练重复。来源、任务、训练重复、评测重复和共享祖先形成依赖。',
 '## 逐来源收益与重复差异',
 table(['来源','早G(pp)','晚G(pp)','晚V(pp)','晚G：r0/r1/r2(pp)','晚V：r0/r1/r2(pp)'],[[g,pp(d['same_four_source_stage_comparison'][g]['early']['G']),pp(d['labels']['G_labels'][g]),pp(d['labels']['V_labels'][g]),' / '.join(pp(v) for v in d['repeat_stability']['groups'][g]['G_by_training_repeat']),' / '.join(pp(v) for v in d['repeat_stability']['groups'][g]['V_by_training_repeat'])] for g in groups]),
 f"晚阶段Uniform的G为−3.1250、−1.5625、0个百分点，平均{pp(f['mean_uniform_G'])}个百分点。因此本批四个平均V均为正，不能解释成每个来源都稳定带来正G。例如TravelAPI r1的G=0，而V=＋1.5625pp。四来源平均G从早期{pp(f['mean_early_same_four_G'])}pp降至{pp(f['mean_G'])}pp；变化不是所有来源等量下移，具体见error_description.json。",
 f"只有MathAPI+TradingBot在三个训练重复的平均G均严格为正。六个匹配来源对中，{f['strict_source_pairs_all_training_repeats']}个在三个训练重复同向；其中{f['source_pairs_also_both_evaluation_means']}个同时在两个评测重复的跨训练平均对比中保持方向。它们是描述性敏感性分析，不是独立重复、置信区间或显著性判据。完整配对值见post_batch_description.json与gains_by_training_repeat.csv；G/V同类排序不是第二份复制。",
 '## 原封存预测与实测',
 table(['来源',*[NAMES[m]+'预测G(pp)' for m in MODELS],'H1预测G(pp)','实测G(pp)'],[[g,*[pp(d['sealed_predictions']['point_predictions'][g][m]) for m in MODELS],pp(d['sealed_separate_H1_predictions']['point_predictions'][g]),pp(d['labels']['G_labels'][g])] for g in groups]),
 table(['原五模型','Stage0原留出MSE(pp²)','Stage1前向MSE(pp²)','Stage1 RMSE(pp)'],[[NAMES[m],number(e['early_original_five'][m]['MSE_original']*10000),number(scores[m]*10000),number(math.sqrt(scores[m])*100)] for m in MODELS]),
 f"独立辅助H1的晚阶段MSE为{d['prediction_results']['separate_H1']['MSE']*10000:.6f} pp²。H1早期拟合是在Stage0标签开放后另行登记的探索模型；本次4个晚预测为事前封存，不能把H1写成原五模型中的第六个确认性模型。",
 f"任务级Beta晚阶段RMSE为{f['task_RMSE_percentage_points']:.4f}pp，冻结均值为{f['mean_baseline_RMSE_percentage_points']:.4f}pp。其点MSE低于均值{100*f['task_MSE_reduction_against_frozen_mean_fraction']:.2f}%，低于组合级Beta约{100*f['task_MSE_reduction_against_pooled_fraction']:.2f}%。任务级相对均值在{f['task_beats_mean_sources']}/4来源、{f['task_beats_mean_training_repeat_views']}/3训练重复视图、{f['task_beats_mean_evaluation_repeat_views']}/2评测重复视图中改善；这些视图共享同一组封存预测及相关标签，不是独立复制。原Stage0任务级Beta未胜过均值的结果继续保留。",
 '每个来源的原始特征、早期均值/标准差、z、截距、斜率、lambda、原预测、实测G/V及误差均在frozen_predictions_and_observations.csv。原任务级/组合级/Frontier/Coverage斜率为正，辅助H1斜率为负；MSE降低不推出高信号来源更值得采样。',
 '## 整体水平与来源关系的误差分解',
 table(['模型','原MSE(pp²)','整体偏差平方(pp²)','中心化来源误差(pp²)','平均预测−实测(pp)'],[[NAMES[m],number(e['late_original_five'][m]['MSE_original']*10000),number(e['late_original_five'][m]['overall_squared_bias']*10000),number(e['late_original_five'][m]['centered_between_source_error']*10000),pp(e['late_original_five'][m]['mean_signed_error'])] for m in MODELS]),
 f"使用原预测的恒等式MSE=平均误差²＋中心化误差。任务级Beta的{100*f['task_MSE_fraction_overall_squared_bias']:.2f}% MSE属于整体偏差平方；它相对均值的MSE改善约{100*f['task_advantage_fraction_from_squared_bias_difference']:.2f}%来自这一分量差，其余来自中心化来源误差差。组合级Beta整体高估较少但来源间误差更大；Coverage相对均值的额外误差全部来自中心化来源分量。这里没有用晚标签校正截距或报告校正后预测。六个两两对比只编码同一个中心化分量。",
 '## 完整学习状态与有效依赖',
 '本批继承完整AdamW状态。E313/E316的MessageAPI+VehicleControlAPI r0/r1均有5次更新、0次新非零梯度、5次参数变化；父状态动量非零且weight decay=0，与继承动量作用一致，未做数值反事实重放。它们本次平均G均为＋1.5625pp，不能归因为新经验提供了有效奖励梯度。G包含完整学习状态下的获取与更新总作用，V没有剥离动量、步数、任务暴露和剂量差异。',
 f"15个训练终点实际只有{f['distinct_trainable_weight_fingerprints']}个不同的可训练权重指纹。相同政策对在相同评测seed下的outcomes、rollouts和generation_calls字节SHA比较见post_batch_description.json；原三重复和两次真实推理成本全部保留，不把副本冒充独立政策。",
 '## 实际成本与失败边界',
 table(['环节','新生成模型token','输入token位置（不含padding）','外部tool token','rollouts'],[[name,*[s['costs_by_phase'][name][k] for k in ['new_model_tokens','input_token_positions_excluding_padding','external_tokens','rollouts']]] for name in ['probe','training','evaluation']]),
 f"本晚阶段三环节合计新生成{d['costs']['late_total_new_model_tokens_once']:,} token，共享探测只计一次，继承父状态成本重计为0。训练实际{d['costs']['training_new_tokens']:,}，名义{d['costs']['training_nominal_tokens']:,}，完整组超额{d['costs']['training_complete_group_overshoot_tokens']:,}（{100*d['costs']['training_complete_group_overshoot_tokens']/d['costs']['training_nominal_tokens']:.2f}%），仍在登记上限内。113次更新、33次新非零梯度步骤、113次参数变化。逐分支剂量见training_dose.csv。训练与评测worker wall之和分别为{d['costs']['training_worker_wall_seconds']/3600:.3f}与{d['costs']['evaluation_worker_wall_seconds']/3600:.3f}小时；它们包含加载/CPU环境工作，不是GPU-active时间或严格FLOPs。输入位置按实际调用重复计数，不是独特数据量；外部tool token不混入生成预算。",
 '原32科学子进程均RC0且未触发硬超时，无科学任务重跑。首次评测准入的出版查询故障（E323，0科学child）、证据后处理误报（E326）和后续Git/SSH查询或传输故障另列于完整执行审阅。发布故障不当作训练负结果，也不因此重新采样。成本范围是原Stage1探测＋15训练＋32评测，不是整个旧项目全部工程试验总账。',
 '## Claim–evidence与未完成范围',table(['陈述','证据','结论边界'],[[x['claim'],x['evidence'],x['boundary']] for x in s['claims']]),
 '仍未覆盖：晚阶段内重拟合及完整16来源排序、不同机制候选的充分比较、独立ID收益、模型规模/家族与环境外推，以及另行协议下的21道保留题。原五模型中多数是同类代理或历史结构信号，不能据此声称机制广度已经完成。下一科学批次须说明关键解释、正/负/不确定结果如何改变决定、资源上限和停止条件；本报告不自动恢复完整晚51/104，也不宣布整个Goal完成。',
 '## 文件、复现与审阅归属',
 '原始机器结果：audit_status.json（完整15/32原始重放）；解释分解：error_description.json；本页及post_batch_description.json由执行agent整理，不是独立人类复核。冻结协议、原始输入、点预测、当前实际环境及检查点取得方式见../../docs/stage1_forward15_reproduction.md。所有模型二进制仍按原manifest在授权服务器取得，Git clone不是完整模型包。',
 '![训练/评测重复、收益与原预测误差](figures/gains_and_prediction_errors.png)',
 '图中训练重复的细线连接两个评测seed，属于原始配对范围而非置信区间；菱形是三个训练重复均值。误差堆叠图使用原MSE恒等分解。作图输入为plot_data.json，不生成新模型预测。']
 return '\n\n'.join(parts)+'\n'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',required=True);p.add_argument('--report-sha256',required=True);p.add_argument('--description',required=True);p.add_argument('--description-sha256',required=True);a=p.parse_args()
 report=ROOT/a.report;description=ROOT/a.description;out=report.parent
 d,e,s,plot=build(ROOT,report,a.report_sha256,description,a.description_sha256)
 targets=['post_batch_description.json','plot_data.json','frozen_predictions_and_observations.csv','gains_by_training_repeat.csv','training_dose.csv','README.md','claim_evidence.csv']
 assert not any((out/name).exists() for name in targets),'Do not overwrite existing descriptive artifacts'
 for name,value in [('post_batch_description.json',s),('plot_data.json',plot)]:
  with (out/name).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 write_csv(out/'frozen_predictions_and_observations.csv',s['frozen_prediction_rows']);write_csv(out/'gains_by_training_repeat.csv',s['gains_by_training_repeat']);write_csv(out/'training_dose.csv',s['training_rows']);write_csv(out/'claim_evidence.csv',s['claims'])
 with (out/'README.md').open('x') as f:f.write(render(d,e,s))
 (out/'figures').mkdir(exist_ok=True)
 print(json.dumps({'status':'described','source_report_sha256':a.report_sha256,'facts':s['facts'],'duplicate_policy_files':s['duplicate_policy_behavioral_files'],'new_scientific_tokens':0,'goal_complete':False}),flush=True)

if __name__=='__main__':main()
