"""Review completed original forward15 branches; never rerun or overwrite them.

Each invocation reviews one next completed,unreviewed branch, or the complete
batch. Actual scientific auditors are reused. Git publication is a separate step.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from bfcl_source_eval_contract import pinned
from bfcl_stage1_forward15_contract import validate
from bfcl_stage1_forward_bundle import verify_bundle
from report_bfcl_source_training import audit_job
from supervise_bfcl_stage_fork import ROOT, process_has_not_exited, read
from watch_research_events import record

SELF = 'pipeline/review_bfcl_stage1_forward15_training.py'
PROTOCOL = {'path': 'rl/BFCL_SOURCE_STAGE1_FORWARD15_PROTOCOL_20261015.json',
            'sha256': '0f661ed24ac04533be5d12529cf219942f8dfe3dd1af58de641c2eb352d611cf'}
POINTER = 'experiments/.runtime/ssh_stage1_forward15_train_launch_20261016.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def ref(root, path): return {'path': str(path.relative_to(root)), 'sha256': digest(path)}


def exclusive(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2);handle.write('\n')


def lines(path):
    return [json.loads(row) for row in path.read_text().splitlines() if row.strip()]


def context(root):
    master = pinned(root, PROTOCOL);validate(root, master)
    runtime = read(root/POINTER)
    if runtime['protocol'] != PROTOCOL: raise ValueError('Wrong actual training segment')
    authority = pinned(root, runtime['authority'])
    if authority['scope'] != 'stage1_fixed_parent_forward_training_15': raise ValueError('Wrong training scope')
    control = root/runtime['control_root']
    if authority['control_root'] != runtime['control_root']: raise ValueError('Wrong owned control root')
    launch = read(control/'launch.json')
    if launch['supervisor_pid'] != runtime['pid'] or launch['authority'] != runtime['authority']:
        raise ValueError('Actual segment launch identity changed')
    verify_bundle(root, authority['forward_bundle'], PROTOCOL, launch['started_unix'])
    for name, value in authority['control_input_sha256'].items():
        if digest(root/name) != value: raise ValueError('Running original controller changed: '+name)
    return master, runtime, authority, control, read(control/'queue.json')


def register(root, source, kind, facts, review, extra_paths, message):
    idx = read(root/'experiments/index.json');eid = 'E'+str(len(idx)+1).zfill(3)
    if any(e['source'] == source for e in idx): raise ValueError('Already registered original evidence;never duplicate')
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    idx.append({'id': eid, 'source': source, 'kind': kind,
                'sha256': hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest(),
                'facts': facts, 'batch': 'rl/bfcl_source_stage1_train_20261015', 'observed_at': now})
    review.update(status='reviewed', reviewed_at=now,
                  reviewer='Current primary SSH execution agent; not independent human review', goal_complete=False)
    exclusive(root/'experiments/reviews'/(eid+'.json'), review)
    with (root/'experiments'/(eid+'.md')).open('x') as handle:
        handle.write('# '+eid+': '+kind+'\n\n'+now+'\n\n'+review['conclusion']+'\n\n'+review['next_action']+'\n\n```json\n'+json.dumps(facts,ensure_ascii=False,indent=2)+'\n```\n')
    (root/'experiments/index.json').write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n');record(root,[])
    with (root/'experiments/NEXT.md').open('a') as handle:
        handle.write('\n'+now+' '+eid+'：'+review['conclusion']+' '+review['next_action']+'\n')
    with (root/'experiments/SCOPE_REVIEW.md').open('a') as handle:
        handle.write('\n### '+now+' '+eid+'：原15在线学习证据\n\n'+review['conclusion']+' 训练奖励、更新机会、fresh梯度和参数变化分别保留，均不代替迁移G/V；不依据这些局部结果删来源、改seed、追加剂量或修订封存预测。用户当前15/32范围保持，21题关闭，整个研究Goal未完成。\n')
    paths = list(dict.fromkeys([SELF,*extra_paths,'experiments/'+eid+'.md','experiments/reviews/'+eid+'.json',
             'experiments/index.json','experiments/README.md','experiments/events.jsonl','experiments/NEXT.md','experiments/SCOPE_REVIEW.md']))
    ignored = []
    for path in paths:
        p = root/path
        if p.suffix in ('.pt','.pth','.safetensors','.bin'): raise ValueError('Binary checkpoints stay at recorded server paths')
        if re.search(rb'gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|sk-[A-Za-z0-9_-]{30,}',p.read_bytes()):
            raise ValueError('Sensitive content refused: '+path)
        if subprocess.run(['git','check-ignore','-q',path],cwd=root).returncode == 0:
            if p.name not in ('README.md','adapter_config.json') or p.parent.name != 'adapter':
                raise ValueError('Unexpected ignored evidence: '+path)
            ignored.append(path)
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).strip():
        raise ValueError('Unexpected pre-existing staged changes')
    regular = [p for p in paths if p not in ignored]
    subprocess.run(['git','add','--',*regular],cwd=root,check=True)
    if ignored:subprocess.run(['git','add','-f','--',*ignored],cwd=root,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=root,check=True)
    subprocess.run(['git','commit','-m',message],cwd=root,check=True,stdout=subprocess.DEVNULL)
    return {'event': eid, 'review': ref(root,root/'experiments/reviews'/(eid+'.json')),
            'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'new_scientific_tokens_from_review': 0}


def review_one(root):
    master, runtime, authority, control, queue = context(root)
    idx = read(root/'experiments/index.json')
    candidates = [entry for entry in master['jobs'] if entry['name'] in queue['complete']
                  and not any(e['source'] == pinned(root,entry)['output']+'/summary.json' for e in idx)]
    if not candidates:
        return {'reviewed': [], 'queue_counts': {k:len(queue[k]) for k in ['complete','active','pending','failures']}}
    entry = candidates[0];job = pinned(root,entry);out = root/job['output']
    manifest_path = out/'evidence_manifest.json'
    existing = read(manifest_path) if manifest_path.exists() else None
    start = time.time()
    fresh = existing['fresh_raw_audit'] if existing else audit_job(job,entry['sha256'])
    elapsed = time.time()-start
    if existing:
        if existing['original_job'] != entry or existing['fresh_raw_audit_equals_saved'] is not True:
            raise ValueError('Existing partial review manifest belongs to different evidence')
        for name,value in existing['files'].items():
            if digest(out/name) != value['sha256']:raise ValueError('Existing reviewed raw bytes changed')
    if fresh != read(out/'audit.json') or fresh['status'] != 'passed': raise ValueError('Fresh full raw audit differs')
    process = read(control/(entry['name']+'.console.process.json'));run = read(out/'run.json');terminal = read(out/'supervisor.json')
    if (process['pid'] != run['pid'] or process_has_not_exited(process['pid']) or terminal['returncode'] != 0
            or terminal['hard_timeout'] or terminal['wall_seconds'] > job['hard_seconds']+30):
        raise ValueError('Original owned scientific process is not cleanly terminal')
    allocations = lines(out/'allocation.jsonl')
    exposure = {'task_pool_size':len(job['task_pool']),'visited_tasks':dict(Counter(row['task_id'] for row in allocations)),
                'task_sequence':[row['task_id'] for row in allocations],
                'mixed_reward_groups':sum(len(set(row['rewards']))>1 for row in allocations),
                'complete_group_token_overshoot':fresh['cost']['new_model_tokens']-job['token_budget']}
    files = {}
    for p in sorted(out.rglob('*')):
        if not p.is_file():continue
        if p == manifest_path:continue
        if p.is_symlink():raise ValueError('Unexpected symlink in scientific output')
        name = str(p.relative_to(out))
        publication = 'git_evidence' if p.suffix in ('.json','.jsonl','.md','.txt','.log') else 'server_artifact'
        files[name] = {'sha256':digest(p),'bytes':p.stat().st_size,'publication':publication}
    controls = {str(p.relative_to(root)):{'sha256':digest(p),'bytes':p.stat().st_size}
                for p in sorted(control.glob(entry['name']+'.*')) if p.is_file()}
    if existing:
        if files != existing['files'] or controls != existing['controls']:
            raise ValueError('Partial review inputs changed;preserve and investigate')
    else:exclusive(manifest_path,{'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'execution_protocol':PROTOCOL,
        'original_job':entry,'original_control_root':str(control.relative_to(root)),'fresh_raw_audit_equals_saved':True,
        'fresh_raw_audit':fresh,'fresh_replay_wall_seconds':elapsed,'audit_implementation':ref(root,root/'pipeline/report_bfcl_source_training.py'),
        'review_implementation':ref(root,root/SELF),'files':files,'controls':controls,'task_exposure':exposure,
        'original_owned_process':process,'terminal':terminal,'scientific_reruns':0,'new_scientific_tokens_from_review':0,
        'checkpoint_access':{'method':'SSH copy exact server artifact paths and verify SHA256;Git clone alone does not supply binary checkpoints.',
                             'directory':str(out),'server_files':[n for n,v in files.items() if v['publication']=='server_artifact']},
        'scope':'Original real online RL training acquisition and learner-state evidence only;no target G/V or predictor verdict.'})
    facts = {'case_id':entry['name'],'source_group':job['source_group'],'training_repeat':job['training_repeat'],
        'execution_protocol':PROTOCOL,'original_job':entry,'full_raw_audit':fresh,'exposure':exposure,
        'evidence_manifest':ref(root,manifest_path),'original_pid':process['pid'],'original_process_exited':True,
        'wall_seconds':terminal['wall_seconds'],'scientific_reruns':0,'review_added_scientific_tokens':0,'goal_complete':False}
    facts['existing_partial_review_manifest_reused'] = existing is not None
    conclusion = (entry['name']+'原分支完整：'+str(fresh['cost']['new_model_tokens'])+'新模型token、'+str(fresh['optimizer_steps'])+
        '次优化器更新、fresh非零梯度step='+str(fresh['fresh_nonzero_gradient_steps'])+'、参数变化step='+str(fresh['changed_parameter_steps'])+
        '；原完整组/allocator/token-mask/采样seed/完整learner状态raw重放等于保存审计，原PID已RC0退出且未超硬界。实际任务暴露和完整组超额单列，继承父成本不逐分支重计。这里只是训练证据，尚无本批迁移G/V。')
    extra = [str(manifest_path.relative_to(root)),*controls,
             *[str((out/n).relative_to(root)) for n,v in files.items() if v['publication']=='git_evidence']]
    return register(root,job['output']+'/summary.json','完整原forward15训练分支审阅',facts,
        {'evidence':[job['output']+'/summary.json',job['output']+'/audit.json',str(manifest_path.relative_to(root))],
         'conclusion':conclusion,'next_action':'继续原15未完成job；全部15审计审阅齐后，完整整批review再登记独立32评测控制。原20＋4预测保持冻结，原晚全量51/104延后。',
         'new_sampled_tokens':fresh['cost']['new_model_tokens'],'cost_accounting':'Original branch tokens charged once;CPU review adds0sampling.'},
        extra,'Review original '+entry['name']+' complete online RL raw evidence without reruns')


def review_complete(root):
    master,runtime,authority,control,queue = context(root)
    certificate_path = root/master['complete_training_audit_path'];certificate = read(certificate_path)
    idx = read(root/'experiments/index.json');source = str(certificate_path.relative_to(root))
    prior = [e for e in idx if e['source']==source]
    if prior:return {'already_reviewed':prior[0]['id'],'review':ref(root,root/'experiments/reviews'/(prior[0]['id']+'.json'))}
    if (len(queue['complete']) != 15 or any(queue[k] for k in ['active','pending','failures'])
            or process_has_not_exited(runtime['pid']) or read(control/'supervisor.json')['status'] != 'audited_complete'):
        raise ValueError('Actual original15 queue is not fully complete and exited')
    if (certificate['status'] != 'passed' or certificate['complete_jobs'] != 15 or certificate['execution_protocol'] != PROTOCOL
            or certificate['original_full51_complete'] is not False):
        raise ValueError('Wrong active15 complete certificate')
    rows=[];cases=[]
    for entry in master['jobs']:
        job=pinned(root,entry);out=root/job['output'];manifest=read(out/'evidence_manifest.json')
        if not any(e['source']==job['output']+'/summary.json' for e in idx):raise ValueError('Missing per-case raw execution review')
        if (manifest['original_job'] != entry or manifest['fresh_raw_audit_equals_saved'] is not True
                or manifest['fresh_raw_audit'] != read(out/'audit.json')):
            raise ValueError('Fresh case audit/review identity changed')
        for name,value in manifest['files'].items():
            if digest(out/name) != value['sha256']:raise ValueError('Reviewed immutable raw/checkpoint bytes changed')
        rows.append(manifest['fresh_raw_audit']);cases.append({'job':entry,'manifest':ref(root,out/'evidence_manifest.json'),'exposure':manifest['task_exposure']})
    if certificate['jobs'] != rows:raise ValueError('Complete15 certificate does not equal the reviewed original raw audits')
    total=sum(row['cost']['new_model_tokens'] for row in rows)
    if total != certificate['new_sampled_tokens'] or total > master['new_training_token_ceiling']:
        raise ValueError('Complete15 actual cost differs or exceeds budget')
    facts={'execution_protocol':PROTOCOL,'complete_audit':ref(root,certificate_path),'complete_jobs':15,
        'fresh_raw_audits_match_saved':True,'fresh_per_case_audits_reused_with_all_raw_and_checkpoint_hashes_reverified':True,
        'no_failed_or_omitted_active_branches':True,'cases':cases,'actual_new_training_tokens':total,
        'nominal_tokens':master['nominal_new_training_tokens'],'complete_group_overshoot_tokens':total-master['nominal_new_training_tokens'],
        'optimizer_steps':sum(row['optimizer_steps'] for row in rows),
        'fresh_nonzero_gradient_steps':sum(len(row['fresh_nonzero_gradient_steps']) for row in rows),
        'changed_parameter_steps':sum(len(row['changed_parameter_steps']) for row in rows),
        'ancestor_tokens_recharged':0,'target_evaluation_performed':False,'full51_complete':False,'goal_complete':False}
    return register(root,source,'原forward15整批训练完整审阅',facts,
        {'kind':'complete_stage1_forward15_training_execution_review','execution_protocol':PROTOCOL,
         'complete_audit':ref(root,certificate_path),'fresh_raw_audits_match_saved':True,'no_failed_or_omitted_active_branches':True,
         'evidence':[source,*[case['manifest']['path'] for case in cases]],
         'conclusion':'当前批准的原15训练全部完成，逐条原raw审阅/完整learner检查点及整批证书一致，所有原字节复核通过；实际剂量、更新机会和任务暴露不同，不能宣称严格等计算量或已获正迁移。原36训练继续延后，本批没有晚内拟合，不能冒称完整51或104。',
         'next_action':'按已批准32面板准备原job及实际终点SHA，分别绑定当前评测归属、现有停止标记和原预测出版证明，发布后执行32。完成原G/V、固定早五模型及H1误差/方向/重复/成本报告后再作扩展判断。',
         'new_sampled_tokens':0,'cost_accounting':'Aggregate prior15 original costs only;CPU integrity review adds0sampling and does not recharge parent/probe.'},
        [source,str((control/'supervisor.json').relative_to(root)),str((control/'training_control_result.json').relative_to(root))],
        'Review complete original forward15 training and actual costs before separate32 target admission')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--complete',action='store_true');args=parser.parse_args()
    print(json.dumps(review_complete(ROOT) if args.complete else review_one(ROOT),ensure_ascii=False,indent=2),flush=True)
