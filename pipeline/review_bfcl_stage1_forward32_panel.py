"""Review one completed original forward32 panel without rerunning inference.

Reuse the frozen raw auditor. Publish small individual evidence commits;
the caller verifies GitHub publication before reviewing another panel.
"""
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

from bfcl_stage1_forward32_ssh import load_manifest
from report_bfcl_stage1_forward32 import audit_job
from supervise_bfcl_stage_fork import ROOT, read, process_has_not_exited
from watch_research_events import record

SELF = 'pipeline/review_bfcl_stage1_forward32_panel.py'
POINTER = 'experiments/.runtime/ssh_stage1_forward32_eval_launch_cont1_20261016.json'
MANIFEST = {'path': 'rl/bfcl_source_stage1_forward_jobs_20261015/manifest.json',
            'sha256': '01bd201c1ccfab44582e7bfc61148ea0216eccb917197496a1f144548a9e1191'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):h.update(block)
    return h.hexdigest()


def ref(root, path):return {'path': str(path.relative_to(root)), 'sha256': digest(path)}


def exclusive(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2);handle.write('\n')


def review_one(root):
    runtime = read(root/POINTER)
    if runtime['manifest'] != MANIFEST:raise ValueError('Wrong original32 manifest')
    manifest, master, binding, jobs = load_manifest(root, MANIFEST)
    control = root/runtime['control_root'];authority = read(root/runtime['authority']['path'])
    if digest(root/runtime['authority']['path']) != runtime['authority']['sha256']:
        raise ValueError('Actual evaluation authority changed')
    launch = read(control/'launch.json')
    if launch['supervisor_pid'] != runtime['pid'] or launch['authority'] != runtime['authority']:
        raise ValueError('Wrong original evaluation control identity')
    queue = read(control/'queue.json');idx = read(root/'experiments/index.json')
    done = {row['source'] for row in idx}
    candidates = [(entry,job) for entry,job in zip(manifest['jobs'],jobs)
                  if entry['name'] in queue['complete'] and job['output']+'/summary.json' not in done]
    if not candidates:
        return {'reviewed': [], 'queue_counts': {k:len(v) for k,v in queue.items()}}
    entry,job = candidates[0];out = root/job['output'];proof_path = out/'evidence_manifest.json'
    existing = read(proof_path) if proof_path.exists() else None
    start = time.monotonic()
    if existing:
        if existing['original_job'] != entry or existing['fresh_full_raw_audit_matches_saved'] is not True:
            raise ValueError('Different existing partial review evidence')
        for name,value in existing['files'].items():
            if digest(out/name) != value['sha256'] or (out/name).stat().st_size != value['bytes']:
                raise ValueError('Previously reviewed immutable raw bytes changed')
        fresh = existing['fresh_raw_audit']
    else:fresh = audit_job(root,root/entry['path'])
    if fresh != read(out/'audit.json') or fresh['status'] != 'passed':raise ValueError('Fresh raw audit differs')
    run,summary,terminal = [read(out/n) for n in ('run.json','summary.json','supervisor.json')]
    process = read(control/(entry['name']+'.console.process.json'))
    if (run['pid'] != process['pid'] or process_has_not_exited(process['pid']) or terminal['returncode'] != 0
            or terminal['hard_timeout'] or process['hard_seconds'] != manifest['hard_seconds_per_panel']
            or terminal['wall_seconds'] > manifest['hard_seconds_per_panel']+30):
        raise ValueError('Original scientific process is not cleanly terminal')
    command = process['command']
    if (command[command.index('--job')+1] != str(root/entry['path'])
            or command[command.index('--job-sha256')+1] != entry['sha256']
            or run['current_eval_authority'] != runtime['authority']
            or summary['status'] != 'complete' or summary['completed_tasks'] != 32
            or not summary['weights_unchanged'] or run['optimizer_updates'] != 0):
        raise ValueError('Original job, source policy or complete target identity changed')
    prior_controls = authority.get('prior_admission_control_roots',[])
    if any((root/folder/(entry['name']+'.console.process.json')).exists() for folder in prior_controls):
        raise ValueError('Unexpected previous scientific execution; inspect all lifetimes before review')
    files = {}
    for path in sorted(out.rglob('*')):
        if not path.is_file() or path == proof_path:continue
        if path.is_symlink() or path.suffix not in ('.json','.jsonl','.md','.txt','.log'):
            raise ValueError('Unexpected target evidence artifact: '+str(path))
        files[str(path.relative_to(out))] = {'sha256':digest(path),'bytes':path.stat().st_size,'publication':'git_evidence'}
    controls = {str(path.relative_to(root)):{'sha256':digest(path),'bytes':path.stat().st_size}
                for path in sorted(control.glob(entry['name']+'.*')) if path.is_file()}
    if existing:
        if files != existing['files'] or controls != existing['controls']:raise ValueError('Partial review evidence changed')
    else:
        exclusive(proof_path,{'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'original_job':entry,'original_manifest':MANIFEST,'original_control_root':runtime['control_root'],
            'fresh_full_raw_audit_matches_saved':True,'fresh_raw_audit':fresh,
            'fresh_replay_wall_seconds':time.monotonic()-start,'files':files,'controls':controls,
            'original_owned_process':process,'terminal':terminal,
            'audit_implementation':ref(root,root/'pipeline/report_bfcl_stage1_forward32.py'),
            'review_implementation':ref(root,root/SELF),'scientific_reruns':0,'new_scientific_tokens_from_review':0,
            'scope':'One original32-task panel, all-role forecast timing and frozen policy checked. No complete G/V or predictor conclusion; shared sources, tasks, ancestors and policies remain dependent.'})
    now = datetime.datetime.now(datetime.timezone.utc).isoformat();eid = 'E'+str(len(idx)+1).zfill(3)
    tokens = fresh['cost']['new_model_tokens'];source = job['output']+'/summary.json'
    facts = {'original_job':entry,'full_raw_audit':fresh,'evidence_manifest':ref(root,proof_path),
             'original_pid':process['pid'],'original_process_exited':True,'worker_wall_seconds':terminal['wall_seconds'],
             'scientific_reruns':0,'review_added_scientific_tokens':0,'goal_complete':False}
    conclusion = (entry['name']+'原32题完整结束；原任务/seed/政策指纹、weights不变、0优化器更新及全部generation/mask/奖励/成本raw重放等于保存审计，所有角色的实际事前预测出版时间与当前评测归属核对通过。原科学进程单次RC0退出，实际'+str(tokens)+'新模型token。单面板仍不构成完整G/V或预测结论。')
    next_action = '继续原32尚未完成面板；全部完成后审阅完整执行链并运行冻结15/32科学报告，原五模型/H1和gpt21描述分解单列，不依部分结果改组、seed或预测。'
    idx.append({'id':eid,'source':source,'kind':'complete raw stage1 forward target panel',
                'sha256':hashlib.sha256(json.dumps(facts,sort_keys=True).encode()).hexdigest(),
                'facts':facts,'batch':manifest['output_root'],'observed_at':now})
    exclusive(root/'experiments/reviews'/(eid+'.json'),{'status':'reviewed','reviewed_at':now,
        'reviewer':'Current primary SSH execution agent; not independent human review',
        'evidence':[source,job['output']+'/audit.json',str(proof_path.relative_to(root))],
        'conclusion':conclusion,'next_action':next_action,'new_sampled_tokens':tokens,
        'cost_accounting':'Original panel tokens charged once; CPU replay adds zero scientific sampling.','goal_complete':False})
    with (root/'experiments'/(eid+'.md')).open('x') as handle:
        handle.write('# '+eid+': 原前向迁移面板 '+entry['name']+'\n\n'+now+'\n\n'+conclusion+'\n\n'+next_action+'\n')
    (root/'experiments/index.json').write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n');record(root,[])
    current_queue = read(control/'queue.json');next_path = root/'experiments/NEXT.md';text = next_path.read_text()
    counts = {k:len(v) for k,v in current_queue.items()}
    status = now+'快照：原15训练全齐；原32前向评测'+str(counts['complete'])+'完成/'+str(counts['active'])+'活跃/'+str(counts['pending'])+'待执行/'+str(counts['failures'])+'当前故障，新增完整raw审阅'+eid+'。当前控制PID'+str(runtime['pid'])+'；原预测已先发布，所有role受守卫。整批后报告G/V和固定误差，晚全51/104延后，21题关闭，Goal未完成。'
    text,n = re.subn(r'(## 当前状态\n\n)[^\n]+',lambda m:m.group(1)+status,text,count=1)
    if n != 1:raise ValueError('NEXT current-status section missing')
    next_path.write_text(text+'\n'+now+' '+eid+'：'+conclusion+' '+next_action+'\n')
    paths = [SELF,str(proof_path.relative_to(root)),*controls,*[str((out/n).relative_to(root)) for n in files],
             'experiments/'+eid+'.md','experiments/reviews/'+eid+'.json','experiments/index.json',
             'experiments/events.jsonl','experiments/README.md','experiments/NEXT.md']
    for name in paths:
        if re.search(rb'gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|sk-[A-Za-z0-9_-]{30,}|X-Amz' + rb'-Signature=',(root/name).read_bytes()):
            raise ValueError('Sensitive content refused: '+name)
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).strip():raise ValueError('Unexpected staged changes')
    subprocess.run(['git','add','--',*paths],cwd=root,check=True)
    subprocess.run(['git','diff','--cached','--check','--','.',':(exclude)*.console.log',':(exclude)*.audit.console.log'],cwd=root,check=True)
    subprocess.run(['git','commit','-m','Review original '+entry['name']+' complete raw forward evidence without reruns'],cwd=root,check=True,stdout=subprocess.DEVNULL)
    return {'event':eid,'panel':entry['name'],'new_model_tokens':tokens,'queue_counts':counts,
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}


if __name__ == '__main__':print(json.dumps(review_one(ROOT),ensure_ascii=False),flush=True)
