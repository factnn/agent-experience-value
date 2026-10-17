"""Audit the one completed Qwen3-8B measurement without new model inference."""
from pathlib import Path
import datetime,hashlib,json,re,subprocess
import run_bfcl_intervention as r
from report_bfcl_intervention import audit_sampling,lines
from watch_research_events import record

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def ref(p):return {'path':str(p.relative_to(r.ROOT)),'sha256':sha(p)}
def new(p,d):
 with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
 root=r.ROOT;runtime=read(root/'experiments/.runtime/ssh_qwen8_capacity_launch_20261017.json')
 protocol_path=root/runtime['protocol']['path'];assert sha(protocol_path)==runtime['protocol']['sha256']
 p=read(protocol_path);out=root/p['output'];control=root/p['control_root'];summary=read(out/'summary.json')
 assert not (out/'audit.json').exists() and summary['status']=='complete'
 terminal=read(out/'supervisor.json');launch=read(control/'launch.json');process=read(control/'capacity.console.process.json');run=read(out/'run.json')
 assert terminal==read(control/'result.json') and terminal['returncode']==0 and terminal['hard_timeout'] is False
 assert process['pid']==run['pid'] and process['hard_seconds']==p['hard_seconds']==5400
 assert launch['supervisor_pid']==runtime['pid'] and launch['protocol']==runtime['protocol']
 assert launch['preflight']['no_compute_process'] and launch['preflight']['memory_mib']<=64
 publication=read(control/'publication_verified.json');assert sha(control/'publication_verified.json')==launch['publication']['sha256']
 assert publication['commit']==publication['remote_main']==runtime['publication_commit']==run['git_commit']
 assert publication['verified_unix']<launch['started_unix']<=process['started_unix']<run['started_unix']
 assert run['protocol_sha256']==runtime['protocol']['sha256'] and run['model_revision']==p['model_revision']
 assert run['target_evaluation'] is False and run['reserved21_accessed'] is False
 cost=audit_sampling(out);assert cost['new_model_tokens']==summary['new_sampled_tokens']==5200<=p['new_model_token_ceiling']
 assert cost['rollouts']==4 and cost['input_token_positions_excluding_padding']==summary['input_token_positions_excluding_padding']
 assert cost['external_tokens']==summary['external_tokens']
 allocations=lines(out/'allocation.jsonl');groups=lines(out/'groups.jsonl');updates=lines(out/'updates.jsonl');micro=lines(out/'microsteps.jsonl')
 assert len(allocations)==len(groups)==len(updates)==summary['optimizer_steps']==1 and len(micro)==summary['microsteps']==4
 assert [x['micro_step_after'] for x in micro]==[1,2,3,4]
 assert all(x['accelerate_backward_divisor']==1 and x['grpo_loss_divisor']==4 for x in micro)
 assert allocations[0]['task_id']==p['task_selection']['selected_task_id'] and allocations[0]['seed']==p['rollout_seed']
 assert allocations[0]['rewards']==groups[0]['rewards']==summary['rewards']==[1,1,1,1]
 assert groups[0]['advantages']==[0,0,0,0] and groups[0]['model_tokens_in_loss']==5200
 initial=read(out/'initial_learning_state.json')
 assert initial['optimizer_state_entries']==0 and initial['history_acquisition_tokens']==0
 assert updates[0]['before_sha256']==updates[0]['after_sha256']==initial['policy_fingerprint']==summary['final_fingerprint']
 assert updates[0]['parameters_changed'] is False
 grad=[x for x in lines(out/'metrics.jsonl') if 'grad_norm' in x];assert len(grad)==1 and grad[0]['grad_norm']==0
 # Replay saved model tokens through the native tokenizer and real simulator;
 # this performs no new generation and rechecks rewards, tool results and masks.
 source_path=root/p['algorithm_protocol']['path'];assert sha(source_path)==p['algorithm_protocol']['sha256'];source=read(source_path)
 tokenizer=r.AutoTokenizer.from_pretrained(Path(run['native_model_path']),local_files_only=True)
 selected=p['task_selection']['selected_task_id'];pair=r.load_tasks(source,[selected],'train')[selected]
 rollouts=lines(out/'rollouts.jsonl');assert [x['rollout_index'] for x in rollouts]==[0,1,2,3]
 for row in rollouts:
  replay=r.episode(tokenizer,pair,source)
  for segment in row['segments']:
   assert replay.input_ids==segment['input_ids']
   replay.accept(segment['generated_ids'])
  for key,value in replay.evidence().items():assert value==row[key],(row['rollout_index'],key)
 assert summary['target_evaluation_performed'] is False and summary['reserved21_accessed'] is False
 audit={'status':'passed','protocol':runtime['protocol'],'original_process_pid':process['pid'],'original_returncode':0,
  'cost':cost,'native_token_and_simulator_rollout_replay':4,'semantic_rewards_replayed':[1,1,1,1],
  'optimizer_steps':1,'fresh_nonzero_gradient_steps':0,'changed_parameter_steps':0,
  'original_generation_memory':summary['generation_memory'],'original_scoring_backward_optimizer_memory':summary['scoring_backward_optimizer_memory'],
  'worker_wall_seconds':terminal['wall_seconds'],'runner_wall_seconds':summary['wall_seconds'],
  'generation_seconds':summary['generation_seconds'],'non_generation_training_seconds':summary['non_generation_training_seconds'],
  'observed_max_prompt_plus_completion_tokens':summary['maximum_prompt_plus_completion_length'],
  'scope':'One model-specific capacity measurement, not transfer efficacy; observed lengths only, no full8192-completion stress guarantee.',
  'new_tokens_from_audit':0,'automatic_retry':False,'reserved21_accessed':False,'goal_complete':False}
 new(out/'audit.json',audit)
 files={str(f.relative_to(root)):{'sha256':sha(f),'bytes':f.stat().st_size} for folder in [out,control] for f in sorted(folder.rglob('*')) if f.is_file()}
 proof=out/'evidence_manifest.json';new(proof,{'protocol':runtime['protocol'],'audit':ref(out/'audit.json'),'files':files,'original_attempts':1,'new_model_inference_from_review':0})
 idx=read(root/'experiments/index.json');assert idx[-1]['id']=='E360';eid='E361';now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 facts={'audit':ref(out/'audit.json'),'evidence_manifest':ref(proof),'summary':ref(out/'summary.json'),
  'cost':cost,'original_single_attempt_completed':True,'resource_feasible_at_observed_length':True,
  'fresh_nonzero_gradient_steps':0,'changed_parameter_steps':0,'future_scientific_matrix_started':False,'goal_complete':False}
 conclusion='Qwen3-8B原生thinking/LoRA/校正GRPO的单次长prompt实测完成并通过4轨迹完整token/mask/工具/奖励重放：实际5200新模型token、136538输入位置、968外部token；单次1更新/4microsteps，worker287.578秒。生成峰值allocated22173.49MiB，后续scoring/backward/optimizer峰值reserved33908MiB，所测最长prompt+completion6703token，单张40GiB A100可运行。四条奖励全1、优势/新梯度均0、权重指纹不变，不能称学到了能力；不追加轨迹追非零梯度，也不保证所有未来最大长度均可容纳。'
 next_action='用此真实容量/成本登记有限的一阶段8B规模验证：使用8B自己的共同起点与采样前信号、冻结4B函数跨模型预测并在目标前封存；完整拟合/留出或固定子集的职责、总预算与停止条件须在新科学协议中明确。保留非Qwen、机制广度、ID和环境外推，原21题仍关闭。'
 idx.append({'id':eid,'source':str((out/'audit.json').relative_to(root)),'kind':'completed single nativeQwen3-8B cost and capacity measurement','sha256':hashlib.sha256(json.dumps(facts,sort_keys=True).encode()).hexdigest(),'facts':facts,'batch':p['output'],'observed_at':now})
 new(root/'experiments/reviews/E361.json',{'status':'reviewed','reviewed_at':now,'reviewer':'Current SSH execution agent; not independent human review','evidence':[str(proof.relative_to(root)),str((out/'audit.json').relative_to(root))],'conclusion':conclusion,'next_action':next_action,'new_sampled_tokens':5200,'cost_accounting':'One original engineering acquisition charged once; CPU audit adds zero generation; future8B science does not recharge this cost.','goal_complete':False})
 with (root/'experiments/E361.md').open('x') as f:f.write('# E361: Qwen3-8B单次实际更新、容量与成本\n\n'+now+'\n\n'+conclusion+'\n\n'+next_action+'\n\n```json\n'+json.dumps(facts,ensure_ascii=False,indent=2)+'\n```\n')
 (root/'experiments/index.json').write_text(json.dumps(idx,ensure_ascii=False,indent=2)+'\n');record(root,[])
 for name in ['experiments/NEXT.md','experiments/SCOPE_REVIEW.md']:
  with (root/name).open('a') as f:f.write('\n### '+now+' E361：一次8B实测完成\n\n'+conclusion+' '+next_action+' 本批明确为工程成本证据，不代替新的经验价值干预或G/V。\n')
 n=root/'experiments/NEXT.md';text=n.read_text();text,count=re.subn(r'(## 当前状态\n\n)[^\n]+',lambda m:m.group(1)+now+'：4B Stage0 51/104与Stage1 15/32报告均完成并发布（E276/E359）；8B唯一一次容量/成本测量已完成并审计（E361），原科学矩阵尚未启动，当前无项目活跃GPU子进程。下一步登记一阶段8B有限规模验证；21题关闭，晚全矩阵延后，Goal未完成。',text,count=1);assert count==1;n.write_text(text)
 paths=['pipeline/review_qwen8_capacity_check.py',*files,str(proof.relative_to(root)),'experiments/E361.md','experiments/reviews/E361.json','experiments/index.json','experiments/README.md','experiments/events.jsonl','experiments/NEXT.md','experiments/SCOPE_REVIEW.md']
 for name in paths:
  f=root/name;assert f.suffix not in ['.pt','.bin','.safetensors'] and not f.is_symlink()
  assert not re.search(rb'gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|sk-[A-Za-z0-9_-]{30,}',f.read_bytes()),name
 assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=root,text=True).strip()
 subprocess.run(['git','add','--',*paths],cwd=root,check=True)
 subprocess.run(['git','diff','--cached','--check'],cwd=root,check=True)
 subprocess.run(['git','commit','-m','Publish single Qwen3-8B capacity measurement with semantic replay and zero-gradient limitation'],cwd=root,check=True,stdout=subprocess.DEVNULL)
 print(json.dumps({'event':eid,'audit':audit,'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'goal_complete':False}),flush=True)

if __name__=='__main__':main()
