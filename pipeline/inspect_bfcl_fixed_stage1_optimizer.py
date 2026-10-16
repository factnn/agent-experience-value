from pathlib import Path
import hashlib,json,torch,datetime
r=Path('/share/project/zpy/RL/agent_experience_project_20260928');master=json.loads((r/'rl/BFCL_SOURCE_STAGE1_FORWARD15_PROTOCOL_20261015.json').read_text());path=r/master['stage_start']['path']/'components/learning_state.pt'
h=hashlib.sha256()
with path.open('rb') as f:
 for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
assert h.hexdigest()==master['stage_start']['payload_sha256']
p=torch.load(path,map_location='cpu',weights_only=False);opt=p['optimizer'];states=list(opt['state'].values());groups=[]
for g in opt['param_groups']:groups.append({k:g[k] for k in ['lr','weight_decay','betas','eps','foreach','fused','maximize'] if k in g})
slots={}
for key in ['exp_avg','exp_avg_sq']:
 tensors=[s[key] for s in states if key in s];counts=[int(torch.count_nonzero(t)) for t in tensors]
 slots[key]={'tensor_count':len(tensors),'nonzero_tensor_count':sum(c>0 for c in counts),'nonzero_elements':sum(counts),'elements':sum(t.numel() for t in tensors)}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'kind':'read_only_prespecified_parent_optimizer_metadata','source_payload':{'path':str(path.relative_to(r)),'sha256':h.hexdigest()},'progress':p['progress'],'optimizer_type':p['optimizer_type'],'optimizer_step_counters':sorted({int(s['step']) for s in states}),'param_group_settings':groups,'moment_slots':slots,'new_scientific_tokens':0,'new_gpu_jobs':0,'scope':'Existing fixed parent state only. Metadata interpretation,not an independent gradient replay or normalization/dose/recovery calibration. Does not establish late transferG/V.'},indent=2))
