"""Same-input greedy runtime parity diagnostic; run separately in each venv."""
import json,sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
root=Path(__file__).resolve().parents[1]
r=json.loads((root/'rl/smoke_002/generation_calls.jsonl').read_text().splitlines()[0])
t=AutoTokenizer.from_pretrained(root/'smoke/model',local_files_only=True)
torch.set_num_threads(4)
m=AutoModelForCausalLM.from_pretrained(root/'smoke/model',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda').eval()
x=torch.tensor([r['prompt_ids'][0]],device='cuda')
with torch.no_grad():
 o=m.generate(x,attention_mask=torch.ones_like(x),max_new_tokens=160,do_sample=False,temperature=None,top_p=None,top_k=None,return_dict_in_generate=True,output_scores=True)
p=o.scores[0].float().softmax(-1)
ent=-(p*p.clamp_min(1e-20).log()).sum().item()
result={'runtime':sys.argv[1],'first_token_entropy':ent,'output':t.decode(o.sequences[0,x.shape[1]:]),'token_ids':o.sequences[0,x.shape[1]:].tolist()}
(root/f'rl/runtime_compare_{sys.argv[1]}.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['output'])
