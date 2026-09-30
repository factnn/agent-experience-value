"""Token lengths only: CPU tokenizer, no model weights or GPU allocation."""
import json,pathlib,os
os.environ['TOKENIZERS_PARALLELISM']='false'
from transformers import AutoTokenizer
ROOT=pathlib.Path(__file__).resolve().parents[1]
REV=json.loads((ROOT/'audit/tokenizer_model_metadata.json').read_text())['sha']
tok=AutoTokenizer.from_pretrained('Qwen/Qwen3-4B',revision=REV,trust_remote_code=False)
rows={}
for name in ['annotation_random200','annotation_error100']:
 for line in (ROOT/'prepared'/f'{name}.jsonl').open():
  x=json.loads(line);rows[x['features']['row_id']]=x
counts=[]
for idx,x in rows.items():
 msgs=x['conversations']
 inference_total=len(tok.apply_chat_template(msgs,tokenize=True,add_generation_prompt=False))
 rendered=''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in msgs)
 total=len(tok.encode(rendered,add_special_tokens=False))
 assistant=sum(len(tok.encode(m['content'],add_special_tokens=False)) for m in msgs if m['role']=='assistant')
 counts.append({'row_id':idx,'chat_tokens':total,'official_template_tokens':inference_total,'assistant_content_tokens_diagnostic':assistant})
(ROOT/'prepared/token_lengths.json').write_text(json.dumps({'model':'Qwen/Qwen3-4B','revision':REV,'note':'CPU tokenizer only. Assistant content counts exclude chat framing and are NOT an implemented training loss mask. chat_tokens uses explicit ChatML preserving ALL original content including historic thoughts. official_template_tokens uses upstream template which strips earlier assistant thoughts. Neither is a tested training recipe.','rows':counts},indent=2))
random_ids={json.loads(l)['features']['row_id'] for l in (ROOT/'prepared/annotation_random200.jsonl').open()}
a=sorted(r['chat_tokens'] for r in counts if r['row_id'] in random_ids)
summary={'random_n':len(a),'min':a[0],'median':a[len(a)//2],'p90':a[int(len(a)*.9)],'max':a[-1],'over_8192':sum(x>8192 for x in a),'over_16384':sum(x>16384 for x in a),'over_32768':sum(x>32768 for x in a),'over_65536':sum(x>65536 for x in a)}
(ROOT/'prepared/token_summary.json').write_text(json.dumps(summary,indent=2));print(summary)
