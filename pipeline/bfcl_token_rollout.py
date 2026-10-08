"""Append-only Qwen conversation tokens; no retokenization of policy output."""
import copy
import json
import re
from bfcl_safe_runtime import ROOT,PKG,MULTI_TURN_FUNC_DOC_FILE_MAPPING
from bfcl_conversation import BFCLConversation


def development_tasks(ids):
    manifest=json.loads((ROOT/'rl/bfcl_research_split_candidate/manifest.json').read_text())
    assert set(ids)<={r['id'] for r in manifest['development']}
    tasks={r['id']:r for r in map(json.loads,(PKG/'bfcl_eval/data/BFCL_v4_multi_turn_base.json').read_text().splitlines()) if r['id'] in ids}
    answers={r['id']:r['ground_truth'] for r in map(json.loads,(PKG/'bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json').read_text().splitlines()) if r['id'] in ids}
    return tasks,answers


def normalize_schema(value):
    if isinstance(value,list):return [normalize_schema(x) for x in value]
    if not isinstance(value,dict):return value
    result={k:normalize_schema(v) for k,v in value.items()}
    if 'type' in result:
        result['type']={'dict':'object','float':'number','int':'integer','bool':'boolean','str':'string','list':'array'}.get(result['type'],result['type'])
    return result


def initial_prompt(task):
    schemas=[]
    for name in task['involved_classes']:
        path=PKG/'bfcl_eval/data/multi_turn_func_doc'/MULTI_TURN_FUNC_DOC_FILE_MAPPING[name]
        for line in path.read_text().splitlines():
            doc=json.loads(line)
            if doc['name'] in task.get('excluded_function',[]):continue
            function=normalize_schema({k:doc[k] for k in ['name','description','parameters']})
            function['name']=name+'.'+function['name']
            schemas.append({'type':'function','function':function})
    system=('Complete the current user request using the documented simulated tools. '
        'Tool calls change a persistent workspace. Do not guess tool results. '
        'After completing this request give a brief answer; another user request may follow.\n'
        '# Tools\n<tools>\n'+'\n'.join(json.dumps(x,ensure_ascii=False) for x in schemas)+
        '\n</tools>\nFor each function call, return a JSON object within tags:\n'
        '<tool_call>{"name": "Class.method", "arguments": {"parameter": "value"}}</tool_call>')
    return [{'role':'system','content':system}]+copy.deepcopy(task['question'][0])


def parse_calls(text):
    # Reasoning may quote examples; only the post-thinking answer is actionable.
    text=text.rsplit('</think>',1)[-1]
    chunks=re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S)
    if text.count('<tool_call>')!=len(chunks) or text.count('</tool_call>')!=len(chunks):
        raise ValueError('Malformed tool_call tags')
    calls=[]
    for chunk in chunks:
        item=json.loads(chunk);name=item['name'];arguments=item['arguments']
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z]\w*(?:\.[A-Za-z]\w*)?',name):
            raise ValueError('Invalid tool name')
        if not isinstance(arguments,dict) or any(not isinstance(k,str) or not k.isidentifier() for k in arguments):
            raise ValueError('Arguments must be a JSON object with parameter names')
        calls.append((name,name+'('+','.join(k+'='+repr(v) for k,v in arguments.items())+')'))
    return calls


class TokenEpisode:
    def __init__(self,tokenizer,task,reference,budget=4096,max_calls=16,max_segments=16):
        self.tokenizer=tokenizer;self.conversation=BFCLConversation(task,reference)
        self.prompt=initial_prompt(task)
        self.prompt_ids=tokenizer.apply_chat_template(self.prompt,tokenize=True,
            add_generation_prompt=True,enable_thinking=False,return_dict=False)
        self.completion_ids=[];self.mask=[];self.segments=[];self.bridges=[]
        self.budget=budget;self.max_calls=max_calls;self.max_segments=max_segments
        self.stop_reason=None

    @property
    def active(self):return self.stop_reason is None

    @property
    def input_ids(self):return self.prompt_ids+self.completion_ids

    def bridge(self,messages):
        # Template ONLY new external messages. Old generated assistant bytes
        # (including embedded EOS and reasoning) remain exactly untouched.
        text='\n'+self.tokenizer.apply_chat_template(messages,tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        ids=self.tokenizer.encode(text,add_special_tokens=False)
        if len(self.completion_ids)+len(ids)>=self.budget:
            self.stop_reason='external_bridge_budget';return
        start=len(self.completion_ids);self.completion_ids.extend(ids);self.mask.extend([0]*len(ids))
        self.bridges.append({'start':start,'ids':ids,'messages':copy.deepcopy(messages)})

    def accept(self,ids):
        assert self.active and ids and len(ids)<=self.budget-len(self.completion_ids)
        prefix=self.input_ids.copy();start=len(self.completion_ids)
        self.completion_ids.extend(ids);self.mask.extend([1]*len(ids))
        self.segments.append({'start':start,'input_ids':prefix,'generated_ids':ids.copy()})
        if ids[-1]!=self.tokenizer.eos_token_id:
            self.stop_reason='generation_cap';return
        text=self.tokenizer.decode(ids,skip_special_tokens=True)
        try:calls=parse_calls(text)
        except (ValueError,KeyError,TypeError) as error:
            self.conversation.assistant_message({'role':'assistant','content':text})
            self.bridge([{'role':'tool','content':f'Tool format error: {error}'}]);calls=None
        if calls:
            self.conversation.assistant_message({'role':'assistant','content':text})
            replies=[]
            for name,call in calls:
                if len(self.conversation.events)>=self.max_calls:
                    self.stop_reason='tool_call_limit';break
                response=self.conversation.tool_call(call,name)
                replies.append({'role':'tool','name':name,'content':response})
            if self.active:self.bridge(replies)
        elif calls is not None:
            following=self.conversation.finish_user_turn({'role':'assistant','content':text})
            if following:self.bridge(following)
            else:self.stop_reason='completed'
        if self.active and len(self.segments)>=self.max_segments:self.stop_reason='segment_limit'

    def evidence(self):
        c=self.conversation
        return {'task_id':c.task['id'],'prompt_ids':self.prompt_ids,'completion_ids':self.completion_ids,
            'model_token_mask':self.mask,'segments':self.segments,'bridges':self.bridges,
            'events':c.events,'grades':json.loads(json.dumps(c.grades,default=lambda value:{
                'runtime_type':type(value).__qualname__,'representation':str(value)})),
            'completed_user_turns':c.turn,
            'total_user_turns':len(c.task['question']),'reward':c.reward(),'stop_reason':self.stop_reason}
