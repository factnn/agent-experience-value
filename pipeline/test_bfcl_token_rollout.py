"""Real tokenizer tests for original-token preservation and user/tool masks."""
import ast
import inspect
import json
import unittest
from transformers import AutoTokenizer
from bfcl_safe_runtime import ROOT,PKG
from bfcl_token_rollout import TokenEpisode,parse_calls,development_tasks,generation_buckets


class TokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=AutoTokenizer.from_pretrained(ROOT/'smoke/model',local_files_only=True)
        cls.tasks,cls.answers=development_tasks(['multi_turn_base_26','multi_turn_base_70'])

    def tokens(self,text):return self.tokenizer.encode(text,add_special_tokens=False)+[self.tokenizer.eos_token_id]

    def test_oracle_tokens_and_roles(self):
        for key in self.tasks:
            e=TokenEpisode(self.tokenizer,self.tasks[key],self.answers[key],budget=20000,max_calls=64,max_segments=64)
            self.assertNotIn(self.tasks[key]['question'][1][0]['content'],self.tokenizer.decode(e.prompt_ids))
            for calls in self.answers[key]:
                for call in calls:
                    tree=ast.parse(call,mode='eval').body
                    name=tree.func.id if isinstance(tree.func,ast.Name) else tree.func.attr
                    matches=[k for k in e.conversation.actual.methods if k[1]==name]
                    self.assertEqual(len(matches),1)
                    method=e.conversation.actual.methods[matches[0]]
                    bound=inspect.signature(method).bind(*[ast.literal_eval(a) for a in tree.args],**{k.arg:ast.literal_eval(k.value) for k in tree.keywords})
                    e.accept(self.tokens('<tool_call>'+json.dumps({'name':'.'.join(matches[0]),'arguments':dict(bound.arguments)})+'</tool_call>'))
                e.accept(self.tokens('Completed.'))
            self.assertEqual(e.stop_reason,'completed');self.assertEqual(e.conversation.reward(),1)
            full=e.input_ids
            for s in e.segments:
                start=len(e.prompt_ids)+s['start']
                self.assertEqual(full[:start],s['input_ids'])
                self.assertEqual(full[start:start+len(s['generated_ids'])],s['generated_ids'])
                self.assertTrue(all(e.mask[s['start']:s['start']+len(s['generated_ids'])]))
            for b in e.bridges:self.assertFalse(any(e.mask[b['start']:b['start']+len(b['ids'])]))
            users=[m for b in e.bridges for m in b['messages'] if m['role']=='user']
            self.assertEqual(users,[m for turn in self.tasks[key]['question'][1:] for m in turn])

    def test_cap_and_malformed_call_do_not_advance(self):
        key='multi_turn_base_26';e=TokenEpisode(self.tokenizer,self.tasks[key],self.answers[key])
        e.accept(self.tokens('<tool_call>broken</tool_call>'))
        self.assertEqual(e.conversation.turn,0);self.assertTrue(e.active)
        e.accept(self.tokenizer.encode('unfinished thinking',add_special_tokens=False))
        self.assertEqual(e.stop_reason,'generation_cap');self.assertEqual(e.conversation.reward(),0)

    def test_reasoning_is_not_executed_and_calls_are_literal(self):
        self.assertEqual(parse_calls('<think><tool_call>broken</tool_call></think>Answer.'),[])
        with self.assertRaises(ValueError):parse_calls('<tool_call>{"name":"__import__","arguments":{}}</tool_call>')
        with self.assertRaises(ValueError):parse_calls('<tool_call>unfinished')

    def test_failed_state_diagnostics_are_json_serializable(self):
        key='multi_turn_base_26';e=TokenEpisode(self.tokenizer,self.tasks[key],self.answers[key])
        e.accept(self.tokens('Done without actions.'))
        self.assertFalse(e.conversation.grades[0]['valid'])
        evidence=json.loads(json.dumps(e.evidence()))
        self.assertFalse(evidence['grades'][0]['valid'])

    def test_batch_peer_cannot_shorten_remaining_allowance(self):
        key='multi_turn_base_26'
        a=TokenEpisode(self.tokenizer,self.tasks[key],self.answers[key])
        b=TokenEpisode(self.tokenizer,self.tasks[key],self.answers[key])
        # A has 64 left; B has 3000 left. Only model-independent budget
        # bookkeeping is synthetic here; actual accept/bridge/tokenizer run.
        for e,used in [(a,4032),(b,1096)]:
            e.completion_ids=[self.tokenizer.pad_token_id]*used;e.mask=[1]*used
        buckets=generation_buckets([a,b])
        self.assertEqual([(cap,[id(e) for e in es]) for cap,es in buckets],[(64,[id(a)]),(1024,[id(b)])])
        a.accept([1]*64);self.assertEqual(a.stop_reason,'generation_cap')
        answer=self.tokens('word '*90)
        self.assertGreater(len(answer),64);self.assertLess(len(answer),1024)
        b.accept(answer)
        self.assertTrue(b.active);self.assertEqual(b.conversation.turn,1)

    def test_thinking_mode_reaches_initial_and_all_external_headers(self):
        key='multi_turn_base_26'
        for thinking in [False,True]:
            e=TokenEpisode(self.tokenizer,self.tasks[key],self.answers[key],enable_thinking=thinking)
            suffix='<|im_start|>assistant\n'+('' if thinking else '<think>\n\n</think>\n\n')
            self.assertTrue(self.tokenizer.decode(e.prompt_ids).endswith(suffix))
            for message in [{'role':'tool','content':'Tool response.'},{'role':'user','content':'Next request.'}]:
                e.bridge([message])
                self.assertTrue(self.tokenizer.decode(e.bridges[-1]['ids']).endswith(suffix))
            self.assertEqual(e.evidence()['enable_thinking'],thinking)


if __name__=='__main__':unittest.main(verbosity=2)
