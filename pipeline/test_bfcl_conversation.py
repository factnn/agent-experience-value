"""All development oracle conversations and user-boundary failure controls."""
import json
from pathlib import Path
import unittest
from bfcl_conversation import BFCLConversation
from bfcl_safe_runtime import ROOT,PKG


class ConversationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest=json.loads((ROOT/'rl/bfcl_research_split_candidate/manifest.json').read_text())
        cls.ids={r['id'] for r in manifest['development']}
        cls.tasks={r['id']:r for r in map(json.loads,(PKG/'bfcl_eval/data/BFCL_v4_multi_turn_base.json').read_text().splitlines()) if r['id'] in cls.ids}
        cls.answers={r['id']:r['ground_truth'] for r in map(json.loads,(PKG/'bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_base.json').read_text().splitlines()) if r['id'] in cls.ids}

    def oracle_turn(self,conversation,calls):
        for call in calls:
            conversation.assistant_message({'role':'assistant','content':call})
            conversation.tool_call(call)
        return conversation.finish_user_turn({'role':'assistant','content':'Completed.'})

    def test_all_22_full_conversations_preserve_user_messages(self):
        for key in sorted(self.ids):
            with self.subTest(task=key):
                task=self.tasks[key];conversation=BFCLConversation(task,self.answers[key])
                self.assertEqual(conversation.messages(),task['question'][0])
                self.assertEqual(conversation.reward(),0)
                for i,calls in enumerate(self.answers[key]):
                    following=self.oracle_turn(conversation,calls)
                    self.assertEqual(following,task['question'][i+1] if i+1<len(task['question']) else [])
                    if not conversation.completed:self.assertEqual(conversation.reward(),0)
                self.assertEqual(conversation.reward(),1)
                expected=[m for turn in task['question'] for m in turn]
                actual=[m for m in conversation.messages() if m['role']=='user']
                self.assertEqual(actual,expected)
                with self.assertRaises(RuntimeError):conversation.tool_call('ls()')

    def test_failed_earlier_turn_is_not_erased(self):
        key='multi_turn_base_116';c=BFCLConversation(self.tasks[key],self.answers[key])
        next_messages=c.finish_user_turn({'role':'assistant','content':'Done without actions.'})
        self.assertEqual(next_messages,self.tasks[key]['question'][1])
        self.assertFalse(c.grades[0]['valid'])
        # Apply missing earlier actions as a late repair; current state may recover,
        # but the previous user-boundary failure remains part of terminal reward.
        for call in self.answers[key][0]:c.tool_call(call)
        for calls in self.answers[key][1:]:self.oracle_turn(c,calls)
        self.assertTrue(c.completed);self.assertEqual(c.reward(),0)

    def test_partial_and_tool_failure_do_not_reveal_grades(self):
        key=sorted(self.ids)[0];c=BFCLConversation(self.tasks[key],self.answers[key])
        response=c.tool_call('__import__("os").getcwd()')
        self.assertTrue(response.startswith('Error during execution:'))
        self.assertEqual(c.turn,0);self.assertFalse(c.grades);self.assertEqual(c.reward(),0)
        visible=c.messages()
        self.assertFalse(any('check' in m or 'valid' in m for m in visible))
        visible.clear();self.assertTrue(c.messages())

    def test_failed_boundary_diagnostics_do_not_mutate_later(self):
        key='multi_turn_base_26';c=BFCLConversation(self.tasks[key],self.answers[key])
        c.finish_user_turn({'role':'assistant','content':'Done without actions.'})
        before=repr(c.grades[0])
        c.tool_call("touch(file_name='late_change.txt')")
        self.assertEqual(repr(c.grades[0]),before)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ConversationTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'scope':'CPU controller acceptance; not policy rollouts or full TRL integration',
        'tests':result.testsRun,'passed':result.wasSuccessful(),'development_tasks':22,
        'verified':['current-turn-only initial observation','exact user-role message advancement',
            'all development oracle terminal rewards','per-user-turn failure latch',
            'incomplete episode fails','tool errors do not advance user turn','no oracle grade in observations'],
        'remaining':['token lineage/masks across user/tool messages','Qwen3 generation + official GRPO integration']}
    (ROOT/'rl/bfcl_research_split_candidate/conversation_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
