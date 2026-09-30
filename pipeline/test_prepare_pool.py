import unittest
from prepare_pool import parse_action,operation,feature,extract_instruction
class ParserChecks(unittest.TestCase):
 def test_thought_example_is_not_action(self):
  a=parse_action('<think>{"commands": []}</think>{"commands":[{"keystrokes":"pytest -q\\n"}]}')
  self.assertEqual(a['commands'][0]['keystrokes'],'pytest -q\n')
 def test_malformed_command_rejected(self):
  self.assertIsNone(parse_action('{"commands":[{"keystrokes":4}]}'))
  self.assertIsNone(parse_action('not json'))
 def test_shell_wrapper_and_redirection(self):
  self.assertEqual(operation('bash -lc "pip install pytest"'),'install')
  self.assertEqual(operation('cat > file.py <<EOF\nx\nEOF'),'edit')
  self.assertEqual(operation('python -m pytest -q'),'test')
 def test_completion_and_runtime_exception_are_not_success(self):
  for result in [None,'AgentTimeoutError']:
   f=feature({'result':result,'conversations':[{'role':'assistant','content':'{"commands":[],"task_complete":true}'}]},0,'x')
   self.assertEqual(f['terminal_outcome'],'unknown')
   self.assertTrue(f['self_reported_complete'])
 def test_instruction_drops_terminal_identity(self):
  a=extract_instruction([{'role':'user','content':'header\nTask Description:\nFix x\nCurrent terminal state:\nroot@a'}])
  b=extract_instruction([{'role':'user','content':'header\nTask Description:\nFix x\nCurrent terminal state:\nroot@b'}])
  self.assertEqual(a,b)
if __name__=='__main__':unittest.main()
