"""Regression: real tokenizer must left-pad unequal chat prompts before generate."""
import unittest
from pathlib import Path
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


class PaddingTest(unittest.TestCase):
    def test_batched_generation_has_no_trailing_padding(self):
        tokenizer = AutoTokenizer.from_pretrained(ROOT/'smoke/model', local_files_only=True)
        prompts = [[{'role': 'user', 'content': 'Short'}],
                   [{'role': 'user', 'content': 'A substantially longer request containing many different words.'}]]
        tokenizer.padding_side = 'left'
        batch = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=True,
                                              padding=True, return_tensors='pt', return_dict=True,
                                              enable_thinking=False)
        self.assertTrue(batch['attention_mask'][:, -1].all())
        self.assertEqual(int(batch['attention_mask'][0, 0]), 0)
        for i, prompt in enumerate(prompts):
            single = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True,
                                                   return_dict=False, enable_thinking=False)
            self.assertEqual(batch['input_ids'][i][batch['attention_mask'][i].bool()].tolist(), single)


if __name__ == '__main__':
    unittest.main()
