"""Checks for the official-path BFCL runner (pipeline/eval_bfcl.py).

These run without a GPU: they exercise prompt construction and the official
parse/decode/check pipeline against BFCL's own possible_answer files.
"""
import unittest

from eval_bfcl import (MODEL_CONFIG_SOURCE, build_prompt, load_category, score)


def ideal_output(ground_truth):
    """Render a ground-truth answer in the official python return format.

    possible_answer maps each parameter to the LIST of acceptable values, so one
    acceptable value is chosen per parameter. An empty string in that list marks
    an optional parameter; parameters with no other acceptable value are omitted.
    """
    entry = ground_truth[0]
    name = list(entry)[0]
    args = []
    for param, acceptable in entry[name].items():
        value = pick(acceptable)
        if value == '':
            continue
        args.append(f'{param}={value!r}')
    return f'[{name}({", ".join(args)})]'


def pick(acceptable):
    if isinstance(acceptable, list):
        for value in acceptable:
            if value != '':
                return value
        return ''
    return acceptable


class RunnerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items, cls.answers = load_category('simple_python')

    def test_model_config_available(self):
        self.assertIn(MODEL_CONFIG_SOURCE, ('real', 'shim'))

    def test_prompt_uses_official_template(self):
        item = self.items[0]
        prompt = build_prompt(item, 'simple_python')
        self.assertTrue(prompt.startswith('<|im_start|>system\n'))
        self.assertIn('You are an expert in composing functions.', prompt)
        self.assertIn(item['function'][0]['name'], prompt)
        self.assertTrue(prompt.endswith('<|im_start|>assistant\n'))

    def test_ground_truth_output_scores_valid(self):
        for item in self.items[:20]:
            item['_answers'] = self.answers
            decodable, valid, reason = score(item, 'simple_python',
                                             ideal_output(self.answers[item['id']]), 1, 1)
            self.assertTrue(decodable, f'{item["id"]}: {reason}')
            self.assertTrue(valid, f'{item["id"]}: {reason}')

    def test_wrong_function_name_is_invalid(self):
        item = self.items[0]
        item['_answers'] = self.answers
        decodable, valid, reason = score(item, 'simple_python', '[not_a_function(x=1)]', 1, 1)
        self.assertTrue(decodable)
        self.assertFalse(valid)
        self.assertIn('wrong_func_name', reason)

    def test_plain_text_is_not_decodable(self):
        item = self.items[0]
        item['_answers'] = self.answers
        decodable, valid, _ = score(item, 'simple_python', 'I cannot answer this question.', 1, 1)
        self.assertFalse(decodable)
        self.assertFalse(valid)

    def test_think_block_is_stripped_before_decoding(self):
        item = self.items[0]
        item['_answers'] = self.answers
        text = '<think>\nsome reasoning\n</think>\n' + ideal_output(self.answers[item['id']])
        decodable, valid, reason = score(item, 'simple_python', text, 1, 1)
        self.assertTrue(valid, reason)

    def test_irrelevance_passes_on_refusal_and_fails_on_call(self):
        items, _ = load_category('irrelevance')
        item = items[0]
        item['_answers'] = {}
        _, valid_refusal, _ = score(item, 'irrelevance',
                                    '<think>x</think>\nNo suitable function is available.', 1, 1)
        self.assertTrue(valid_refusal, 'a refusal must pass the irrelevance category')
        _, valid_call, reason = score(item, 'irrelevance', '[math.sum(numbers=[1,2])]', 1, 1)
        self.assertFalse(valid_call, 'a function call must fail the irrelevance category')
        self.assertIn('made_call', reason)


if __name__ == '__main__':
    unittest.main()
