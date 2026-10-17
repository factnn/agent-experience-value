"""Synthetic CPU checks for token-lineage-preserving feedback ablation."""
import copy
import json

from bfcl_model_conditioned_feature_inputs import content_token_indices, feedback_inputs, saved_model_inputs


class TinyTokenizer:
    all_special_ids = [10001]

    def apply_chat_template(self, messages, **kwargs):
        return ''.join('<' + x['role'] + '>' + x['content'] + '</' + x['role'] + '>' for x in messages) + '<assistant>'

    def __call__(self, text, **kwargs):
        ids = [13 if ch == '.' else 10001 if ch == '§' else ord(ch) for ch in text]
        return {'input_ids': ids, 'offset_mapping': [(i, i + 1) for i in range(len(text))]}


def episode(messages, tail=False):
    tokenizer = TinyTokenizer()
    bridge_ids = tokenizer('\n' + tokenizer.apply_chat_template(messages))['input_ids']
    prompt, first, last = [500, 501], [700, 701], [702, 703]
    completion = first + bridge_ids + last
    result = {'prompt_ids': prompt, 'completion_ids': completion,
              'model_token_mask': [1, 1] + [0] * len(bridge_ids) + [1, 1], 'enable_thinking': True,
              'segments': [{'start': 0, 'input_ids': prompt, 'generated_ids': first},
                           {'start': len(first) + len(bridge_ids),
                            'input_ids': prompt + first + bridge_ids, 'generated_ids': last}],
              'bridges': [{'start': len(first), 'ids': bridge_ids, 'messages': messages}]}
    if tail:
        text = [{'role': 'tool', 'content': 'unscored_tail'}]
        ids = tokenizer('\n' + tokenizer.apply_chat_template(text))['input_ids']
        result['bridges'].append({'start': len(completion), 'ids': ids, 'messages': text})
        result['completion_ids'] = completion + ids
        result['model_token_mask'] += [0] * len(ids)
    return result


def main():
    messages = [{'role': 'tool', 'content': 'same甲§.'},
                {'role': 'user', 'content': 'same甲§.'},
                {'role': 'tool', 'content': 'same甲§.'}]
    row = episode(messages, tail=True)
    before = copy.deepcopy(row)
    result = feedback_inputs(row, TinyTokenizer())
    assert row == before
    assert len(result['input_ids']) == len(result['ablated_input_ids'])
    assert result['feedback_sensitive_target_indices'] == [2, 3]
    assert result['mask_audit']['changed_tool_content_tokens_in_scored_prefix'] == 10
    assert result['unscored_external_tail_tokens'] > 0
    assert result['input_ids'].count(10001) == result['ablated_input_ids'].count(10001)
    assert all(result['input_ids'][i] == result['ablated_input_ids'][i] for i in result['target_positions'])
    assert ''.join(chr(x) for x in result['ablated_input_ids'] if x < 256).count('<user>same') == 1
    empty = feedback_inputs(episode([{'role': 'user', 'content': 'same'}]), TinyTokenizer())
    assert empty['input_ids'] == empty['ablated_input_ids']
    assert empty['feedback_sensitive_target_indices'] == []
    assert content_token_indices([1, 2, 3], [(0, 3), (3, 5), (5, 7)], [(1, 6)], set()) == ([1], [0, 2])
    assert content_token_indices([1, 2, 3], [(0, 3), (3, 5), (5, 7)], [(1, 6)], {2}) == ([], [0, 2])
    bad = copy.deepcopy(row)
    bad['segments'][1]['input_ids'][0] += 1
    try:
        saved_model_inputs(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Altered generation prefix was accepted')
    print(json.dumps({'status': 'passed', 'checks': ['Unicode/repeated content located by role',
        'generated/control/user tokens preserved', 'external tail omitted causally', 'no feedback stays unchanged',
        'boundary-spanning tokens excluded', 'altered saved prefix rejected'],
        'new_model_tokens': 0, 'gpu_jobs': 0, 'fixtures': 'synthetic token metadata only'}))


if __name__ == '__main__':
    main()
