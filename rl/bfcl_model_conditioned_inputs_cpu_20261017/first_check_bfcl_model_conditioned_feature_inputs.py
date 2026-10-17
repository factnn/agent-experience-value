"""Exact saved-token inputs for feedback-content sensitivity and reference likelihood.

Only external bridge messages are re-rendered, solely to locate tool-content
spans. Generated model tokens, all control tokens, user questions and positions
stay untouched. This module does not load a model or compute a value forecast.
"""
import copy


def render_bridge(tokenizer, messages, enable_thinking):
    return '\n' + tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=enable_thinking)


def content_token_indices(ids, offsets, spans, special_ids):
    selected, boundary = [], []
    for i, (a, b) in enumerate(offsets):
        if a >= b or ids[i] in special_ids:
            continue
        if any(start <= a and b <= end for start, end in spans):
            selected.append(i)
        elif any(a < end and start < b for start, end in spans):
            boundary.append(i)
    return selected, boundary


def saved_model_inputs(row):
    prompt = row['prompt_ids']
    completion = row['completion_ids']
    mask = row['model_token_mask']
    assert prompt and len(completion) == len(mask)
    assert all(type(x) is int and x in (0, 1) for x in mask)
    coverage = [None] * len(completion)
    for segment in row['segments']:
        start, ids = segment['start'], segment['generated_ids']
        assert segment['input_ids'] == prompt + completion[:start]
        assert ids == completion[start:start + len(ids)]
        for i in range(start, start + len(ids)):
            assert coverage[i] is None and mask[i] == 1
            coverage[i] = 1
    for bridge in row['bridges']:
        start, ids = bridge['start'], bridge['ids']
        assert ids == completion[start:start + len(ids)]
        for i in range(start, start + len(ids)):
            assert coverage[i] is None and mask[i] == 0
            coverage[i] = 0
    assert coverage == mask
    positions = [len(prompt) + i for i, value in enumerate(mask) if value]
    # Unscored external text after the final model token cannot affect causal
    # likelihoods, and is omitted from the forward input without re-tokenizing.
    cut = positions[-1] + 1 if positions else len(prompt)
    sequence = prompt + completion
    return {'input_ids': sequence[:cut], 'target_positions': positions,
            'target_ids': [sequence[i] for i in positions],
            'full_sequence_length': len(sequence), 'scored_prefix_length': cut,
            'unscored_external_tail_tokens': len(sequence) - cut}


def feedback_inputs(row, tokenizer, replacement_token_id=13):
    result = saved_model_inputs(row)
    assert tokenizer('.', add_special_tokens=False)['input_ids'] == [replacement_token_id]
    special = set(tokenizer.all_special_ids)
    assert replacement_token_id not in special
    original = row['prompt_ids'] + row['completion_ids']
    all_content, boundaries = set(), set()
    tool_messages, rendered_content_chars = 0, 0
    for bridge_index, bridge in enumerate(row['bridges']):
        messages = bridge['messages']
        text = render_bridge(tokenizer, messages, row['enable_thinking'])
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        assert encoded['input_ids'] == bridge['ids'], 'External bridge reconstruction differs'
        spans = []
        for message_index, message in enumerate(messages):
            if message['role'] != 'tool':
                continue
            assert isinstance(message['content'], str)
            tool_messages += 1
            marker = 'EXPERIENCE_VALUE_CONTENT_SENTINEL_%d_%d_20261017' % (bridge_index, message_index)
            assert marker not in text
            changed = copy.deepcopy(messages)
            changed[message_index]['content'] = marker
            marked = render_bridge(tokenizer, changed, row['enable_thinking'])
            assert marked.count(marker) == 1
            prefix, suffix = marked.split(marker)
            start, end = len(prefix), len(text) - len(suffix)
            assert start <= end and text[:start] == prefix and text[end:] == suffix
            spans.append((start, end))
            rendered_content_chars += end - start
        spans.sort()
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
        selected, boundary = content_token_indices(encoded['input_ids'], encoded['offset_mapping'], spans, special)
        base = len(row['prompt_ids']) + bridge['start']
        for indices, destination in [(selected, all_content), (boundary, boundaries)]:
            for i in indices:
                assert row['model_token_mask'][bridge['start'] + i] == 0
                destination.add(base + i)
    prefix_length = result['scored_prefix_length']
    inside = sorted(i for i in all_content if i < prefix_length)
    changed = [i for i in inside if original[i] != replacement_token_id]
    ablated = result['input_ids'].copy()
    for i in changed:
        assert row['model_token_mask'][i - len(row['prompt_ids'])] == 0
        assert original[i] not in special
        ablated[i] = replacement_token_id
    assert len(ablated) == len(result['input_ids'])
    assert all(ablated[i] == original[i] for i in result['target_positions'])
    earliest = changed[0] if changed else None
    eligible = [j for j, position in enumerate(result['target_positions'])
                if earliest is not None and position > earliest]
    result.update({'ablated_input_ids': ablated,
                   'feedback_sensitive_target_indices': eligible,
                   'changed_feedback_positions': changed,
                   'mask_audit': {'tool_messages': tool_messages,
                       'rendered_tool_content_characters': rendered_content_chars,
                       'tool_content_tokens_in_full_trace': len(all_content),
                       'tool_content_tokens_in_scored_prefix': len(inside),
                       'changed_tool_content_tokens_in_scored_prefix': len(changed),
                       'boundary_tokens_preserved': len(boundaries),
                       'model_tokens': len(result['target_positions']),
                       'eligible_post_feedback_model_tokens': len(eligible),
                       'first_changed_position': earliest,
                       'replacement_token_id': replacement_token_id,
                       'sequence_length_preserved': True, 'generated_tokens_preserved': True,
                       'special_control_tokens_preserved': True}})
    return result
