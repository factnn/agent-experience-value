"""BFCL evaluation that drives the OFFICIAL inference + scoring loop with a local HF model.

The official runner normally talks to a vLLM server through an OpenAI-compatible
endpoint. Here a QwenHandler subclass keeps every official piece -- prompt
construction (`_pre_query_processing_prompting` -> `_format_prompt`), the
single-turn and multi-turn inference loops, function execution for multi-turn
categories, response parsing, AST decoding and `ast_checker` -- and replaces only
`_query_prompting` with local transformers generation.

Supported categories (all local, no external API):
  single-turn: simple_python, simple_java, simple_javascript, multiple,
               parallel, parallel_multiple, irrelevance
  multi-turn : multi_turn_base, multi_turn_long_context, multi_turn_miss_func,
               multi_turn_miss_param

  CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/eval_bfcl_local.py \
      --category multi_turn_base --limit 20 --tag base_mt20
"""
import argparse, collections, json, pathlib, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / '.third_party/bfcl_eval_pkg'
sys.path.insert(0, str(PKG))

MODEL_NAME = 'qwen3-4b'
MAX_CONTEXT = 40960


def install_model_config():
    """See pipeline/eval_bfcl.py: model_config pulls in every vendor API client."""
    try:
        import bfcl_eval.constants.model_config  # noqa: F401
        return 'real'
    except ImportError:
        import types
        shim = types.ModuleType('bfcl_eval.constants.model_config')

        class _Config:
            underscore_to_dot = False
        shim.MODEL_CONFIG_MAPPING = {MODEL_NAME: _Config()}
        sys.modules['bfcl_eval.constants.model_config'] = shim
        return 'shim'


MODEL_CONFIG_SOURCE = install_model_config()

from overrides import override                                                       # noqa: E402
from bfcl_eval.constants.enums import Language, ReturnFormat                          # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker                   # noqa: E402
from bfcl_eval.eval_checker.eval_runner import _evaluate_single_multi_turn_entry      # noqa: E402
from bfcl_eval.model_handler.local_inference.qwen import QwenHandler                  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (               # noqa: E402
    is_empty_execute_response,
)
from bfcl_eval.utils import (                                                         # noqa: E402
    is_empty_output,
    load_dataset_entry,
    load_ground_truth_entry,
)

DATA = PKG / 'bfcl_eval/data'
MULTI_TURN = ('multi_turn_base', 'multi_turn_long_context', 'multi_turn_miss_func',
              'multi_turn_miss_param')
LANGUAGE = {'simple_java': Language.JAVA, 'simple_javascript': Language.JAVASCRIPT}
RETURN_FORMAT = {'simple_java': ReturnFormat.JAVA, 'simple_javascript': ReturnFormat.JAVASCRIPT}


class _Text:
    def __init__(self, text):
        self.text = text


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Response:
    """OpenAI-completions-shaped shim so the official parser works unchanged."""

    def __init__(self, text, prompt_tokens, completion_tokens):
        self.choices = [_Text(text)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


def load_category(category):
    """Official loaders.

    Official split: the PROMPT comes from load_dataset_entry() with language hints
    (java/js parameter types rewritten to "string"), while ast_checker is called with
    the UNHINTED docs (eval_runner.py:683). Keep both.
    """
    items = load_dataset_entry(category)
    plain = {e['id']: e for e in load_dataset_entry(category, include_language_specific_hint=False)}
    for item in items:
        item['_check_function'] = plain[item['id']]['function']
    try:
        answers = {e['id']: e['ground_truth'] for e in load_ground_truth_entry(category)}
    except Exception:
        answers = {}   # relevance/irrelevance have no ground-truth file
    return items, answers


class LocalQwenHandler(QwenHandler):
    """Official Qwen prompting handler with local transformers generation."""

    def __init__(self, hf_model, tokenizer, max_new_tokens, **kwargs):
        super().__init__(**kwargs)
        self.hf_model = hf_model
        self.tokenizer = tokenizer
        self.max_context_length = MAX_CONTEXT
        self.max_new_tokens = max_new_tokens

    @override
    def _query_prompting(self, inference_data: dict):
        prompt = self._format_prompt(inference_data['message'], inference_data['function'])
        inference_data['inference_input_log'] = {'formatted_prompt': prompt}
        inputs = self.tokenizer(prompt, return_tensors='pt',
                                add_special_tokens=False).to('cuda')
        started = time.time()
        with torch.no_grad():
            generated = self.hf_model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                temperature=None, top_p=None, pad_token_id=self.tokenizer.pad_token_id)
        latency = time.time() - started
        prompt_tokens = int(inputs['input_ids'].shape[1])
        text = self.tokenizer.decode(generated[0][prompt_tokens:], skip_special_tokens=True)
        return _Response(text, prompt_tokens, int(generated.shape[1] - prompt_tokens)), latency


def score_single_turn(handler, item, answers, category):
    """Official parse -> decode -> ast_checker (or the official relevance/irrelevance rule)."""
    result, metadata = handler.inference(item, include_input_log=False, exclude_state_log=False)
    raw = result
    if category == 'irrelevance':
        try:
            decoded = handler.decode_ast(raw, ReturnFormat.PYTHON, False)
            contain_call = not is_empty_output(decoded)
        except Exception:
            contain_call = False
        return {'valid': not contain_call, 'decodable': True,
                'reason': 'ok' if not contain_call else 'made_function_call', 'raw': raw}
    try:
        decoded = handler.decode_ast(raw, RETURN_FORMAT.get(category, ReturnFormat.PYTHON), False)
    except Exception as exc:
        return {'valid': False, 'decodable': False,
                'reason': f'decode_failed:{type(exc).__name__}', 'raw': raw}
    if decoded is None:
        return {'valid': False, 'decodable': False,
                'reason': 'decode_failed:decoder_returned_none', 'raw': raw}
    checked = ast_checker(item['_check_function'], decoded, answers[item['id']],
                          LANGUAGE.get(category, Language.PYTHON), category, MODEL_NAME)
    return {'valid': bool(checked['valid']), 'decodable': True,
            'reason': 'ok' if checked['valid'] else checked.get('error_type', 'invalid'), 'raw': raw}


def multi_turn_parse_stats(handler, model_result):
    """Classify every model response the agent produced, using the official decoder.

    `model_result` is `all_model_response` from the official multi-turn loop: one list per
    turn, holding one string per model response in that turn (execution feedback goes to the
    chat history, not here). The first element of the tuple returned by score_multi_turn used
    to be a hardcoded True; this replaces it with an actual measurement.

    Categories are kept apart on purpose: a model that legitimately answers in prose instead of
    calling a function must not be counted as a parse failure.
    """
    stats = collections.Counter()
    samples = []
    for turn in model_result:
        for response in turn if isinstance(turn, list) else [turn]:
            stats['responses'] += 1
            if not isinstance(response, str) or not response.strip():
                stats['empty_response'] += 1
                continue
            if '</think>' not in response and '<think>' in response:
                stats['unterminated_think'] += 1
            try:
                decoded = handler.decode_execute(response, has_tool_call_tag=False)
            except Exception as exc:
                stats['parse_failed'] += 1
                if len(samples) < 5:
                    samples.append({'kind': 'parse_failed', 'error': str(exc)[:120],
                                    'response_tail': response[-200:]})
                continue
            if is_empty_execute_response(decoded):
                stats['no_call'] += 1
                if len(samples) < 5:
                    samples.append({'kind': 'no_call', 'response_tail': response[-200:]})
            else:
                stats['executable_call'] += 1
    stats['decodable_rate'] = (stats['executable_call'] / stats['responses']
                               if stats['responses'] else None)
    return dict(stats), samples


def score_multi_turn(handler, item, answers, category):
    """Official multi-turn inference (executes mock APIs) then official entry evaluation."""
    model_result, metadata = handler.inference(item, include_input_log=False, exclude_state_log=True)
    entry = _evaluate_single_multi_turn_entry(handler, item['id'], model_result,
                                              answers[item['id']], item, MODEL_NAME, category)
    stats, samples = multi_turn_parse_stats(handler, model_result)
    if entry['valid']:
        return {'valid': True, 'reason': 'ok', 'decodable': stats['decodable_rate'],
                'parse_stats': stats, 'parse_samples': samples, 'raw': model_result}
    error = entry.get('error', {})
    reason = error.get('error_type', 'invalid') if isinstance(error, dict) else str(error)[:80]
    return {'valid': False, 'reason': reason, 'decodable': stats['decodable_rate'],
            'parse_stats': stats, 'parse_samples': samples, 'raw': model_result}


def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(str(ROOT / args.model), dtype=torch.bfloat16)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / args.adapter))
    model.to('cuda')
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='smoke/model')
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--category', action='append', default=None)
    ap.add_argument('--limit', type=int, default=20, help='0 = all items')
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--max-new-tokens', type=int, default=640)
    ap.add_argument('--out', default='smoke/eval_bfcl')
    ap.add_argument('--tag', default='base_local')
    args = ap.parse_args()
    categories = args.category or ['simple_python']

    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / args.model))
    hf_model = load_model(args)
    handler = LocalQwenHandler(hf_model=hf_model, tokenizer=tokenizer,
                               max_new_tokens=args.max_new_tokens,
                               model_name=MODEL_NAME, temperature=0.0,
                               registry_name=MODEL_NAME, is_fc_model=False)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    full_dir = out / f'{args.tag}_full'      # untruncated per-item payloads
    full_dir.mkdir(exist_ok=True)
    log = (out / f'{args.tag}.jsonl').open('w')
    summaries, started = [], time.time()

    for category in categories:
        items, answers = load_category(category)
        items = items[args.offset:]
        if args.limit:
            items = items[:args.limit]
        multi = category in MULTI_TURN
        n_decodable = n_valid = 0
        parse_totals = collections.Counter()
        cat_started = time.time()
        for item in items:
            if multi:
                res = score_multi_turn(handler, item, answers, category)
                parse_totals.update({k: v for k, v in res.get('parse_stats', {}).items()
                                     if k != 'decodable_rate'})
            else:
                res = score_single_turn(handler, item, answers, category)
            decodable, valid, reason = res['decodable'], res['valid'], res['reason']
            if decodable:
                n_decodable += 1
            n_valid += valid
            # full payload saved separately; the jsonl keeps a short readable summary
            (full_dir / f"{item['id']}.json").write_text(
                json.dumps({'id': item['id'], 'category': category, 'valid': valid,
                            'reason': reason, 'decodable': decodable,
                            'parse_stats': res.get('parse_stats'),
                            'parse_samples': res.get('parse_samples'),
                            'raw': res['raw']}, ensure_ascii=False))
            log.write(json.dumps({'id': item['id'], 'category': category, 'decodable': decodable,
                                  'valid': valid, 'reason': reason,
                                  'output': json.dumps(res['raw'])[:4000],
                                  'parse_stats': res.get('parse_stats'),
                                  'full': f'{args.tag}_full/{item["id"]}.json'},
                                 ensure_ascii=False) + '\n')
            log.flush()
            print(json.dumps({'id': item['id'], 'valid': valid, 'reason': str(reason)[:90]}), flush=True)
        summary = {'category': category, 'n': len(items), 'valid': n_valid,
                   'accuracy': n_valid / len(items),
                   'seconds': round(time.time() - cat_started, 1), 'multi_turn': multi}
        if multi:
            # decodability is a per-response rate here, not a per-item count
            summary['decodable_measured'] = bool(parse_totals.get('responses'))
            summary['parse_stats'] = dict(parse_totals)
            summary['decodable_rate'] = (parse_totals['executable_call'] / parse_totals['responses']
                                         if parse_totals.get('responses') else None)
        else:
            summary['decodable'] = n_decodable
            summary['decodable_rate'] = n_decodable / len(items)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    log.close()

    report = {'tag': args.tag, 'model': args.model, 'adapter': args.adapter,
              'max_new_tokens': args.max_new_tokens, 'offset': args.offset,
              'bfcl_model_name': MODEL_NAME, 'model_config_source': MODEL_CONFIG_SOURCE,
              'runner': 'official bfcl_eval inference loops (single + multi turn) with local generation',
              'categories': summaries,
              'total_n': sum(s['n'] for s in summaries),
              'overall_accuracy': (sum(s['valid'] for s in summaries) /
                                   max(1, sum(s['n'] for s in summaries))),
              'seconds': round(time.time() - started, 1)}
    (out / f'{args.tag}_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
