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
import argparse, json, pathlib, sys, time
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
        return (True, not contain_call, 'ok' if not contain_call else 'made_function_call',
                raw, metadata)
    try:
        decoded = handler.decode_ast(raw, RETURN_FORMAT.get(category, ReturnFormat.PYTHON), False)
    except Exception as exc:
        return (False, False, f'decode_failed:{type(exc).__name__}', raw, metadata)
    checked = ast_checker(item['_check_function'], decoded, answers[item['id']],
                          LANGUAGE.get(category, Language.PYTHON), category, MODEL_NAME)
    return (True, bool(checked['valid']), 'ok' if checked['valid'] else checked.get('error_type', 'invalid'),
            raw, metadata)


def score_multi_turn(handler, item, answers, category):
    """Official multi-turn inference (executes mock APIs) then official entry evaluation."""
    model_result, metadata = handler.inference(item, include_input_log=False, exclude_state_log=True)
    entry = _evaluate_single_multi_turn_entry(handler, item['id'], model_result,
                                              answers[item['id']], item, MODEL_NAME, category)
    if entry['valid']:
        return True, True, 'ok', json.dumps(model_result)[:4000], metadata
    error = entry.get('error', {})
    reason = error.get('error_type', 'invalid') if isinstance(error, dict) else str(error)[:80]
    return True, False, reason, json.dumps(model_result)[:4000], metadata


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
    log = (out / f'{args.tag}.jsonl').open('w')
    summaries, started = [], time.time()

    for category in categories:
        items, answers = load_category(category)
        items = items[args.offset:]
        if args.limit:
            items = items[:args.limit]
        multi = category in MULTI_TURN
        n_decodable = n_valid = 0
        cat_started = time.time()
        for item in items:
            if multi:
                decodable, valid, reason, raw, meta = score_multi_turn(handler, item, answers, category)
            else:
                decodable, valid, reason, raw, meta = score_single_turn(handler, item, answers, category)
            n_decodable += decodable
            n_valid += valid
            log.write(json.dumps({'id': item['id'], 'category': category, 'decodable': decodable,
                                  'valid': valid, 'reason': reason, 'output': raw,
                                  'prompt_tokens': meta.get('input_token_count'),
                                  'output_tokens': meta.get('output_token_count'),
                                  'latency': meta.get('latency')}, ensure_ascii=False) + '\n')
            log.flush()
            print(json.dumps({'id': item['id'], 'valid': valid, 'reason': str(reason)[:90]}), flush=True)
        summary = {'category': category, 'n': len(items), 'decodable': n_decodable, 'valid': n_valid,
                   'accuracy': n_valid / len(items), 'decodable_rate': n_decodable / len(items),
                   'seconds': round(time.time() - cat_started, 1), 'multi_turn': multi}
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
