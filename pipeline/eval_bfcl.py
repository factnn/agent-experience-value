"""BFCL evaluation with the OFFICIAL bfcl_eval prompt, parser, decoder and AST checker.

Generation is local (HF transformers, greedy) instead of the vLLM/OpenAI-compatible
server the official runner normally drives; everything downstream of generation is
the official code path, so scores are comparable to the leaderboard:
  prompt  : system_prompt_pre_processing_chat_model + QwenHandler._format_prompt
  parse   : QwenHandler._parse_query_response_prompting (strips think)
  decode  : default_decode_ast_prompting
  score   : eval_checker.ast_eval.ast_checker with model_name='qwen3-4b' (prompt variant)

Locally scorable categories (no external API): simple_python, simple_java,
simple_javascript, multiple, parallel, parallel_multiple, irrelevance.

  CUDA_VISIBLE_DEVICES=4 .venv-train/bin/python pipeline/eval_bfcl.py \
      --category simple_python --limit 50 --tag base_50
"""
import argparse, copy, json, pathlib, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / '.third_party/bfcl_eval_pkg'
sys.path.insert(0, str(PKG))

MODEL_NAME = 'qwen3-4b'          # prompt (non-FC) variant of Qwen3-4B in MODEL_CONFIG_MAPPING


def install_model_config():
    """bfcl_eval.constants.model_config imports every vendor API client (anthropic, cohere,
    mistralai, boto3, ...), which we do not need and do not want to install. Try the real
    module first; if its imports are missing, register a minimal stand-in carrying only the
    field the AST checker reads. `underscore_to_dot=False` is the value in the pinned wheel
    for the `qwen3-4b` prompt variant (bfcl_eval/constants/model_config.py:936)."""
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

from bfcl_eval.constants.enums import Language, ReturnFormat                        # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker                 # noqa: E402
from bfcl_eval.model_handler.local_inference.qwen import QwenHandler                # noqa: E402
from bfcl_eval.model_handler.utils import (                                         # noqa: E402
    default_decode_ast_prompting,
    system_prompt_pre_processing_chat_model,
)
from bfcl_eval.utils import (                                                         # noqa: E402
    is_empty_output,
    load_dataset_entry,
    load_ground_truth_entry,
)

DATA = PKG / 'bfcl_eval/data'
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
    """Minimal stand-in for the API response the official Qwen parser expects."""

    def __init__(self, text, prompt_tokens, completion_tokens):
        self.choices = [_Text(text)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


def load_category(category):
    """Official loaders.

    The official runner builds the PROMPT from load_dataset_entry() with language
    hints (which rewrite java/js parameter types to "string") but calls ast_checker
    with the UNHINTED docs (official eval_runner.py:683). Both are kept here:
    the hinted docs drive the prompt, the plain ones drive the checker.
    """
    items = load_dataset_entry(category)
    plain = {e['id']: e for e in load_dataset_entry(category, include_language_specific_hint=False)}
    for item in items:
        item['_check_function'] = plain[item['id']]['function']
    try:
        answers = {e['id']: e['ground_truth'] for e in load_ground_truth_entry(category)}
    except Exception:
        answers = {}
    return items, answers


def build_prompt(item, category):
    messages = copy.deepcopy(item['question'][0])
    messages = system_prompt_pre_processing_chat_model(messages, item['function'], item['id'])
    return QwenHandler._format_prompt(None, messages, item['function'])


def score(item, category, text, prompt_tokens, completion_tokens):
    """Return (decodable, valid, reason) using official parse/decode/check."""
    parsed = QwenHandler._parse_query_response_prompting(
        None, _Response(text, prompt_tokens, completion_tokens))
    response = parsed['model_responses']
    if category == 'irrelevance':
        # Official rule (eval_runner.py:290): correct behaviour is to produce NO function call.
        # A decode failure means the output was not call format, which PASSES here.
        try:
            decoded = default_decode_ast_prompting(response, ReturnFormat.PYTHON, False)
            contain_call = not is_empty_output(decoded)
        except Exception:
            contain_call = False
        ok = not contain_call
        return True, ok, ('ok' if ok else f'made_call:{str(decoded)[:120]}')
    answers = item['_answers']
    try:
        decoded = default_decode_ast_prompting(
            response, RETURN_FORMAT.get(category, ReturnFormat.PYTHON), False)
    except Exception as exc:
        return False, False, f'decode_failed:{type(exc).__name__}:{str(exc)[:120]}'
    if decoded is None:
        # the official decoder can return None instead of raising; ast_checker would
        # then fail with "object of type 'NoneType' has no len()"
        return False, False, 'decode_failed:decoder_returned_none'
    result = ast_checker(item['_check_function'], decoded, answers[item['id']],
                         LANGUAGE.get(category, Language.PYTHON), category, MODEL_NAME)
    if result['valid']:
        return True, True, 'ok'
    return True, False, result.get('error_type', 'invalid')


def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(str(ROOT / args.model), dtype=torch.bfloat16)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ROOT / args.adapter))
    model.to('cuda')
    model.eval()
    return model


def generate_batch(model, tok, prompts, max_new_tokens):
    """Greedy generation for one batch of prompts (left-padded); returns (texts, prompt_lens, out_lens)."""
    enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to('cuda')
    with torch.no_grad():
        generated = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                   temperature=None, top_p=None, pad_token_id=tok.pad_token_id)
    prompt_lens = enc['attention_mask'].sum(dim=1).tolist()
    texts = []
    for row, plen in zip(generated, prompt_lens):
        texts.append(tok.decode(row[plen:], skip_special_tokens=True))
    return texts, prompt_lens, [int(generated.shape[1] - p) for p in prompt_lens]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='smoke/model')
    ap.add_argument('--adapter', default=None)
    ap.add_argument('--category', action='append', default=None)
    ap.add_argument('--limit', type=int, default=50, help='0 = all items')
    ap.add_argument('--offset', type=int, default=0,
                    help='skip this many items first; allows sharding one evaluation across GPUs')
    ap.add_argument('--max-new-tokens', type=int, default=640)
    ap.add_argument('--batch-size', type=int, default=1,
                    help='greedy batch size; verified to reproduce batch-1 output byte for byte')
    ap.add_argument('--out', default='smoke/eval_bfcl')
    ap.add_argument('--tag', default='base')
    args = ap.parse_args()
    categories = args.category or ['simple_python']

    tok = AutoTokenizer.from_pretrained(str(ROOT / args.model))
    if args.batch_size > 1:
        tok.padding_side = 'left'
    model = load_model(args)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = (out / f'{args.tag}.jsonl').open('w')
    summaries, started = [], time.time()

    for category in categories:
        items, answers = load_category(category)
        items = items[args.offset:]
        if args.limit:
            items = items[:args.limit]
        n_decodable = n_valid = 0
        cat_started = time.time()
        for start in range(0, len(items), args.batch_size):
            chunk = items[start:start + args.batch_size]
            prompts = [build_prompt(item, category) for item in chunk]
            texts, prompt_lens, out_lens = generate_batch(model, tok, prompts, args.max_new_tokens)
            for item, text, ptok, otok in zip(chunk, texts, prompt_lens, out_lens):
                item['_answers'] = answers
                decodable, valid, reason = score(item, category, text, ptok, otok)
                n_decodable += decodable
                n_valid += valid
                log.write(json.dumps({'id': item['id'], 'category': category,
                                      'prompt_tokens': int(ptok), 'output_tokens': int(otok),
                                      'output': text[:4000], 'decodable': decodable,
                                      'valid': valid, 'reason': reason}, ensure_ascii=False) + '\n')
                log.flush()
                print(json.dumps({'id': item['id'], 'valid': valid, 'reason': reason[:80]}), flush=True)
        summary = {'category': category, 'n': len(items), 'decodable': n_decodable,
                   'valid': n_valid, 'accuracy': n_valid / len(items),
                   'decodable_rate': n_decodable / len(items),
                   'seconds': round(time.time() - cat_started, 1)}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    log.close()

    report = {'tag': args.tag, 'model': args.model, 'adapter': args.adapter,
              'max_new_tokens': args.max_new_tokens, 'offset': args.offset,
              'batch_size': args.batch_size,
              'bfcl_model_name': MODEL_NAME,
              'model_config_source': MODEL_CONFIG_SOURCE,
              'scorer': 'official bfcl_eval ast_checker + official prompts/parser/decoder',
              'categories': summaries, 'seconds': round(time.time() - started, 1),
              'total_n': sum(s['n'] for s in summaries),
              'overall_accuracy': (sum(s['valid'] for s in summaries) /
                                   sum(s['n'] for s in summaries))}
    (out / f'{args.tag}_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
