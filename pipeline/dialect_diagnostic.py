"""Diagnostic: how much of an accuracy drop is pure output-serialization convention?

Terminal-agent SFT drifts the model's function-call dialect away from what the official
BFCL decoder accepts (see 09_evaluation_sensitivity.md 2.8). This script re-scores already
saved per-item outputs after normalizing three observed dialect forms, and reports both
numbers side by side.

**This is a diagnostic, not the official metric.** The official number stays official; the
normalized number exists only to separate "different serialization convention" from
"different capability". Always report both.

Dialect forms normalized (all observed in trained-model outputs; the list is not exhaustive, so
the reported serialization share is a lower bound):
  1. markdown fence        ```json\n[CALL]\n```
  2. envelope              {"function_calls": ["[CALL]"]}
  3. func_name/params      {"func_name": "X", "params": {...}}   (the terminus-2 harness dialect)

  python pipeline/dialect_diagnostic.py \
      --a smoke/eval_bfcl/java_js_b3072.jsonl \
      --b smoke/eval_bfcl/trained_java_js_b3072.jsonl --category simple_java
"""
import argparse, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.third_party/bfcl_eval_pkg'))
sys.path.insert(0, str(ROOT / 'pipeline'))

from eval_bfcl import install_model_config                                        # noqa: E402
install_model_config()

from eval_bfcl import LANGUAGE, MODEL_NAME, RETURN_FORMAT, load_category           # noqa: E402
from bfcl_eval.constants.enums import Language, ReturnFormat                       # noqa: E402
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker                # noqa: E402
from bfcl_eval.model_handler.utils import default_decode_ast_prompting             # noqa: E402



def _as_call(name, args):
    if not isinstance(args, dict):
        return None
    body = ', '.join(f'{k}={v!r}' if isinstance(v, str) else f'{k}={v}' for k, v in args.items())
    return f'{name}({body})'


def normalize(text):
    """Best-effort rewrite of an observed dialect form into the expected `[name(args)]`.

    This covers the forms seen in practice, not every possible one, so the serialization
    share it produces is a LOWER BOUND on the true share.
    """
    t = text.strip()
    if '</think>' in t:
        t = t.rsplit('</think>', 1)[1].strip()
    # drop a trailing/leading markdown fence, with or without a language tag
    for pattern in (re.compile(r'^```[a-zA-Z]*\s*\n?(.*?)\n?```$', re.S),
                    re.compile(r'^```[a-zA-Z]*\s*\n?(.*)$', re.S),
                    re.compile(r'^(.*?)\n?```$', re.S)):
        m = pattern.match(t)
        if m:
            t = m.group(1).strip()
            break
    try:
        obj = json.loads(t)
    except Exception:
        return t
    if not isinstance(obj, dict):
        return t
    calls = obj.get('function_calls')
    if isinstance(calls, list) and calls:
        rendered = []
        for c in calls:
            if isinstance(c, dict):
                inner = _as_call(c.get('name') or c.get('func_name'),
                                 c.get('parameters') or c.get('arguments') or c.get('params'))
                if inner:
                    rendered.append(inner)
            else:
                rendered.append(str(c).strip('[]'))
        if rendered:
            return '[' + ', '.join(rendered) + ']'
    for name_key, args_key in (('func_name', 'params'), ('name', 'arguments'),
                               ('name', 'parameters')):
        if name_key in obj and isinstance(obj.get(args_key), dict):
            inner = _as_call(obj[name_key], obj[args_key])
            if inner:
                return f'[{inner}]'
    return t


def correct(item, answers, category, text):
    try:
        decoded = default_decode_ast_prompting(
            normalize(text), RETURN_FORMAT.get(category, ReturnFormat.PYTHON), False)
    except Exception:
        return False
    if decoded is None:
        return False
    try:
        return bool(ast_checker(item['_check_function'], decoded, answers[item['id']],
                                LANGUAGE.get(category, Language.PYTHON), category,
                                MODEL_NAME)['valid'])
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--category', required=True)
    ap.add_argument('--out', default='smoke/compare/dialect_diagnostic.json')
    args = ap.parse_args()

    items, answers = load_category(args.category)
    by_id = {i['id']: i for i in items}
    report = {'category': args.category, 'a': args.a, 'b': args.b,
              'note': 'normalized = diagnostic only; the official metric is the official one'}
    for label, path in (('a', args.a), ('b', args.b)):
        rows = [json.loads(l) for l in open(ROOT / path)
                if json.loads(l)['category'] == args.category]
        official = sum(r['valid'] for r in rows)
        normalized = sum(1 for r in rows
                         if r['valid'] or correct(by_id[r['id']], answers, args.category, r['output']))
        report[label] = {'n': len(rows), 'official_correct': official,
                         'official_accuracy': round(official / len(rows), 4),
                         'normalized_correct': normalized,
                         'normalized_accuracy': round(normalized / len(rows), 4),
                         'recovered_by_normalization': normalized - official}

    ra, rb = report['a'], report['b']
    official_gap = ra['official_accuracy'] - rb['official_accuracy']
    normalized_gap = ra['normalized_accuracy'] - rb['normalized_accuracy']
    report['gap'] = {
        'official': round(official_gap, 4),
        'normalized': round(normalized_gap, 4),
        'attributable_to_serialization': round(official_gap - normalized_gap, 4),
        'serialization_share_of_gap': (round((official_gap - normalized_gap) / official_gap, 3)
                                       if official_gap else None),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
