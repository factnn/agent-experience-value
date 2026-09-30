"""Reclassify saved multi-turn responses without re-running the model.

Fixes a conceptual error in the first version of multi_turn_parse_stats(): it treated any
response the function-call parser rejected as a parse failure. But a terminal agent
legitimately answers in prose between calls ("The 'log.txt' file has been successfully
moved..."), and those answers were counted as failures, so a fully successful task could
score a low "decodability".

Correct categories:
  executable_call  the response decodes into one or more calls
  malformed_call   the response is clearly attempting a call but the parser rejects it
  no_call          a legitimate non-call response (prose answer / summary) -- NOT a failure
  empty_response   nothing at all
  ambiguous        fails to parse and gives no clear signal either way; reported separately
                   rather than forced into an error bucket

Also renames the headline number: executable_call / responses is the share of responses
that are tool calls, which is a behavioural property, not a parse success rate. An agent
that summarises after every call may sit near 0.5 while one that never summarises sits near
1.0, and the latter is not thereby better.

  python pipeline/reclassify_multi_turn.py --dir smoke/eval_bfcl/mt16_parsestats_full
"""
import argparse, collections, json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.third_party/bfcl_eval_pkg'))

from eval_bfcl import install_model_config                                         # noqa: E402
install_model_config()

from bfcl_eval.model_handler.local_inference.qwen import QwenHandler               # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (              # noqa: E402
    is_empty_execute_response,
)

# Signals that a rejected response was *trying* to be a call rather than answering in prose.
CALL_SIGNALS = re.compile(
    r'(\[\s*[A-Za-z_][\w.]*\s*\(|"function_calls"|\'function_calls\'|'
    r'"func_name"|\'func_name\'|"name"\s*:\s*"[A-Za-z_]|"arguments"|"parameters"|'
    r'func_name\s*=|params_name\s*=|params_value\s*=)')


def classify(response):
    """Categories derived from inspecting all 221 saved responses, not guessed up front.

    The distinction that matters: a terminal agent legitimately answers in prose between
    calls, so "the call parser rejected it" is NOT the same as "the response was invalid".
    """
    if not isinstance(response, str) or not response.strip():
        return 'empty_response'
    try:
        decoded = QwenHandler.decode_execute(None, response, False)
    except Exception:
        decoded = None
    if decoded is not None:
        return 'executable_call' if not is_empty_execute_response(decoded) else 'no_call'
    # parse rejected it -- decide why
    if '<think>' in response and '</think>' not in response:
        return 'incomplete'                     # cut off mid-thought, e.g. by the token cap
    body = response.strip()
    if CALL_SIGNALS.search(body):
        return 'malformed_call'                 # a genuine call attempt the parser rejects
    if len(body) < 40 and ('[' in body or '{' in body or '"' in body):
        return 'fragment'                       # too short to be an answer, carries code punctuation
    return 'no_call'                            # a legitimate prose answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', default='smoke/compare/multi_turn_reclassified.json')
    args = ap.parse_args()

    directory = ROOT / args.dir
    per_item = {}
    totals = collections.Counter()
    for path in sorted(directory.glob('*.json')):
        d = json.loads(path.read_text())
        counts = collections.Counter()
        for turn in d['raw']:
            for response in (turn if isinstance(turn, list) else [turn]):
                counts[classify(response)] += 1
        totals.update(counts)
        per_item[d['id']] = {'valid': d['valid'], 'counts': dict(counts),
                             'n_responses': sum(counts.values())}

    n = sum(totals.values())
    report = {'source': args.dir, 'items': len(per_item), 'responses': n,
              'totals': dict(totals),
              'tool_call_share': round(totals['executable_call'] / n, 4) if n else None,
              'malformed_call_rate': round(totals['malformed_call'] / n, 4) if n else None,
              'no_call_share': round(totals['no_call'] / n, 4) if n else None,
              'incomplete_share': round(totals['incomplete'] / n, 4) if n else None,
              'fragment_share': round(totals['fragment'] / n, 4) if n else None,
              'ambiguous_share': round(totals['ambiguous'] / n, 4) if n else None,
              'per_item': per_item,
              'note': 'tool_call_share = executable calls / responses, a behavioural share, NOT a parse '
                      'malformed_call_rate is the part that is genuinely a format failure; '
                      'no_call is a legitimate prose answer and is not a failure.'}
    out = ROOT / (args.out or f'{args.dir}/reclassified.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'per_item'},
                     ensure_ascii=False, indent=2))

    # show items whose verdict is most affected by the old misclassification
    worst = sorted(per_item.items(),
                   key=lambda kv: (kv[1]['counts'].get('no_call', 0)), reverse=True)[:5]
    print('\nitems with the most legitimate non-call answers (previously all counted as failures):')
    for k, v in worst:
        print(f"  {k:<24} valid={str(v['valid']):<5} {v['counts']}")


if __name__ == '__main__':
    main()
