"""Fetch the two large third-party assets this repo does not commit.

  1. Qwen/Qwen3-4B weights  -> smoke/model/            (~7.6 GB)
  2. official bfcl_eval wheel -> .third_party/bfcl_eval_pkg/  (~14 MB unpacked)

Both are pinned. Nothing here needs credentials; everything is public.

  .venv-train/bin/python pipeline/fetch_assets.py            # both
  .venv-train/bin/python pipeline/fetch_assets.py --model    # weights only
  .venv-train/bin/python pipeline/fetch_assets.py --bfcl     # BFCL only

If PyPI is slow from your network, point pip at a mirror, e.g.
  PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple .venv-train/bin/python pipeline/fetch_assets.py --bfcl
"""
import argparse, pathlib, subprocess, sys, tempfile, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'smoke/model'
BFCL_PKG_DIR = ROOT / '.third_party/bfcl_eval_pkg'
MODEL_ID = 'Qwen/Qwen3-4B'
BFCL_VERSION = 'bfcl_eval==2026.3.23'
BFCL_REQUIRED = [
    'bfcl_eval/eval_checker/ast_eval/ast_checker.py',
    'bfcl_eval/model_handler/local_inference/qwen.py',
    'bfcl_eval/constants/enums.py',
    'bfcl_eval/data/BFCL_v4_simple_python.json',
    'bfcl_eval/data/possible_answer/BFCL_v4_simple_python.json',
]


def fetch_model():
    from huggingface_hub import snapshot_download
    print(f'downloading {MODEL_ID} -> {MODEL_DIR}', flush=True)
    snapshot_download(repo_id=MODEL_ID, local_dir=str(MODEL_DIR))
    index = MODEL_DIR / 'model.safetensors.index.json'
    assert index.exists(), f'missing {index}'
    shards = sorted(MODEL_DIR.glob('model-*.safetensors'))
    total = sum(p.stat().st_size for p in shards)
    assert len(shards) == 3, f'expected 3 shards, found {len(shards)}'
    print(f'ok: {len(shards)} shards, {total / 2**30:.2f} GiB')


def fetch_bfcl():
    """Download the official wheel and unpack it. It is pure Python; we deliberately do
    NOT pip-install it, because its dependency list pulls every vendor API client."""
    if BFCL_PKG_DIR.exists():
        missing = [f for f in BFCL_REQUIRED if not (BFCL_PKG_DIR / f).exists()]
        if not missing:
            print(f'ok: {BFCL_PKG_DIR} already present')
            return
    BFCL_PKG_DIR.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        print(f'downloading {BFCL_VERSION}', flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps',
                        '--dest', tmp, BFCL_VERSION], check=True)
        wheels = list(pathlib.Path(tmp).glob('bfcl_eval-*.whl'))
        assert len(wheels) == 1, f'expected one wheel, found {wheels}'
        with zipfile.ZipFile(wheels[0]) as z:
            z.extractall(BFCL_PKG_DIR)
    missing = [f for f in BFCL_REQUIRED if not (BFCL_PKG_DIR / f).exists()]
    assert not missing, f'unpacked package is missing {missing}'
    print(f'ok: unpacked to {BFCL_PKG_DIR}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', action='store_true', help='fetch Qwen3-4B weights only')
    ap.add_argument('--bfcl', action='store_true', help='fetch the official bfcl_eval package only')
    args = ap.parse_args()
    both = not (args.model or args.bfcl)
    if args.model or both:
        fetch_model()
    if args.bfcl or both:
        fetch_bfcl()
    print('\nnext: .venv-train/bin/python -m unittest discover -s pipeline -p "test_*.py"')


if __name__ == '__main__':
    main()
