"""Portable entry for the separately reported E1/E2 experiments.

Commands are dry-run by default. --execute runs only the requested stage.
Data and pretrained/model weights are supplied separately by the user.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
VARIANTS = ('baseline', 'single_large', 'tlrvq', 'pamvq', 'pamrvq', 'pamrvq_random')


def build_command(args):
    if args.dataset == 'csl' and args.variant in ('single_large', 'pamrvq_random'):
        raise ValueError('Published CSL controls use baseline4096 vs tlrvq2048+2048; random grouping is PHX-only')
    command = [str(args.python), '-u']
    if args.stage == 'vq':
        command += [str(BASE / 'tools/run_vq_experiment.py')]
    else:
        command += [str(BASE / 'tools/prepare_and_train_ar.py'), '--stage',
                    'tokenize' if args.stage == 'tokenize' else 'train',
                    '--protocol', str(args.protocol.resolve())]
        if args.char_vocab_path:
            command += ['--char-vocab-path', str(args.char_vocab_path.resolve())]
        if args.mbart_name:
            command += ['--mbart-name', args.mbart_name]
    command += ['--dataset', args.dataset, '--variant', args.variant, '--seed', str(args.seed),
                '--data-root', str(args.data_root.expanduser().resolve()),
                '--run-root', str(args.run_root.expanduser().resolve()), '--python', str(args.python)]
    if not args.execute:
        command.append('--dry-run')
    return command


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('list', 'vq', 'tokenize', 'ar'))
    ap.add_argument('--dataset', choices=('phix', 'csl'))
    ap.add_argument('--variant', choices=VARIANTS)
    ap.add_argument('--seed', type=int, default=42, choices=(42, 43, 44))
    ap.add_argument('--data-root', type=Path, default=Path(os.environ.get('SLP_SIGN_DATA_DIR', str(BASE / 'runtime/data'))))
    ap.add_argument('--run-root', type=Path, default=BASE / 'runs')
    ap.add_argument('--protocol', type=Path, default=BASE / 'configs/protocol.json')
    ap.add_argument('--python', type=Path, default=Path(sys.executable))
    ap.add_argument('--char-vocab-path', type=Path)
    ap.add_argument('--mbart-name', help='Original mBART model ID or the same model in a local directory')
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args(argv)
    if args.stage == 'list':
        print((BASE / 'configs/experiments.json').read_text(encoding='utf-8'))
        return 0
    if not args.dataset or not args.variant:
        ap.error('--dataset and --variant are required for a stage')
    env = os.environ.copy()
    env['SLP_SIGN_DATA_DIR'] = str(args.data_root.expanduser().resolve())
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return subprocess.run(build_command(args), env=env, cwd=BASE).returncode


if __name__ == '__main__':
    raise SystemExit(main())
