"""Bucket analysis: BLEU stratified by sentence length.

Given a BT eval log (signjoey test output) parses per-sample BLEU OR
reconstructs from predicted/reference text and groups by text length.

Inputs:
  --pred-pickle: SLP gen pickle (contains 'text' field per sample)
  --gt-pickle:   GT pose pickle (for sentence lengths)
  --references:  signjoey ref/hyp .txt output (per-sample text)

Output: BLEU per (short / medium / long) bucket.

Note: requires running with sacrebleu or signjoey-compatible BLEU calc.
"""
from __future__ import annotations
import argparse, gzip, pickle
from pathlib import Path

import numpy as np
import torch


def text_length(t: str) -> int:
    """Char-length for Chinese, word-length for German (whitespace-tokenized)."""
    if not t: return 0
    return len(t.replace(' ', '')) if any('一' <= c <= '鿿' for c in t) else len(t.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-pickle', required=True,
                     help='SLP gen pickle')
    ap.add_argument('--ref-txt', required=True, help='reference text file (one line per sample)')
    ap.add_argument('--hyp-txt', required=True, help='hypothesis text file (one line per sample)')
    ap.add_argument('--lang', default='zh', choices=['zh', 'de'])
    ap.add_argument('--buckets', default='10,20',
                     help='comma-separated boundaries: short<a, a<=med<b, long>=b')
    args = ap.parse_args()

    # Load samples for sid order
    with gzip.open(args.pred_pickle, 'rb') as f:
        preds = pickle.load(f)
    print(f'[*] {len(preds)} samples')

    # Load refs / hyps
    with open(args.ref_txt, 'r', encoding='utf-8') as f:
        refs = [l.rstrip('\n') for l in f]
    with open(args.hyp_txt, 'r', encoding='utf-8') as f:
        hyps = [l.rstrip('\n') for l in f]
    assert len(refs) == len(hyps) == len(preds), \
        f'mismatch: refs={len(refs)} hyps={len(hyps)} preds={len(preds)}'

    # Bucket by GT text length
    boundaries = sorted([int(x) for x in args.buckets.split(',')])
    bucket_labels = ['short', 'medium', 'long']
    buckets = {l: {'refs': [], 'hyps': [], 'lens': []} for l in bucket_labels}
    for i, p in enumerate(preds):
        L = text_length(p.get('text', ''))
        if L < boundaries[0]:
            b = 'short'
        elif L < boundaries[1]:
            b = 'medium'
        else:
            b = 'long'
        buckets[b]['refs'].append(refs[i])
        buckets[b]['hyps'].append(hyps[i])
        buckets[b]['lens'].append(L)

    # Compute corpus BLEU per bucket
    try:
        import sacrebleu
        for b in bucket_labels:
            n = len(buckets[b]['refs'])
            if n == 0:
                print(f'{b}: no samples')
                continue
            bleu = sacrebleu.corpus_bleu(buckets[b]['hyps'], [buckets[b]['refs']])
            mean_len = np.mean(buckets[b]['lens'])
            print(f'{b:>6s} (n={n}, len mean={mean_len:.1f}, range {min(buckets[b]["lens"])}-{max(buckets[b]["lens"])}): '
                  f'BLEU-4 = {bleu.score:.2f}')
    except ImportError:
        print('sacrebleu not installed; install: pip install sacrebleu')
        print('Per-bucket sample counts:')
        for b in bucket_labels:
            n = len(buckets[b]['refs'])
            print(f'{b:>6s}: n={n}, len range {min(buckets[b]["lens"]) if n else "-"}-{max(buckets[b]["lens"]) if n else "-"}')


if __name__ == '__main__':
    main()
