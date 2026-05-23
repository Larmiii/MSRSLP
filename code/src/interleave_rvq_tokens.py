"""Interleave RVQ (single-stream base + residual) tokens into a single 'tokens' field.

Layout: at step k, stream = k%2; positions 0=base, 1=residual.
Offsets: base [0, nb_base); residual [nb_base, nb_base+nb_res).

Usage:
    python interleave_rvq_tokens.py \
        --in-dir <release>/checkpoints/csl/tokens/vq_M2 \
        --out-dir <release>/checkpoints/csl/tokens/vq_M2_interleaved \
        --nb-base 2048 --nb-res 2048
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--nb-base', type=int, default=2048)
    ap.add_argument('--nb-res', type=int, default=2048)
    ap.add_argument('--splits', default='train,dev,test')
    args = ap.parse_args()

    print(f'[*] offsets: base=0, residual={args.nb_base}')
    print(f'[*] total num_vq = {args.nb_base + args.nb_res}')

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits.split(','):
        src = in_dir / f'{split}_tokens.pt'
        dst = out_dir / f'{split}_tokens.pt'
        print(f'[*] {split}: {src}')
        cache = torch.load(src, map_location='cpu', weights_only=False)
        new_cache = {}
        T_total = 0
        for sid, v in cache.items():
            t_b = v.get('tokens_base')
            t_r = v.get('tokens_residual')
            if t_b is None or t_r is None:
                continue
            T_tok = len(t_b)
            assert len(t_r) == T_tok
            flat = np.zeros(T_tok * 2, dtype=np.int64)
            flat[0::2] = t_b.astype(np.int64)                      # base at even
            flat[1::2] = t_r.astype(np.int64) + args.nb_base       # residual at odd
            new_cache[sid] = {
                'text': v.get('text', ''),
                'gloss': v.get('gloss', ''),
                'tokens': flat,
                'T_orig': v.get('T_orig', 0),
                'T_tok': T_tok * 2,
            }
            T_total += T_tok * 2
        torch.save(new_cache, dst)
        print(f'[OK] {split}: {len(new_cache)} samples, total interleaved tokens {T_total}')


if __name__ == '__main__':
    main()
