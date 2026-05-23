"""Interleave MSR per-substream tokens into a single 'tokens' field.

MSR has 6 substreams: body_base, body_res, hand_base, hand_res, face_base, face_res
(SUB_ORDER from dataset_TM_sign_msr).

Layout: at step k, stream = SUB_ORDER[k % 6]; offsets cumulative over SUB_ORDER.

Usage:
    python interleave_msr_tokens.py \
        --in-dir checkpoints/csl_lift3d/vq_csl_lift3d_msr_v2/tokens \
        --out-dir checkpoints/csl_lift3d/vq_csl_lift3d_msr_v2/tokens_interleaved \
        --nb-base-body 512 --nb-res-body 512 \
        --nb-base-hand 512 --nb-res-hand 512 \
        --nb-base-face 512 --nb-res-face 512
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset.dataset_TM_sign_msr import SUB_ORDER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--nb-base-body', type=int, default=512)
    ap.add_argument('--nb-res-body',  type=int, default=512)
    ap.add_argument('--nb-base-hand', type=int, default=512)
    ap.add_argument('--nb-res-hand',  type=int, default=512)
    ap.add_argument('--nb-base-face', type=int, default=512)
    ap.add_argument('--nb-res-face',  type=int, default=512)
    ap.add_argument('--splits', default='train,dev,test')
    args = ap.parse_args()

    sub_codes = {
        'body_base': args.nb_base_body, 'body_res': args.nb_res_body,
        'hand_base': args.nb_base_hand, 'hand_res': args.nb_res_hand,
        'face_base': args.nb_base_face, 'face_res': args.nb_res_face,
    }
    print(f'[*] SUB_ORDER = {SUB_ORDER}')
    offsets = {}
    off = 0
    for s in SUB_ORDER:
        offsets[s] = off
        off += sub_codes[s]
    print(f'[*] offsets = {offsets}')
    total = sum(sub_codes.values())
    print(f'[*] total num_vq = {total}')

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits.split(','):
        src = in_dir / f'{split}_tokens.pt'
        dst = out_dir / f'{split}_tokens.pt'
        print(f'[*] processing {split}: {src}')
        cache = torch.load(src, map_location='cpu', weights_only=False)
        new_cache = {}
        for sid, v in cache.items():
            substreams = {s: v.get(f'tokens_{s}') for s in SUB_ORDER}
            if any(t is None for t in substreams.values()):
                continue
            T_tok = len(substreams[SUB_ORDER[0]])
            assert all(len(t) == T_tok for t in substreams.values()), \
                f'{sid}: mismatched lengths'
            flat = np.zeros(T_tok * len(SUB_ORDER), dtype=np.int64)
            for i, s in enumerate(SUB_ORDER):
                flat[i::len(SUB_ORDER)] = substreams[s].astype(np.int64) + offsets[s]
            new_cache[sid] = {
                'text': v.get('text', ''),
                'gloss': v.get('gloss', ''),
                'tokens': flat,
                'T_orig': v.get('T_orig', 0),
                'T_tok': T_tok * len(SUB_ORDER),
            }
        torch.save(new_cache, dst)
        print(f'[OK] {split}: {len(new_cache)} samples')

    print(f'[DONE] wrote to {out_dir}')


if __name__ == '__main__':
    main()
