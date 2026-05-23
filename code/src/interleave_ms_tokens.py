"""Interleave per-stream MS tokens into a single 'tokens' field.

Reads {train,dev,test}_tokens.pt from a MS VQ token cache (has tokens_body/hand/face)
and writes a baseline-compatible cache with a single 'tokens' field.

Layout matches eval_cross_slt_lift3d.get_stream_ranges_ms / decode_motion_ms:
  stream_order = sorted(splits.keys(), key=lambda n: splits[n][0])
  For csl_lift3d (body 0-8, face 8-136, hand 136-178) → (body, face, hand)
  Offsets: body=0, face=nb_per_stream, hand=2*nb_per_stream

Usage:
    python interleave_ms_tokens.py \
        --in-dir checkpoints/csl_lift3d/vq_csl_lift3d_ms_v2/tokens \
        --out-dir checkpoints/csl_lift3d/vq_csl_lift3d_ms_v2/tokens_interleaved \
        --dataname csl_lift3d \
        --nb-per-stream 1024
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.vqvae_multistream import KP_SPLITS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--dataname', default='csl_lift3d')
    ap.add_argument('--nb-per-stream', type=int, default=1024,
                     help='uniform per-stream codes (used if *_body/hand/face not set)')
    ap.add_argument('--nb-body', type=int, default=None)
    ap.add_argument('--nb-hand', type=int, default=None)
    ap.add_argument('--nb-face', type=int, default=None)
    ap.add_argument('--splits', default='train,dev,test')
    args = ap.parse_args()

    splits_layout = KP_SPLITS[args.dataname]
    stream_order = sorted(splits_layout.keys(), key=lambda n: splits_layout[n][0])
    print(f'[*] stream_order = {stream_order}')
    nb_per = {n: getattr(args, f'nb_{n}', None) or args.nb_per_stream for n in stream_order}
    print(f'[*] per-stream codes = {nb_per}')
    offsets = {}
    cursor = 0
    for s in stream_order:
        offsets[s] = cursor
        cursor += nb_per[s]
    print(f'[*] offsets = {offsets}, total num_vq = {cursor}')

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits.split(','):
        src = in_dir / f'{split}_tokens.pt'
        dst = out_dir / f'{split}_tokens.pt'
        print(f'[*] processing {split}: {src}')
        cache = torch.load(src, map_location='cpu', weights_only=False)
        new_cache = {}
        T_total = 0
        for sid, v in cache.items():
            t_b = v.get('tokens_body')
            t_f = v.get('tokens_face')
            t_h = v.get('tokens_hand')
            if t_b is None or t_f is None or t_h is None:
                continue
            T_tok = len(t_b)
            assert len(t_f) == T_tok and len(t_h) == T_tok, \
                f'{sid}: mismatched lengths body={len(t_b)} face={len(t_f)} hand={len(t_h)}'
            streams = {'body': t_b, 'face': t_f, 'hand': t_h}
            flat = np.zeros(T_tok * 3, dtype=np.int64)
            for i, s in enumerate(stream_order):
                flat[i::3] = streams[s].astype(np.int64) + offsets[s]
            new_cache[sid] = {
                'text': v.get('text', ''),
                'gloss': v.get('gloss', ''),
                'tokens': flat,
                'T_orig': v.get('T_orig', 0),
                'T_tok': T_tok * 3,
            }
            T_total += T_tok * 3
        torch.save(new_cache, dst)
        print(f'[OK] {split}: {len(new_cache)} samples, total interleaved tokens {T_total}')

    print(f'[DONE] wrote to {out_dir}')


if __name__ == '__main__':
    main()
