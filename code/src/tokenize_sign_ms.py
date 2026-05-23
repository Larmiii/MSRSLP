"""Tokenize PHIX/CSL data using Multi-Stream VQ-VAE.

Outputs cache file with per-sample dict:
    {'text', 'gloss', 'T_orig', 'T_tok',
     'tokens_body', 'tokens_hand', 'tokens_face'}   # each (T_tok,) int array

Usage:
    python tokenize_sign_ms.py --dataname phix --vq-ckpt output_sign/vq_phix_ms/best.pt
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.vqvae_multistream import MultiStreamVQVAE
from dataset.dataset_VQ_sign import REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname', required=True, choices=['phix', 'csl', 'phix14t', 'csl_lift3d', 'phix_lift3d'])
    ap.add_argument('--vq-ckpt', required=True)
    ap.add_argument('--splits', default='train,dev,test')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--max-len', type=int, default=200)
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--stats-dir', default=None)
    ap.add_argument('--mean-name', default=None)
    ap.add_argument('--std-name', default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.vq_ckpt, map_location='cpu', weights_only=False)
    vq_args = SimpleNamespace(**ckpt['args'])
    # Support asymmetric per-stream codes if present in args
    from models.vqvae_multistream import KP_SPLITS
    _stream_codes = None
    _splits = KP_SPLITS[args.dataname]
    if any(getattr(vq_args, f'nb_code_{n}', None) for n in _splits):
        _stream_codes = {n: (getattr(vq_args, f'nb_code_{n}', None) or vq_args.nb_code) for n in _splits}
        print(f'[*] asymmetric stream_codes: {_stream_codes}')
    model = MultiStreamVQVAE(
        vq_args, dataset_name=args.dataname,
        nb_code=vq_args.nb_code, code_dim=vq_args.code_dim,
        stream_codes=_stream_codes,
        output_emb_width=vq_args.output_emb_width,
        down_t=vq_args.down_t, stride_t=vq_args.stride_t,
        width=vq_args.width, depth=vq_args.depth,
        dilation_growth_rate=vq_args.dilation_growth_rate,
        activation=vq_args.vq_act, norm=vq_args.vq_norm,
    )
    model.load_state_dict(ckpt['model'])
    model.to(args.device).eval()
    print(f'[*] MS VQ-VAE loaded from {args.vq_ckpt} (streams={list(model.stream_dims.keys())})')

    ckpt_path = Path(args.vq_ckpt); ckpt_dir = ckpt_path.parent; stem = ckpt_path.stem
    stats_dir = Path(args.stats_dir) if args.stats_dir else ckpt_dir
    mean_name = args.mean_name or f'{stem}_mean.npy'
    std_name = args.std_name or f'{stem}_std.npy'
    mean_path = None
    for cand in [stats_dir / mean_name, stats_dir / 'stats' / mean_name,
                  stats_dir / 'mean.npy', ckpt_dir / 'mean.npy']:
        if cand.exists(): mean_path = cand; break
    if mean_path is None: raise FileNotFoundError(f'mean.npy for {args.vq_ckpt}')
    std_path = mean_path.with_name(mean_path.name.replace('mean', 'std'))
    mean = np.load(mean_path); std = np.load(std_path)
    print(f'[*] using stats from {mean_path}')

    out_dir = Path(args.out_dir) if args.out_dir else (ckpt_dir / 'tokens')
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits.split(','):
        if args.dataname == 'phix':
            raw = torch.load(REPO_ROOT / f"data/SLRTP-178/data/{split}.pt",
                              map_location='cpu', weights_only=False)
            pose_key = 'poses_3d'; text_key = 'text'
        elif args.dataname == 'phix_lift3d':
            raw = torch.load(Path(__file__).resolve().parents[2] / 'data' / 'phix' / f'phix_lift3d.{split}.pt', map_location='cpu', weights_only=False)
            pose_key = 'poses_3d'; text_key = 'text'
        elif args.dataname == 'csl_lift3d':
            raw = torch.load(Path(__file__).resolve().parents[2] / 'data' / 'csl' / f'csl_daily_lift3d.{split}.pt',
                              map_location='cpu', weights_only=False)
            pose_key = 'poses_3d'; text_key = 'text'
        elif args.dataname == 'phix14t':
            with open(REPO_ROOT / f"mska_bt/data/Phoenix-2014T/Phoenix-2014T.{split}", 'rb') as f:
                raw = pickle.load(f)
            pose_key = 'keypoint'; text_key = 'text'
        else:
            with open(REPO_ROOT / f"mska_bt/data/CSL-Daily/CSL-Daily.{split}", 'rb') as f:
                raw = pickle.load(f)
            pose_key = 'keypoint'; text_key = 'text'

        print(f'[*] tokenizing {split}: {len(raw)} samples')
        cache = {}
        with torch.no_grad():
            for sid, sample in tqdm(raw.items()):
                pose = sample[pose_key]
                if torch.is_tensor(pose): pose = pose.numpy()
                T = pose.shape[0]
                if T > args.max_len:
                    pose = pose[:args.max_len]; T = args.max_len
                motion = pose.reshape(T, -1).astype(np.float32)
                # Clip outliers in CSL/PHIX-14T HRNet data to image bounds
                if args.dataname in ('csl', 'phix14t'):
                    _W, _H = {'csl': (512.0, 512.0), 'phix14t': (210.0, 260.0)}[args.dataname]
                    motion[:, 0::3] = np.clip(motion[:, 0::3], 0.0, _W)
                    motion[:, 1::3] = np.clip(motion[:, 1::3], 0.0, _H)
                    motion[:, 2::3] = np.clip(motion[:, 2::3], 0.0, 1.0)
                motion = (motion - mean) / std
                inp = torch.from_numpy(motion).unsqueeze(0).to(args.device)
                tokens = model.encode(inp)   # dict of stream → (1, T_tok)
                entry = {
                    'text': sample.get(text_key, ''),
                    'gloss': sample.get('gloss', ''),
                    'T_orig': T,
                    'T_tok': tokens[next(iter(tokens))].size(1),
                }
                for name, t in tokens.items():
                    entry[f'tokens_{name}'] = t.squeeze(0).cpu().numpy().astype(np.int32)
                cache[sid] = entry

        out_path = out_dir / f'{split}_tokens.pt'
        torch.save(cache, out_path)
        ts = [v['T_tok'] for v in cache.values()]
        orig = [v['T_orig'] for v in cache.values()]
        print(f'    saved {out_path} | {len(cache)} samples | tok len min/mean/max = '
              f'{min(ts)}/{np.mean(ts):.1f}/{max(ts)} | orig {min(orig)}/{np.mean(orig):.1f}/{max(orig)}')


if __name__ == '__main__':
    main()
