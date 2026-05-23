"""Tokenize PHIX-14T or CSL-Daily full pose sequences using trained VQ-VAE.

Outputs cache file with {sample_id: {text, gloss, tokens (T_tok,), T_orig}}.
Used as input for Stage 2 (text → motion token Transformer).

Usage:
    python tokenize_sign.py --dataname phix --vq-ckpt output_sign/vq_phix_base/best.pt
    python tokenize_sign.py --dataname csl  --vq-ckpt output_sign/vq_csl_base/best.pt
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models.vqvae as vqvae
from dataset.dataset_VQ_sign import VQSignDataset, REPO_ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataname', required=True, choices=['phix', 'csl', 'phix14t', 'csl_lift3d', 'phix_lift3d'])
    ap.add_argument('--vq-ckpt', required=True, help='path to trained VQ-VAE checkpoint')
    ap.add_argument('--splits', default='train,dev,test')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--max-len', type=int, default=200)
    ap.add_argument('--out-dir', default=None, help='override output directory for tokens')
    ap.add_argument('--stats-dir', default=None, help='override mean.npy/std.npy lookup (also tries <stats-dir>/<stem>_{mean,std}.npy)')
    ap.add_argument('--mean-name', default=None, help='custom mean.npy filename (e.g. vq_baseline_mean.npy)')
    ap.add_argument('--std-name', default=None, help='custom std.npy filename')
    args = ap.parse_args()

    ckpt = torch.load(args.vq_ckpt, map_location='cpu', weights_only=False)
    vq_args = SimpleNamespace(**ckpt['args'])
    model = vqvae.VQVAE_251(
        vq_args, nb_code=vq_args.nb_code, code_dim=vq_args.code_dim,
        output_emb_width=vq_args.output_emb_width,
        down_t=vq_args.down_t, stride_t=vq_args.stride_t,
        width=vq_args.width, depth=vq_args.depth,
        dilation_growth_rate=vq_args.dilation_growth_rate,
        activation=vq_args.vq_act, norm=vq_args.vq_norm,
    )
    model.load_state_dict(ckpt['model'])
    model.to(args.device).eval()
    print(f'[*] VQ-VAE loaded from {args.vq_ckpt} (input_dim={model.input_dim})')

    # Load mean/std saved alongside checkpoint (with overrides)
    ckpt_path = Path(args.vq_ckpt)
    ckpt_dir = ckpt_path.parent
    stem = ckpt_path.stem
    stats_dir = Path(args.stats_dir) if args.stats_dir else ckpt_dir
    mean_name = args.mean_name or f'{stem}_mean.npy'
    std_name = args.std_name or f'{stem}_std.npy'
    mean_path = None
    for cand in [stats_dir / mean_name, stats_dir / 'stats' / mean_name,
                  stats_dir / 'mean.npy', ckpt_dir / 'mean.npy']:
        if cand.exists():
            mean_path = cand; break
    std_path = mean_path.with_name(mean_path.name.replace('mean', 'std')) if mean_path else None
    if mean_path is None or not std_path.exists():
        raise FileNotFoundError(f'mean/std not found for {args.vq_ckpt}')
    mean = np.load(mean_path); std = np.load(std_path)
    print(f'[*] using stats from {mean_path}')

    out_dir = Path(args.out_dir) if args.out_dir else (ckpt_dir / 'tokens')
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits.split(','):
        # Load raw data (full sequence, not windowed)
        if args.dataname == 'phix':
            raw_path = REPO_ROOT / "data/SLRTP-178/data" / f"{split}.pt"
            raw = torch.load(raw_path, map_location='cpu', weights_only=False)
            text_key, pose_key = 'text', 'poses_3d'
        elif args.dataname == 'phix_lift3d':
            raw_path = Path(__file__).resolve().parents[2] / 'data' / 'phix' / f'phix_lift3d.{split}.pt'
            raw = torch.load(raw_path, map_location='cpu', weights_only=False)
            pose_key = 'poses_3d'; text_key = 'text'
        elif args.dataname == 'csl_lift3d':
            raw_path = Path(str(Path(__file__).resolve().parents[2] / "data" / "csl")) / f"csl_daily_lift3d.{split}.pt"
            raw = torch.load(raw_path, map_location='cpu', weights_only=False)
            text_key, pose_key = 'text', 'poses_3d'
        elif args.dataname == 'phix14t':
            with open(REPO_ROOT / "mska_bt/data/Phoenix-2014T" / f"Phoenix-2014T.{split}", 'rb') as f:
                raw = pickle.load(f)
            text_key, pose_key = 'text', 'keypoint'
        else:
            with open(REPO_ROOT / "mska_bt/data/CSL-Daily" / f"CSL-Daily.{split}", 'rb') as f:
                raw = pickle.load(f)
            text_key, pose_key = 'text', 'keypoint'

        print(f'[*] tokenizing {split}: {len(raw)} samples')
        cache = {}
        with torch.no_grad():
            for sid, sample in tqdm(raw.items()):
                pose = sample[pose_key]
                if torch.is_tensor(pose):
                    pose = pose.numpy()
                T = pose.shape[0]
                if T > args.max_len:
                    pose = pose[:args.max_len]
                    T = args.max_len
                # flatten + normalize
                motion = pose.reshape(T, -1).astype(np.float32)
                # Clip outliers in CSL/PHIX-14T HRNet data to image bounds
                if args.dataname in ('csl', 'phix14t'):
                    _W, _H = {'csl': (512.0, 512.0), 'phix14t': (210.0, 260.0)}[args.dataname]
                    motion[:, 0::3] = np.clip(motion[:, 0::3], 0.0, _W)
                    motion[:, 1::3] = np.clip(motion[:, 1::3], 0.0, _H)
                    motion[:, 2::3] = np.clip(motion[:, 2::3], 0.0, 1.0)
                motion = (motion - mean) / std
                # encode → tokens (model encode expects (B, T, D))
                inp = torch.from_numpy(motion).unsqueeze(0).to(args.device)
                tokens = model.encode(inp)   # (1, T_tok)
                tokens = tokens.squeeze(0).cpu().numpy().astype(np.int32)
                cache[sid] = {
                    'text': sample.get(text_key, ''),
                    'gloss': sample.get('gloss', ''),
                    'tokens': tokens,
                    'T_orig': T,
                    'T_tok': len(tokens),
                }

        out_path = out_dir / f'{split}_tokens.pt'
        torch.save(cache, out_path)
        # Stats
        n_tok = [v['T_tok'] for v in cache.values()]
        n_orig = [v['T_orig'] for v in cache.values()]
        print(f'    saved {out_path} | {len(cache)} samples | '
              f'tokens len min/mean/max = {min(n_tok)}/{np.mean(n_tok):.1f}/{max(n_tok)} | '
              f'orig frames min/mean/max = {min(n_orig)}/{np.mean(n_orig):.1f}/{max(n_orig)} | '
              f'compression ratio ≈ {np.mean(n_orig)/np.mean(n_tok):.2f}x')


if __name__ == '__main__':
    main()
