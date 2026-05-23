"""Tokenize PHIX/CSL using trained Residual VQ-VAE.

Each sample → {'text', 'gloss', 'T_orig', 'T_tok',
              'tokens_base', 'tokens_residual'}
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.vqvae_residual import ResidualVQVAE
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
    model = ResidualVQVAE(
        vq_args, nb_code=vq_args.nb_code, code_dim=vq_args.code_dim,
        output_emb_width=vq_args.output_emb_width,
        down_t=vq_args.down_t, stride_t=vq_args.stride_t,
        width=vq_args.width, depth=vq_args.depth,
        dilation_growth_rate=vq_args.dilation_growth_rate,
        activation=vq_args.vq_act, norm=vq_args.vq_norm,
        nb_code_residual=vq_args.nb_code_residual,
    )
    model.load_state_dict(ckpt['model'])
    model.to(args.device).eval()
    print(f'[*] R-VQ-VAE loaded (base nb={vq_args.nb_code}, residual nb={vq_args.nb_code_residual})')

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
            pose_key = 'poses_3d'
        elif args.dataname == 'phix_lift3d':
            raw = torch.load(Path(__file__).resolve().parents[2] / 'data' / 'phix' / f'phix_lift3d.{split}.pt', map_location='cpu', weights_only=False)
            pose_key = 'poses_3d'
        elif args.dataname == 'csl_lift3d':
            raw = torch.load(Path(__file__).resolve().parents[2] / 'data' / 'csl' / f'csl_daily_lift3d.{split}.pt',
                              map_location='cpu', weights_only=False)
            pose_key = 'poses_3d'
        elif args.dataname == 'phix14t':
            with open(REPO_ROOT / f"mska_bt/data/Phoenix-2014T/Phoenix-2014T.{split}", 'rb') as f:
                raw = pickle.load(f)
            pose_key = 'keypoint'
        else:
            with open(REPO_ROOT / f"mska_bt/data/CSL-Daily/CSL-Daily.{split}", 'rb') as f:
                raw = pickle.load(f)
            pose_key = 'keypoint'

        print(f'[*] tokenizing {split}: {len(raw)} samples')
        cache = {}
        with torch.no_grad():
            for sid, sample in tqdm(raw.items()):
                pose = sample[pose_key]
                if torch.is_tensor(pose): pose = pose.numpy()
                T = pose.shape[0]
                if T > args.max_len: pose = pose[:args.max_len]; T = args.max_len
                motion = pose.reshape(T, -1).astype(np.float32)
                # Clip outliers in CSL/PHIX-14T HRNet data to image bounds
                if args.dataname in ('csl', 'phix14t'):
                    _W, _H = {'csl': (512.0, 512.0), 'phix14t': (210.0, 260.0)}[args.dataname]
                    motion[:, 0::3] = np.clip(motion[:, 0::3], 0.0, _W)
                    motion[:, 1::3] = np.clip(motion[:, 1::3], 0.0, _H)
                    motion[:, 2::3] = np.clip(motion[:, 2::3], 0.0, 1.0)
                motion = (motion - mean) / std
                inp = torch.from_numpy(motion).unsqueeze(0).to(args.device)
                base_idx, res_idx = model.encode(inp)
                cache[sid] = {
                    'text': sample.get('text', ''),
                    'gloss': sample.get('gloss', ''),
                    'T_orig': T, 'T_tok': base_idx.size(1),
                    'tokens_base': base_idx.squeeze(0).cpu().numpy().astype(np.int32),
                    'tokens_residual': res_idx.squeeze(0).cpu().numpy().astype(np.int32),
                }
        out_path = out_dir / f'{split}_tokens.pt'
        torch.save(cache, out_path)
        ts = [v['T_tok'] for v in cache.values()]
        print(f'    saved {out_path} ({len(cache)} samples, T_tok {min(ts)}/{np.mean(ts):.1f}/{max(ts)})')


if __name__ == '__main__':
    main()
