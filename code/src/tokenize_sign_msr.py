"""Tokenize PHIX/CSL with Multi-Stream Residual VQ-VAE.

Each sample → {'text', 'gloss', 'T_orig', 'T_tok',
              'tokens_body_base', 'tokens_body_res',
              'tokens_hand_base', 'tokens_hand_res',
              'tokens_face_base', 'tokens_face_res'}
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.vqvae_multistream_residual import MultiStreamResidualVQVAE
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
    a = SimpleNamespace(**ckpt['args'])
    stream_codes_base = {'body': a.nb_base_body, 'hand': a.nb_base_hand, 'face': a.nb_base_face}
    stream_codes_res = {'body': a.nb_res_body, 'hand': a.nb_res_hand, 'face': a.nb_res_face}
    model = MultiStreamResidualVQVAE(
        a, dataset_name=args.dataname,
        code_dim=a.code_dim, output_emb_width=a.output_emb_width,
        down_t=a.down_t, stride_t=a.stride_t,
        width=a.width, depth=a.depth,
        dilation_growth_rate=a.dilation_growth_rate,
        activation=a.vq_act, norm=a.vq_norm,
        stream_codes_base=stream_codes_base,
        stream_codes_residual=stream_codes_res,
    )
    model.load_state_dict(ckpt['model']); model.to(args.device).eval()

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
                toks = model.encode(inp)
                entry = {'text': sample.get('text', ''),
                         'gloss': sample.get('gloss', ''),
                         'T_orig': T,
                         'T_tok': toks['body_base'].size(1)}
                for k, v in toks.items():
                    entry[f'tokens_{k}'] = v.squeeze(0).cpu().numpy().astype(np.int32)
                cache[sid] = entry
        out_path = out_dir / f'{split}_tokens.pt'
        torch.save(cache, out_path)
        print(f'    saved {out_path} ({len(cache)} samples)')


if __name__ == '__main__':
    main()
