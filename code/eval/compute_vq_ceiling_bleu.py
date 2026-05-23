"""Compute VQ ceiling BLEU: GT pose → VQ encode → VQ decode → SLT pickle → BT eval.

This gives the upper bound on what trans+VQ can achieve under the same BT evaluator.

Usage:
    python compute_vq_ceiling_bleu.py \
        --variant msr \
        --vq-ckpt ../checkpoints/csl/vq/vq_M1M2.pt \
        --splits dev,test \
        --out ../results/vq_ceiling_M1M2/pickles

Then run BT eval on the output pickle dir.
"""
from __future__ import annotations
import argparse, gzip, pickle, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

LIFT3D_DATA_DIR = Path(__file__).resolve().parents[2] / 'data'


def _resolve_stats(ckpt_path):
    p = Path(ckpt_path)
    stem = p.stem
    candidates = [
        p.parent / 'stats' / f'{stem}_mean.npy',
        p.parent / f'{stem}_mean.npy',
        p.parent / 'mean.npy',
    ]
    for m in candidates:
        s = m.with_name(m.name.replace('_mean', '_std').replace('mean', 'std'))
        if m.exists() and s.exists():
            return np.load(m), np.load(s)
    raise FileNotFoundError(f"mean/std for {ckpt_path}")


def load_vq(variant, ckpt_path, device):
    if variant == 'base':
        import models.vqvae as vqvae
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        a = SimpleNamespace(**ck['args'])
        m = vqvae.VQVAE_251(a, nb_code=a.nb_code, code_dim=a.code_dim,
                             output_emb_width=a.output_emb_width,
                             down_t=a.down_t, stride_t=a.stride_t, width=a.width,
                             depth=a.depth, dilation_growth_rate=a.dilation_growth_rate,
                             activation=a.vq_act, norm=a.vq_norm)
    elif variant == 'ms':
        from models.vqvae_multistream import MultiStreamVQVAE, KP_SPLITS
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        a = SimpleNamespace(**ck['args'])
        stream_codes = None
        if any(getattr(a, f'nb_code_{n}', None) for n in KP_SPLITS[a.dataname]):
            stream_codes = {n: (getattr(a, f'nb_code_{n}', None) or a.nb_code) for n in KP_SPLITS[a.dataname]}
        m = MultiStreamVQVAE(a, dataset_name=a.dataname, nb_code=a.nb_code,
                              stream_codes=stream_codes,
                              code_dim=a.code_dim, output_emb_width=a.output_emb_width,
                              down_t=a.down_t, stride_t=a.stride_t, width=a.width,
                              depth=a.depth, dilation_growth_rate=a.dilation_growth_rate,
                              activation=a.vq_act, norm=a.vq_norm)
    elif variant == 'rvq':
        from models.vqvae_residual import ResidualVQVAE
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        a = SimpleNamespace(**ck['args'])
        m = ResidualVQVAE(a, nb_code=a.nb_code, code_dim=a.code_dim,
                           output_emb_width=a.output_emb_width,
                           down_t=a.down_t, stride_t=a.stride_t, width=a.width,
                           depth=a.depth, dilation_growth_rate=a.dilation_growth_rate,
                           activation=a.vq_act, norm=a.vq_norm,
                           nb_code_residual=a.nb_code_residual)
    else:  # msr
        from models.vqvae_multistream_residual import MultiStreamResidualVQVAE
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        a = SimpleNamespace(**ck['args'])
        sb = {'body': a.nb_base_body, 'hand': a.nb_base_hand, 'face': a.nb_base_face}
        sr = {'body': a.nb_res_body, 'hand': a.nb_res_hand, 'face': a.nb_res_face}
        m = MultiStreamResidualVQVAE(a, dataset_name=a.dataname, code_dim=a.code_dim,
                                       output_emb_width=a.output_emb_width, down_t=a.down_t,
                                       stride_t=a.stride_t, width=a.width, depth=a.depth,
                                       dilation_growth_rate=a.dilation_growth_rate,
                                       activation=a.vq_act, norm=a.vq_norm,
                                       stream_codes_base=sb, stream_codes_residual=sr)
    m.load_state_dict(ck['model']); m.to(device).eval()
    mean, std = _resolve_stats(ckpt_path)
    return m, mean, std, a


@torch.no_grad()
def encode_decode(vq, motion_534, mean, std, device, variant, av):
    mn = (motion_534 - mean) / std
    x = torch.from_numpy(mn).float().unsqueeze(0).to(device)
    if variant == 'base':
        toks = vq.encode(x)
        rec = vq.forward_decoder(toks)
    elif variant == 'rvq':
        bi, ri = vq.encode(x)
        rec = vq.forward_decoder(bi, ri)
    else:                                          # ms / msr
        rec = vq(x)[0]
    return (rec[0].cpu().numpy() * std + mean).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['base', 'ms', 'rvq', 'msr'])
    ap.add_argument('--vq-ckpt', required=True)
    ap.add_argument('--dataset', default='csl', choices=['csl', 'phix'])
    ap.add_argument('--splits', default='dev,test')
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f'[*] loading VQ ({args.variant}) from {args.vq_ckpt}')
    vq, mean, std, av = load_vq(args.variant, args.vq_ckpt, args.device)

    for split in args.splits.split(','):
        if args.dataset == 'csl':
            src = LIFT3D_DATA_DIR / 'csl' / f'csl_daily_lift3d.{split}.pt'
            pickle_name = f'csl_daily.{split}'
        else:
            src = LIFT3D_DATA_DIR / 'phix' / f'phix_lift3d.{split}.pt'
            pickle_name = f'{split}.pickle'
        gt = torch.load(src, map_location='cpu', weights_only=False)
        sids = list(gt.keys())
        print(f'[*] {split}: {len(sids)} samples')

        out_list = []; t0 = time.time()
        for sid in tqdm(sids):
            v = gt[sid]
            pose = v['poses_3d']
            if torch.is_tensor(pose): pose = pose.numpy()
            T = pose.shape[0]
            motion = pose.reshape(T, -1).astype(np.float32)
            rec = encode_decode(vq, motion, mean, std, args.device, args.variant, av)
            out_list.append({
                'name': sid, 'signer': '', 'gloss': v.get('gloss', ''), 'text': v.get('text', ''),
                'sign': torch.from_numpy(rec).float(),
            })

        out_path = out_dir / pickle_name
        with gzip.open(out_path, 'wb') as f:
            pickle.dump(out_list, f, protocol=4)
        elapsed = time.time() - t0
        Ts = [s['sign'].shape[0] for s in out_list]
        print(f'[OK] {split}: {len(out_list)}, frames {min(Ts)}/{np.mean(Ts):.1f}/{max(Ts)}, '
              f'elapsed {elapsed:.0f}s')


if __name__ == '__main__':
    main()
