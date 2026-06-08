"""MDM sampling on sign data -> SLRTP pickle (DDIM + classifier-free guidance).

    python gen_mdm_sign.py --dataset phix --name mdm_phix --text mbart:de_DE --splits dev,test
"""
import argparse, gzip, pickle, os
from os.path import join as pjoin
import numpy as np
import torch
import sys
sys.path.insert(0, '.')
from train_mdm_sign import MDM, cosine_beta_schedule
from sign_text_encoder import build_sign_text_encoder

SRC = {
    'phix': (r'D:\Graduation\MSRSLP\data\phix', 'phix_lift3d.{split}.pt', '{split}.pickle'),
    'csl':  (r'D:\Graduation\MSRSLP\data\csl', 'csl_daily_lift3d.{split}.pt', 'csl_daily.{split}'),
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['phix', 'csl'])
    ap.add_argument('--name', required=True)
    ap.add_argument('--text', default='mbart:de_DE')
    ap.add_argument('--splits', default='dev,test')
    ap.add_argument('--latent', type=int, default=512)
    ap.add_argument('--ff', type=int, default=1024)
    ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--T_diff', type=int, default=1000)
    ap.add_argument('--ddim_steps', type=int, default=50)
    ap.add_argument('--guidance', type=float, default=2.5)
    ap.add_argument('--max_motion_length', type=int, default=196)
    ap.add_argument('--batch_size', type=int, default=32)
    args = ap.parse_args()

    device = torch.device('cuda')
    data_dir, tmpl, pkl_tmpl = SRC[args.dataset]
    droot = pjoin('.', 'dataset', f'{args.dataset}_sign')
    ckpt = pjoin('.', 'sign_ckpt', f'{args.dataset}_sign', args.name, 'net_best.tar')
    out_dir = pjoin('.', 'sign_results', f'{args.dataset}_mdm_e2e')
    os.makedirs(out_dir, exist_ok=True)

    mean = np.load(pjoin(droot, 'Mean.npy')); std = np.load(pjoin(droot, 'Std.npy'))
    mean_t = torch.from_numpy(mean).float().to(device); std_t = torch.from_numpy(std).float().to(device)

    src_lang = args.text.split(':')[1] if ':' in args.text else 'de_DE'
    txt, tdim = build_sign_text_encoder('mbart', device, src_lang=src_lang)
    model = MDM(534, args.latent, args.ff, args.layers, args.heads, text_dim=tdim).to(device)
    sd = torch.load(ckpt, map_location='cpu'); model.load_state_dict(sd['model']); model.eval()
    print(f'[*] loaded {ckpt} (iter {sd.get("iter")})')

    acp = torch.cumprod(1 - cosine_beta_schedule(args.T_diff), dim=0).to(device)
    seq = torch.linspace(0, args.T_diff - 1, args.ddim_steps).long().flip(0).tolist()

    for split in args.splits.split(','):
        d = torch.load(pjoin(data_dir, tmpl.format(split=split)), map_location='cpu', weights_only=False)
        sids = list(d.keys())
        out_list = []
        for i in range(0, len(sids), args.batch_size):
            chunk = sids[i:i + args.batch_size]
            caps = [(d[s].get('text', '') or '').strip() for s in chunk]
            lens = [min(int(d[s]['poses_3d'].shape[0]), args.max_motion_length) for s in chunk]
            T = max(lens); B = len(chunk)
            pad = torch.arange(T, device=device)[None] >= torch.tensor(lens, device=device)[:, None]
            tf = txt(caps)
            null = model.null_text[None].expand(B, -1)
            x = torch.randn(B, T, 534, device=device)
            for j, t in enumerate(seq):
                tb = torch.full((B,), t, device=device, dtype=torch.long)
                x0c = model(x, tb, tf, pad)
                x0u = model(x, tb, null, pad)
                x0 = x0u + args.guidance * (x0c - x0u)
                at = acp[t]
                eps = (x - at.sqrt() * x0) / (1 - at).sqrt().clamp(min=1e-6)
                t_prev = seq[j + 1] if j + 1 < len(seq) else -1
                ap_prev = acp[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=device)
                x = ap_prev.sqrt() * x0 + (1 - ap_prev).sqrt() * eps
            pred = (x0 * std_t + mean_t)
            for k, s in enumerate(chunk):
                sign = pred[k, :lens[k]].cpu().float()
                out_list.append({'name': s, 'signer': '', 'gloss': d[s].get('gloss', ''),
                                 'text': d[s].get('text', ''), 'sign': sign})
        out_path = pjoin(out_dir, pkl_tmpl.format(split=split))
        with gzip.open(out_path, 'wb') as f:
            pickle.dump(out_list, f, protocol=4)
        Ts = [s['sign'].shape[0] for s in out_list]
        print(f'[OK] {split}: {len(out_list)} samples, frames {min(Ts)}/{np.mean(Ts):.1f}/{max(Ts)} -> {out_path}')


if __name__ == '__main__':
    main()
