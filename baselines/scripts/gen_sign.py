"""MoMask end-to-end generation on sign data -> SLRTP pickle.

text -> MaskTransformer.generate (base tokens) -> ResidualTransformer.generate
(all layers) -> RVQVAE.forward_decoder -> poses. Uses GT motion length
(token_lens = frames//4, capped at training max) as MoMask is standardly
evaluated. Output pickle matches MSRSLP slrtp_eval format.

    python gen_sign.py --dataset phix --vq_name momask_vq_phix \
        --mtrans momask_mtrans_phix --rtrans momask_rtrans_phix --text mbart:de_DE --splits dev,test
"""
import argparse, gzip, pickle, os
from os.path import join as pjoin
from types import SimpleNamespace
import numpy as np
import torch
import sys
sys.path.insert(0, '.')
from models.vq.model import RVQVAE
from models.mask_transformer.transformer import MaskTransformer, ResidualTransformer

SRC = {
    'phix': (r'D:\Graduation\MSRSLP\data\phix', 'phix_lift3d.{split}.pt', '{split}.pickle'),
    'csl':  (r'D:\Graduation\MSRSLP\data\csl', 'csl_daily_lift3d.{split}.pt', 'csl_daily.{split}'),
}


def load_trans(cls, ckpt_dir, name, device, **kw):
    m = cls(**kw).to(device)
    sd = torch.load(pjoin(ckpt_dir, name, 'net_best.tar'), map_location='cpu')
    m.load_state_dict(sd['trans']); m.eval()
    print(f'[*] loaded {name} (iter {sd.get("iter")})')
    return m


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['phix', 'csl'])
    ap.add_argument('--vq_name', required=True)
    ap.add_argument('--mtrans', required=True)
    ap.add_argument('--rtrans', required=True)
    ap.add_argument('--text', default='mbart:de_DE')
    ap.add_argument('--splits', default='dev,test')
    ap.add_argument('--num_quantizers', type=int, default=6)
    ap.add_argument('--nb_code', type=int, default=512)
    ap.add_argument('--code_dim', type=int, default=512)
    ap.add_argument('--latent_dim', type=int, default=384)
    ap.add_argument('--ff_size', type=int, default=1024)
    ap.add_argument('--n_layers', type=int, default=8)
    ap.add_argument('--n_heads', type=int, default=6)
    ap.add_argument('--max_motion_length', type=int, default=196)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--timesteps', type=int, default=10)
    ap.add_argument('--cond_scale', type=float, default=4.0)
    ap.add_argument('--res_cond_scale', type=float, default=5.0)
    ap.add_argument('--temperature', type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device('cuda')
    data_dir, tmpl, pkl_tmpl = SRC[args.dataset]
    droot = pjoin('.', 'dataset', f'{args.dataset}_sign')
    ckpt_dir = pjoin('.', 'sign_ckpt', f'{args.dataset}_sign')
    out_dir = pjoin('.', 'sign_results', f'{args.dataset}_momask_e2e')
    os.makedirs(out_dir, exist_ok=True)
    max_tok = args.max_motion_length // 4

    mean = np.load(pjoin(droot, 'Mean.npy')); std = np.load(pjoin(droot, 'Std.npy'))
    mean_t = torch.from_numpy(mean).float().to(device); std_t = torch.from_numpy(std).float().to(device)

    vqa = SimpleNamespace(num_quantizers=args.num_quantizers, shared_codebook=False,
                          quantize_dropout_prob=0.2, quantize_dropout_cutoff_index=0,
                          mu=0.99, nb_code=args.nb_code, code_dim=args.code_dim)
    vq = RVQVAE(vqa, 534, args.nb_code, args.code_dim, args.code_dim, 2, 2, 512, 3, 3, 'relu', None).to(device)
    vq.load_state_dict(torch.load(pjoin(ckpt_dir, args.vq_name, 'net_best.tar'), map_location='cpu')['vq_model'])
    vq.eval()

    topt = SimpleNamespace(num_tokens=args.nb_code, num_quantizers=args.num_quantizers, device=device)
    mt = load_trans(MaskTransformer, ckpt_dir, args.mtrans, device,
                    code_dim=args.code_dim, cond_mode='text', latent_dim=args.latent_dim,
                    ff_size=args.ff_size, num_layers=args.n_layers, num_heads=args.n_heads,
                    dropout=0.1, clip_dim=1024, cond_drop_prob=0.1, clip_version=args.text, opt=topt)
    rt = load_trans(ResidualTransformer, ckpt_dir, args.rtrans, device,
                    code_dim=args.code_dim, cond_mode='text', latent_dim=args.latent_dim,
                    ff_size=args.ff_size, num_layers=args.n_layers, num_heads=args.n_heads,
                    dropout=0.1, clip_dim=1024, shared_codebook=False, share_weight=False,
                    cond_drop_prob=0.1, clip_version=args.text, opt=topt)

    for split in args.splits.split(','):
        d = torch.load(pjoin(data_dir, tmpl.format(split=split)), map_location='cpu', weights_only=False)
        sids = list(d.keys())
        out_list = []
        for i in range(0, len(sids), args.batch_size):
            chunk = sids[i:i + args.batch_size]
            caps = [(d[s].get('text', '') or '').strip() for s in chunk]
            frames = [int(d[s]['poses_3d'].shape[0]) for s in chunk]
            tok = torch.clamp(torch.tensor([f // 4 for f in frames]), min=1, max=max_tok).to(device).long()
            mids = mt.generate(caps, tok, timesteps=args.timesteps, cond_scale=args.cond_scale,
                               temperature=args.temperature)
            mids = rt.generate(mids, caps, tok, temperature=args.temperature, cond_scale=args.res_cond_scale)
            pred = vq.forward_decoder(mids)                       # (b, T, 534) or (b, 534, T)
            if pred.shape[-1] != 534:
                pred = pred.permute(0, 2, 1)
            pred = pred * std_t + mean_t
            for j, s in enumerate(chunk):
                L = int(tok[j].item()) * 4
                sign = pred[j, :L].cpu().float()
                out_list.append({'name': s, 'signer': '', 'gloss': d[s].get('gloss', ''),
                                 'text': d[s].get('text', ''), 'sign': sign})
        out_path = pjoin(out_dir, pkl_tmpl.format(split=split))
        with gzip.open(out_path, 'wb') as f:
            pickle.dump(out_list, f, protocol=4)
        Ts = [s['sign'].shape[0] for s in out_list]
        print(f'[OK] {split}: {len(out_list)} samples, frames {min(Ts)}/{np.mean(Ts):.1f}/{max(Ts)} -> {out_path}')


if __name__ == '__main__':
    main()
