"""Diagnostic: generate SLP poses with OLD T2M_GPT ckpt+algo, save in our pickle format.

Loads OLD MSR VQ + OLD t2m_trans + uses OLD greedy decoding (sample_6stream).
Outputs gzip pickle in `{split}.pickle` form so we can run new BT pipeline on it.

Purpose: isolate whether the v1→v2 gap is from BT pipeline or model.
"""
from __future__ import annotations
import sys, gzip, pickle, argparse
from pathlib import Path

# Make OLD T2M_GPT code importable
OLD_REPO = Path(r"D:/Graduate thesis/eggroll_v2/datasets_v3/T2M_GPT")
sys.path.insert(0, str(OLD_REPO))

import numpy as np
import torch
from tqdm import tqdm
from types import SimpleNamespace

from models.vqvae_multistream_residual import MultiStreamResidualVQVAE
from models.t2m_trans import Text2Motion_Transformer
from dataset.dataset_TM_sign_msr import SUB_ORDER

PHIX_DATA = Path(r"D:/Graduate thesis/eggroll_v2/datasets_v3/SLRTP-Sign-Production-Evaluation-Data/SLRTP-Sign-Production-Evaluation-Data/data")


def load_vq_msr(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = SimpleNamespace(**ck['args'])
    sub_codes = {
        'body_base': a.nb_base_body, 'body_res': a.nb_res_body,
        'hand_base': a.nb_base_hand, 'hand_res': a.nb_res_hand,
        'face_base': a.nb_base_face, 'face_res': a.nb_res_face,
    }
    sb = {'body': a.nb_base_body, 'hand': a.nb_base_hand, 'face': a.nb_base_face}
    sr = {'body': a.nb_res_body, 'hand': a.nb_res_hand, 'face': a.nb_res_face}
    m = MultiStreamResidualVQVAE(
        a, dataset_name='phix',
        code_dim=a.code_dim, output_emb_width=a.output_emb_width,
        down_t=a.down_t, stride_t=a.stride_t,
        width=a.width, depth=a.depth,
        dilation_growth_rate=a.dilation_growth_rate,
        activation=a.vq_act, norm=a.vq_norm,
        stream_codes_base=sb, stream_codes_residual=sr,
    )
    m.load_state_dict(ck['model']); m.to(device).eval()
    mean = np.load(Path(ckpt_path).parent / 'mean.npy')
    std = np.load(Path(ckpt_path).parent / 'std.npy')
    return m, mean, std, sub_codes


def load_trans(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = SimpleNamespace(**ck['args'])
    m = Text2Motion_Transformer(
        num_vq=a.num_vq, embed_dim=a.embed_dim, clip_dim=a.clip_dim,
        block_size=a.block_size, num_layers=a.num_layers, n_head=a.n_head,
        drop_out_rate=a.drop_out_rate, fc_rate=a.fc_rate,
    )
    m.load_state_dict(ck['model']); m.to(device).eval()
    return m, a


@torch.no_grad()
def encode_text(enc, ids, mask):
    out = enc(input_ids=ids, attention_mask=mask).last_hidden_state
    m = mask.unsqueeze(-1).float()
    return (out * m).sum(1) / m.sum(1).clamp(min=1)


@torch.no_grad()
def sample_6stream(trans, feat, max_len, num_vq, offsets, sub_codes):
    device = feat.device; end_id = num_vq; seq = []; n_sub = len(SUB_ORDER)
    for step in range(max_len):
        x = torch.tensor([seq], device=device, dtype=torch.long) if seq else []
        lg = trans(x if isinstance(x, torch.Tensor) else [], feat)
        last = lg[0, -1, :].clone()
        s_name = SUB_ORDER[step % n_sub]
        lo = offsets[s_name]; hi = lo + sub_codes[s_name]
        mask = torch.full_like(last, -float('inf'))
        mask[lo:hi] = 0.0; mask[end_id] = 0.0
        last = last + mask
        nxt = last.argmax(-1).item()
        if nxt == end_id:
            cut = (len(seq) // n_sub) * n_sub; seq = seq[:cut]; break
        seq.append(nxt)
    return seq


@torch.no_grad()
def decode_6stream(vq, flat_ids, offsets, sub_codes, mean, std, device):
    flat = np.asarray(flat_ids); n_sub = len(SUB_ORDER)
    T = len(flat) // n_sub * n_sub
    if T == 0:
        return np.zeros((4, vq.input_dim), dtype=np.float32)
    flat = flat[:T]
    tokens = {}
    for i, s_name in enumerate(SUB_ORDER):
        raw = flat[i::n_sub] - offsets[s_name]
        ids = np.clip(raw, 0, sub_codes[s_name] - 1)
        tokens[s_name] = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)
    recon = vq.forward_decoder(tokens)
    flat_out = recon[0].cpu().numpy() * std + mean   # (T, 534)
    return flat_out.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vq-ckpt', required=True)
    ap.add_argument('--trans-ckpt', required=True)
    ap.add_argument('--mbart-name', default='facebook/mbart-large-50')
    ap.add_argument('--lang', default='de_DE')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--splits', default='dev,test')
    ap.add_argument('--subsample', type=int, default=1, help='1=full rate, 2=::2')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    vq, mean, std, sub_codes = load_vq_msr(args.vq_ckpt, args.device)
    offsets = {}; off = 0
    for s in SUB_ORDER:
        offsets[s] = off; off += sub_codes[s]
    trans, tr_args = load_trans(args.trans_ckpt, args.device)
    num_vq = tr_args.num_vq; block_size = tr_args.block_size
    print(f'[*] num_vq={num_vq}, block_size={block_size}, sub_codes={sub_codes}')

    from transformers import MBart50TokenizerFast, MBartModel
    tok = MBart50TokenizerFast.from_pretrained(args.mbart_name, src_lang=args.lang)
    mbart = MBartModel.from_pretrained(args.mbart_name).encoder.to(args.device).eval()

    for split in args.splits.split(','):
        raw = torch.load(PHIX_DATA / f'{split}.pt', map_location='cpu', weights_only=False)
        sids = list(raw.keys())
        print(f'[*] {split}: {len(sids)} samples')

        out_list = []
        for sid in tqdm(sids, desc=f'gen-{split}'):
            s = raw[sid]
            txt = s.get('text', '')
            gloss = s.get('gloss', '')
            t = tok(txt, truncation=True, max_length=128, return_tensors='pt')
            feat = encode_text(mbart, t['input_ids'].to(args.device),
                                t['attention_mask'].to(args.device))
            toks = sample_6stream(trans, feat, max_len=block_size - 1,
                                     num_vq=num_vq, offsets=offsets, sub_codes=sub_codes)
            pose = decode_6stream(vq, toks, offsets, sub_codes, mean, std, args.device)
            if args.subsample > 1:
                pose = pose[::args.subsample]
            out_list.append({
                'name': sid, 'signer': '', 'gloss': gloss, 'text': txt,
                'sign': torch.from_numpy(pose).float(),
            })

        out_path = out_dir / f'{split}.pickle'
        with gzip.open(out_path, 'wb') as f:
            pickle.dump(out_list, f, protocol=4)
        Ts = [s['sign'].shape[0] for s in out_list]
        print(f'[OK] {split}: {len(out_list)} samples, frames {min(Ts)}/{np.mean(Ts):.1f}/{max(Ts)} → {out_path}')


if __name__ == '__main__':
    main()
