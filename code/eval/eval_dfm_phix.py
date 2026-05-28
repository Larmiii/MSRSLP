"""SLP generation with DFM model + same MSR VQ decoder + SLRTP-format pickle output.

Drop-in replacement for eval_cross_slt_lift3d.py (AR variant) for PHIX M1+M2 with DFM.
Reuses the existing VQ decoders and stream interleaving logic.

Usage:
  python eval_dfm_phix.py \
    --vq-ckpt checkpoints/phix/vq/vq_M1M2.pt \
    --dfm-ckpt output_sign/dfm_phix_M1M2/best.pt \
    --splits dev,test \
    --out results/phix_dfm_M1M2 \
    --n-steps 24 --cfg-scale 2.0
"""
from __future__ import annotations
import argparse, gzip, pickle, sys, time
import os as _os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.t2m_dfm_cross import CrossAttnDFM

# Reuse decoders from existing AR eval
from eval_cross_slt_lift3d import (
    load_vq_msr, decode_motion_msr, get_stream_ranges_msr,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vq-ckpt', required=True)
    ap.add_argument('--dfm-ckpt', required=True)
    ap.add_argument('--splits', default='dev,test')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-steps', type=int, default=24, help='DFM Euler sampling steps')
    ap.add_argument('--cfg-scale', type=float, default=2.0,
                     help='Classifier-free guidance scale (1.0 = off)')
    ap.add_argument('--temperature', type=float, default=1.0,
                     help='Softmax temperature for sampling')
    ap.add_argument('--len-mult', type=float, default=1.0,
                     help='Multiplier on predicted length (>1 = longer, <1 = shorter)')
    ap.add_argument('--mbart-name', default='facebook/mbart-large-50')
    ap.add_argument('--lang', default='de_DE')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print('[*] loading MSR VQ...')
    vq, mean, std, av = load_vq_msr(args.vq_ckpt, args.device)
    print(f'[*] VQ sub_codes: body=({av.nb_base_body}+{av.nb_res_body}), '
          f'hand=({av.nb_base_hand}+{av.nb_res_hand}), '
          f'face=({av.nb_base_face}+{av.nb_res_face})')

    print('[*] loading DFM model...')
    ckt = torch.load(args.dfm_ckpt, map_location='cpu', weights_only=False)
    at = SimpleNamespace(**ckt['args'])
    print(f'[*] DFM ckpt args: num_vq={at.num_vq} embed={at.embed_dim} '
          f'block_size={at.block_size} layers={at.num_layers}')
    model = CrossAttnDFM(
        num_vq=at.num_vq, text_dim=at.text_dim, embed_dim=at.embed_dim,
        block_size=at.block_size, num_layers=at.num_layers, n_head=at.n_head,
        drop_out_rate=0.0, fc_rate=at.fc_rate,
        predict_length=bool(getattr(at, 'predict_length', 1)),
    )
    model.load_state_dict(ckt['model'])
    model.to(args.device).eval()

    print('[*] loading mBART encoder...')
    from transformers import MBart50TokenizerFast, MBartModel
    tok = MBart50TokenizerFast.from_pretrained(args.mbart_name, src_lang=args.lang)
    text_enc = MBartModel.from_pretrained(args.mbart_name).encoder.to(args.device).eval()

    # Null memory (empty text) for CFG
    null_input = tok([""], return_tensors='pt', padding=True, truncation=True, max_length=80)
    null_ids = null_input['input_ids'].to(args.device)
    null_am  = null_input['attention_mask'].to(args.device)
    with torch.no_grad():
        null_mem = text_enc(input_ids=null_ids, attention_mask=null_am).last_hidden_state

    max_len = at.block_size
    stream_ranges, sub_order, sub_codes, offsets = get_stream_ranges_msr(av, max_len)
    n_sub = len(sub_order)
    print(f'[*] MSR substreams: {sub_order}, sub_codes: {sub_codes}, '
          f'offsets: {offsets}, n_sub={n_sub}')

    # PHIX data
    _data_root = Path(_os.environ.get('LIFT3D_DATA_DIR',
                                        str(Path(__file__).resolve().parents[2] / 'data' / 'phix')))

    for split in args.splits.split(','):
        src = _data_root / f'phix_lift3d.{split}.pt'
        gt = torch.load(src, map_location='cpu', weights_only=False)
        sids = list(gt.keys())
        print(f'[*] generating {split}: {len(sids)} samples '
              f'(n_steps={args.n_steps} cfg={args.cfg_scale} len_mult={args.len_mult})')

        out_list = []; t0 = time.time()
        for sid in tqdm(sids):
            text = gt[sid].get('text', '')
            gloss = gt[sid].get('gloss', '')
            t_enc = tok(text, truncation=True, max_length=80, return_tensors='pt')
            with torch.no_grad():
                ids = t_enc['input_ids'].to(args.device)
                am  = t_enc['attention_mask'].to(args.device)
                mem = text_enc(input_ids=ids, attention_mask=am).last_hidden_state

                # Predict length from text (mandatory for DFM)
                mem_proj = model.text_norm(model.text_proj(mem))
                kpm = (am == 0)
                log_pred = model.predict_motion_length(mem_proj, kpm)
                pred_len = float(torch.exp(log_pred).item()) * args.len_mult
                # Round to multiple of n_sub (interleaved tokens)
                gen_len = max(n_sub, int(round(pred_len / n_sub)) * n_sub)
                gen_len = min(gen_len, max_len)

                # Sample (B=1)
                # Pad null_mem to match mem's T_text
                T_text = mem.size(1)
                T_null = null_mem.size(1)
                if T_null < T_text:
                    pad_mem = torch.zeros(1, T_text - T_null, null_mem.size(-1), device=args.device)
                    pad_am  = torch.zeros(1, T_text - T_null, dtype=null_am.dtype, device=args.device)
                    _null_mem = torch.cat([null_mem, pad_mem], dim=1)
                    _null_am  = torch.cat([null_am, pad_am], dim=1)
                else:
                    _null_mem = null_mem[:, :T_text]
                    _null_am  = null_am[:, :T_text]

                lengths = torch.tensor([gen_len], device=args.device)
                tokens = model.sample(
                    mbart_last_hidden=mem,
                    text_attn_mask=am,
                    lengths=lengths,
                    n_steps=args.n_steps,
                    cfg_scale=args.cfg_scale,
                    null_mbart=_null_mem,
                    null_text_attn_mask=_null_am,
                    temperature=args.temperature,
                )                                                                  # (1, T_out)
                # Trim to valid length (drop MASK padding) before VQ decode
                tokens = tokens[:, :gen_len]

                motion = decode_motion_msr(vq, tokens, mean, std, args.device,
                                            sub_codes, offsets, sub_order)

            out_list.append({
                'name': sid, 'signer': '', 'gloss': gloss, 'text': text,
                'sign': torch.from_numpy(motion).float(),
            })

        elapsed = time.time() - t0
        out_path = out_dir / f'{split}.pickle'
        with gzip.open(out_path, 'wb') as f:
            pickle.dump(out_list, f, protocol=4)
        Ts = [s['sign'].shape[0] for s in out_list]
        print(f'[OK] {split}: {len(out_list)} samples, '
              f'gen_frames min/mean/max = {min(Ts)}/{np.mean(Ts):.1f}/{max(Ts)}, '
              f'elapsed {elapsed:.0f}s → {out_path}')


if __name__ == '__main__':
    main()
