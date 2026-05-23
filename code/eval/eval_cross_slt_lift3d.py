"""Cross-attention SLP → SLT BT eval generator for csl_lift3d.

Loads CrossAttnText2MotionTransformer + matching VQ-VAE, generates motion via
autoregressive cross-attention sampling, writes SLT-format gzip pickle for BT eval.

Supports baseline (single VQ), M1 (multi-stream interleaved), M2 (multi-stream + residual).
Variant is auto-detected from VQ checkpoint's args.

Usage:
    # Baseline
    python eval_cross_slt_lift3d.py --variant base \
        --vq-ckpt checkpoints/csl_lift3d/vq_csl_lift3d_base_v2/best.pt \
        --trans-ckpt checkpoints/csl_lift3d/trans_csl_lift3d_base_cross/best.pt \
        --splits dev,test \
        --out checkpoints/csl_lift3d/trans_csl_lift3d_base_cross/slt_eval

    # M1 (multi-stream)
    python eval_cross_slt_lift3d.py --variant ms \
        --vq-ckpt checkpoints/csl_lift3d/vq_csl_lift3d_ms_v2/best.pt \
        --trans-ckpt checkpoints/csl_lift3d/trans_csl_lift3d_ms_cross/best.pt ...

    # M2 (multi-stream residual)
    python eval_cross_slt_lift3d.py --variant msr ...
"""
from __future__ import annotations
import argparse, gzip, pickle, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.t2m_trans_cross import CrossAttnText2MotionTransformer

# Default: <repo>/data/csl (or override with env LIFT3D_DATA_DIR)
# Project root = parents[2] (eval/ -> code/ -> release/)
import os as _os
LIFT3D_DATA_DIR = Path(_os.environ.get('LIFT3D_DATA_DIR',
                                         str(Path(__file__).resolve().parents[2] / 'data' / 'csl')))


def _resolve_stats(ckpt_path):
    """Resolve mean.npy/std.npy paths. Tries:
       1. <stem>_mean.npy / <stem>_std.npy next to ckpt (release flat layout, in `stats/`)
       2. <ckpt_dir>/stats/<stem>_mean.npy
       3. <ckpt_dir>/mean.npy (legacy per-dir layout)
    """
    p = Path(ckpt_path)
    stem = p.stem  # e.g. 'vq_M1M2'
    candidates_mean = [
        p.parent / 'stats' / f'{stem}_mean.npy',
        p.parent / f'{stem}_mean.npy',
        p.parent / 'mean.npy',
    ]
    candidates_std = [c.with_name(c.name.replace('_mean', '_std').replace('mean', 'std')) for c in candidates_mean]
    for m, s in zip(candidates_mean, candidates_std):
        if m.exists() and s.exists():
            return np.load(m), np.load(s)
    raise FileNotFoundError(f"mean/std not found for {ckpt_path}. Tried: {[str(c) for c in candidates_mean]}")


def load_vq_base(ckpt_path, device):
    import models.vqvae as vqvae
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = SimpleNamespace(**ck['args'])
    m = vqvae.VQVAE_251(a, nb_code=a.nb_code, code_dim=a.code_dim,
                         output_emb_width=a.output_emb_width,
                         down_t=a.down_t, stride_t=a.stride_t,
                         width=a.width, depth=a.depth,
                         dilation_growth_rate=a.dilation_growth_rate,
                         activation=a.vq_act, norm=a.vq_norm)
    m.load_state_dict(ck['model']); m.to(device).eval()
    mean, std = _resolve_stats(ckpt_path)
    return m, mean, std, a


def load_vq_ms(ckpt_path, device):
    from models.vqvae_multistream import MultiStreamVQVAE, KP_SPLITS
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = SimpleNamespace(**ck['args'])
    # Support asymmetric per-stream codes if present in args
    stream_codes = None
    splits = KP_SPLITS[a.dataname]
    if any(getattr(a, f'nb_code_{n}', None) for n in splits):
        stream_codes = {n: (getattr(a, f'nb_code_{n}', None) or a.nb_code) for n in splits}
    m = MultiStreamVQVAE(a, dataset_name=a.dataname,
                          nb_code=a.nb_code, code_dim=a.code_dim,
                          stream_codes=stream_codes,
                          output_emb_width=a.output_emb_width,
                          down_t=a.down_t, stride_t=a.stride_t,
                          width=a.width, depth=a.depth,
                          dilation_growth_rate=a.dilation_growth_rate,
                          activation=a.vq_act, norm=a.vq_norm)
    m.load_state_dict(ck['model']); m.to(device).eval()
    mean, std = _resolve_stats(ckpt_path)
    return m, mean, std, a


def load_vq_rvq(ckpt_path, device):
    """Single-stream residual VQ (M2 only)."""
    from models.vqvae_residual import ResidualVQVAE
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = SimpleNamespace(**ck['args'])
    m = ResidualVQVAE(a, nb_code=a.nb_code, code_dim=a.code_dim,
                       output_emb_width=a.output_emb_width,
                       down_t=a.down_t, stride_t=a.stride_t, width=a.width,
                       depth=a.depth, dilation_growth_rate=a.dilation_growth_rate,
                       activation=a.vq_act, norm=a.vq_norm,
                       nb_code_residual=a.nb_code_residual)
    m.load_state_dict(ck['model']); m.to(device).eval()
    mean, std = _resolve_stats(ckpt_path)
    return m, mean, std, a


def load_vq_msr(ckpt_path, device):
    from models.vqvae_multistream_residual import MultiStreamResidualVQVAE
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = SimpleNamespace(**ck['args'])
    stream_codes_base = {'body': a.nb_base_body, 'hand': a.nb_base_hand, 'face': a.nb_base_face}
    stream_codes_res  = {'body': a.nb_res_body,  'hand': a.nb_res_hand,  'face': a.nb_res_face}
    m = MultiStreamResidualVQVAE(
        a, dataset_name=a.dataname,
        code_dim=a.code_dim, output_emb_width=a.output_emb_width,
        down_t=a.down_t, stride_t=a.stride_t, width=a.width,
        depth=a.depth, dilation_growth_rate=a.dilation_growth_rate,
        activation=a.vq_act, norm=a.vq_norm,
        stream_codes_base=stream_codes_base,
        stream_codes_residual=stream_codes_res)
    m.load_state_dict(ck['model']); m.to(device).eval()
    mean, std = _resolve_stats(ckpt_path)
    return m, mean, std, a


def get_stream_ranges_ms(av, max_len):
    """For multi-stream VQ: list of (lo, hi) per token position.
    Tokens are interleaved by stream in keypoint-start-index order (body, face, hand for csl_lift3d).
    Supports asymmetric per-stream codebooks: reads nb_code_<stream> from VQ args, falls back to nb_code.
    """
    from models.vqvae_multistream import KP_SPLITS
    splits = KP_SPLITS[av.dataname]
    stream_names = sorted(splits.keys(), key=lambda n: splits[n][0])
    n_streams = len(stream_names)
    per_stream = {n: getattr(av, f'nb_code_{n}', None) or av.nb_code for n in stream_names}
    offsets = {}
    cursor = 0
    for n in stream_names:
        offsets[n] = cursor
        cursor += per_stream[n]
    ranges = []
    for k in range(max_len):
        s = stream_names[k % n_streams]
        ranges.append((offsets[s], offsets[s] + per_stream[s]))
    return ranges, stream_names, per_stream


def get_stream_ranges_rvq(av, max_len):
    """For single-stream RVQ: 2 substreams [base, residual] interleaved.
    Offsets: base [0, nb_code); residual [nb_code, nb_code + nb_code_residual).
    """
    offsets = {'base': 0, 'res': av.nb_code}
    sub_codes = {'base': av.nb_code, 'res': av.nb_code_residual}
    ranges = []
    sub_order = ('base', 'res')
    for k in range(max_len):
        s = sub_order[k % 2]
        lo = offsets[s]
        hi = lo + sub_codes[s]
        ranges.append((lo, hi))
    return ranges, sub_order, sub_codes, offsets


def decode_motion_rvq(vq, flat, mean, std, device, av):
    """Single-stream RVQ: de-interleave flat → (base_idx, res_idx), call vq.forward_decoder."""
    flat = flat.cpu().numpy().flatten()
    T = (len(flat) // 2) * 2
    if T == 0:
        return np.zeros((4, vq.input_dim), dtype=np.float32)
    flat = flat[:T]
    base_raw = flat[0::2]                      # base tokens (even positions)
    res_raw  = flat[1::2] - av.nb_code         # residual tokens (odd) — remove offset
    base_raw = np.clip(base_raw, 0, av.nb_code - 1)
    res_raw  = np.clip(res_raw,  0, av.nb_code_residual - 1)
    base_idx = torch.tensor(base_raw, device=device, dtype=torch.long).unsqueeze(0)
    res_idx  = torch.tensor(res_raw,  device=device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        motion = vq.forward_decoder(base_idx, res_idx)
    return (motion[0].cpu().numpy() * std + mean).astype(np.float32)


def get_stream_ranges_msr(av, max_len):
    """For multi-stream + residual VQ: 6 substreams in SUB_ORDER."""
    from dataset.dataset_TM_sign_msr import SUB_ORDER
    sub_codes = {
        'body_base': av.nb_base_body, 'body_res': av.nb_res_body,
        'hand_base': av.nb_base_hand, 'hand_res': av.nb_res_hand,
        'face_base': av.nb_base_face, 'face_res': av.nb_res_face,
    }
    offsets = {}; off = 0
    for s in SUB_ORDER:
        offsets[s] = off; off += sub_codes[s]
    ranges = []
    for k in range(max_len):
        s_name = SUB_ORDER[k % len(SUB_ORDER)]
        lo = offsets[s_name]
        hi = lo + sub_codes[s_name]
        ranges.append((lo, hi))
    return ranges, SUB_ORDER, sub_codes, offsets


def decode_motion_base(vq, tokens, mean, std, device):
    """Single-stream baseline: just call forward_decoder."""
    if tokens.numel() == 0:
        return np.zeros((4, vq.input_dim), dtype=np.float32)
    motion = vq.forward_decoder(tokens.to(device))
    motion_np = motion[0].cpu().numpy() * std + mean
    return motion_np.astype(np.float32)


def decode_motion_ms(vq, flat, mean, std, device, av):
    """Multi-stream: split interleaved tokens by stream and decode. Supports asymmetric codes."""
    from models.vqvae_multistream import KP_SPLITS
    splits = KP_SPLITS[av.dataname]
    stream_names = sorted(splits.keys(), key=lambda n: splits[n][0])
    n_streams = len(stream_names)
    per_stream = {n: getattr(av, f'nb_code_{n}', None) or av.nb_code for n in stream_names}
    offsets = {}
    cursor = 0
    for n in stream_names:
        offsets[n] = cursor
        cursor += per_stream[n]

    flat = flat.cpu().numpy().flatten()
    T = (len(flat) // n_streams) * n_streams
    if T == 0:
        return np.zeros((4, vq.input_dim), dtype=np.float32)
    flat = flat[:T]
    tokens_dict = {}
    for s_idx, name in enumerate(stream_names):
        raw = flat[s_idx::n_streams] - offsets[name]
        raw = np.clip(raw, 0, per_stream[name] - 1)
        tokens_dict[name] = torch.tensor(raw, device=device, dtype=torch.long).unsqueeze(0)
    motion = vq.forward_decoder(tokens_dict)
    motion_np = motion[0].cpu().numpy() * std + mean
    return motion_np.astype(np.float32)


def decode_motion_msr(vq, flat, mean, std, device, sub_codes, offsets, sub_order):
    """Multi-stream + residual: split 6-substream interleaved tokens and decode."""
    flat = flat.cpu().numpy().flatten()
    n_sub = len(sub_order)
    T = (len(flat) // n_sub) * n_sub
    if T == 0:
        return np.zeros((4, vq.input_dim), dtype=np.float32)
    flat = flat[:T]
    tokens_dict = {}
    for i, s in enumerate(sub_order):
        raw = flat[i::n_sub] - offsets[s]
        raw = np.clip(raw, 0, sub_codes[s] - 1)
        tokens_dict[s] = torch.tensor(raw, device=device, dtype=torch.long).unsqueeze(0)
    motion = vq.forward_decoder(tokens_dict)
    motion_np = motion[0].cpu().numpy() * std + mean
    return motion_np.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['base', 'ms', 'msr', 'rvq'])
    ap.add_argument('--vq-ckpt', required=True)
    ap.add_argument('--trans-ckpt', required=True)
    ap.add_argument('--splits', default='dev,test')
    ap.add_argument('--out', required=True)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--top-p', type=float, default=1.0)
    ap.add_argument('--rep-penalty', type=float, default=0.0)
    ap.add_argument('--max-run', type=int, default=0)
    ap.add_argument('--rep-streams', type=int, default=1,
                     help='>1: per-stream rep tracking (3 for MS, 6 for MSR)')
    ap.add_argument('--min-len-frac', type=float, default=0.0,
                     help='if model has length_head: min_len = pred * (1 - margin)')
    ap.add_argument('--max-len-frac', type=float, default=0.0,
                     help='if model has length_head: max_len = pred * (1 + margin)')
    ap.add_argument('--max-len', type=int, default=None,
                     help='max motion-token sequence length. default = trans block_size - 1')
    ap.add_argument('--mbart-name', default='facebook/mbart-large-50')
    ap.add_argument('--lang', default='zh_CN')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--dataset', default='csl', choices=['csl', 'phix'],
                     help='csl→csl_daily_lift3d/csl_daily.{split}, phix→phix_lift3d/phoenix14t.{split}')
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[*] loading VQ ({args.variant})...')
    if args.variant == 'base':
        vq, mean, std, av = load_vq_base(args.vq_ckpt, args.device)
    elif args.variant == 'ms':
        vq, mean, std, av = load_vq_ms(args.vq_ckpt, args.device)
    elif args.variant == 'rvq':
        vq, mean, std, av = load_vq_rvq(args.vq_ckpt, args.device)
    else:
        vq, mean, std, av = load_vq_msr(args.vq_ckpt, args.device)

    print('[*] loading CrossAttn Trans...')
    ckt = torch.load(args.trans_ckpt, map_location='cpu', weights_only=False)
    at = SimpleNamespace(**ckt['args'])
    trans = CrossAttnText2MotionTransformer(
        num_vq=at.num_vq,
        text_dim=at.text_dim,
        embed_dim=at.embed_dim,
        block_size=at.block_size,
        num_layers=at.num_layers,
        n_head=at.n_head,
        drop_out_rate=0.0,
        fc_rate=at.fc_rate,
        align_dim=getattr(at, 'align_dim', 0),
    )
    trans.load_state_dict(ckt['model']); trans.to(args.device).eval()

    # Detect text encoder kind from trans ckpt args
    text_enc_kind = getattr(at, 'text_encoder', 'mbart')
    if text_enc_kind == 'char':
        print('[*] loading CharTextEncoder (from trans ckpt)...')
        from models.text_encoder_char import CharTokenizer, CharTextEncoder
        # Override: if ckpt's vocab path doesn't exist on this machine, fall back to release default.
        _vp = getattr(at, 'char_vocab_path', None)
        if _vp is None or not Path(_vp).exists():
            _vp = str(Path(__file__).resolve().parents[2] / 'data' / 'csl' / 'char_vocab' / 'txt.vocab')
        vocab_path = _vp
        tok = CharTokenizer(vocab_path, src_lang=args.lang)
        char_enc_dim = getattr(at, 'char_enc_dim', 0) or at.embed_dim
        text_encoder_module = CharTextEncoder(
            vocab_size=len(tok),
            embed_dim=char_enc_dim,
            num_layers=getattr(at, 'char_enc_layers', 2),
            n_head=getattr(at, 'char_enc_heads', 8),
            drop_out_rate=0.0,
            max_len=getattr(at, 'max_text_len', 128) + 8,
            pad_id=tok.pad_id,
        )
        text_encoder_module.load_state_dict(ckt['text_enc'])
        text_encoder_module.to(args.device).eval()
        print(f'[*] CharTextEncoder: vocab {len(tok)}, dim {char_enc_dim}, layers {getattr(at, "char_enc_layers", 2)}')
    else:
        print('[*] loading mBART encoder...')
        from transformers import MBart50TokenizerFast, MBartModel
        tok = MBart50TokenizerFast.from_pretrained(args.mbart_name, src_lang=args.lang)
        text_encoder_module = MBartModel.from_pretrained(args.mbart_name).encoder.to(args.device).eval()

    max_len = args.max_len if args.max_len else (at.block_size - 1)

    # Pre-compute stream_ranges for multi-stream variants
    stream_ranges = None
    sub_codes = offsets = sub_order = None
    if args.variant == 'ms':
        stream_ranges, stream_names, nb_per_stream = get_stream_ranges_ms(av, max_len)
        print(f'[*] MS streams: {stream_names}, per-stream codes: {nb_per_stream}')
    elif args.variant == 'rvq':
        stream_ranges, sub_order, sub_codes, offsets = get_stream_ranges_rvq(av, max_len)
        print(f'[*] RVQ substreams: {sub_order}, sub_codes: {sub_codes}, offsets: {offsets}')
    elif args.variant == 'msr':
        stream_ranges, sub_order, sub_codes, offsets = get_stream_ranges_msr(av, max_len)
        print(f'[*] MSR substreams: {sub_order}, sub_codes: {sub_codes}, offsets: {offsets}')

    if args.dataset == 'phix':
        _data_root = Path(_os.environ.get('LIFT3D_DATA_DIR',
                                            str(Path(__file__).resolve().parents[2] / 'data' / 'phix')))
        _src_template = '{root}/phix_lift3d.{split}.pt'
        _out_template = '{split}.pickle'
    else:
        _data_root = LIFT3D_DATA_DIR
        _src_template = '{root}/csl_daily_lift3d.{split}.pt'
        _out_template = 'csl_daily.{split}'

    for split in args.splits.split(','):
        src = Path(_src_template.format(root=_data_root, split=split))
        gt = torch.load(src, map_location='cpu', weights_only=False)
        sids = list(gt.keys())
        print(f'[*] generating {split}: {len(sids)} samples (T={args.temperature} k={args.top_k} max_len={max_len})')

        out_list = []; t0 = time.time()
        for sid in tqdm(sids):
            text = gt[sid].get('text', '')
            gloss = gt[sid].get('gloss', '')
            t_enc = tok(text, truncation=True, max_length=128, return_tensors='pt')
            with torch.no_grad():
                ids = t_enc['input_ids'].to(args.device)
                am = t_enc['attention_mask'].to(args.device)
                if text_enc_kind == 'char':
                    mem = text_encoder_module(ids, am)                              # (1, T_text, char_dim)
                else:
                    mem = text_encoder_module(input_ids=ids, attention_mask=am).last_hidden_state

                # Length-window (only if model has length head + flags set)
                min_len_v = None; max_len_eff_v = None
                if (args.min_len_frac > 0 or args.max_len_frac > 0) and \
                   getattr(trans, 'predict_length', False) and trans.length_head is not None:
                    text_mem_proj = trans.text_norm(trans.text_proj(mem))
                    kpm = (am == 0)
                    log_pred = trans.predict_motion_length(text_mem_proj, kpm)
                    pred_len = float(torch.exp(log_pred).item())
                    if args.min_len_frac > 0:
                        min_len_v = int(pred_len * (1 - args.min_len_frac))
                    if args.max_len_frac > 0:
                        max_len_eff_v = int(pred_len * (1 + args.max_len_frac))
                tokens = trans.sample(
                    mem, am,
                    max_len=max_len,
                    temperature=args.temperature,
                    top_k=args.top_k, top_p=args.top_p,
                    stream_ranges=stream_ranges,
                    rep_penalty=args.rep_penalty,
                    max_run=args.max_run,
                    rep_streams=args.rep_streams,
                    min_len=min_len_v,
                    max_len_eff=max_len_eff_v,
                )                                                                  # (1, T_out)

                if args.variant == 'base':
                    motion = decode_motion_base(vq, tokens, mean, std, args.device)
                elif args.variant == 'ms':
                    motion = decode_motion_ms(vq, tokens, mean, std, args.device, av)
                elif args.variant == 'rvq':
                    motion = decode_motion_rvq(vq, tokens, mean, std, args.device, av)
                else:
                    motion = decode_motion_msr(vq, tokens, mean, std, args.device,
                                                 sub_codes, offsets, sub_order)

            out_list.append({
                'name': sid, 'signer': '', 'gloss': gloss, 'text': text,
                'sign': torch.from_numpy(motion).float(),
            })

        elapsed = time.time() - t0
        out_path = out_dir / _out_template.format(split=split)
        with gzip.open(out_path, 'wb') as f:
            pickle.dump(out_list, f, protocol=4)
        Ts = [s['sign'].shape[0] for s in out_list]
        print(f'[OK] {split}: {len(out_list)} samples, '
              f'gen_frames min/mean/max = {min(Ts)}/{np.mean(Ts):.1f}/{max(Ts)}, '
              f'elapsed {elapsed:.0f}s → {out_path}')


if __name__ == '__main__':
    main()
