"""Trans 诊断闭环：TF vs Free-running, length, pose error。

诊断 4 个维度：
1. TF: 给真实前缀，下一个 token 的 CE/acc — 模型是否学会条件分布
2. Free-greedy: 自己滚（argmax）下的 token acc + length — exposure bias 程度
3. Free-sampled: 实际 inference 用的 top-k/T 采样下 token acc
4. Pose: GT 3D vs (a) GT-tokens VQ-decode (b) TF-pred VQ-decode (c) Free VQ-decode
   - MPJPE, velocity err, jerk, mean motion magnitude

输出表格：

| ckpt | TF CE | TF acc | Free-greedy acc | Free-sample acc | Len MAE | MPJPE_VQ | MPJPE_TF | MPJPE_free | Jerk_free | Vel_err_free |

Usage:
    python diagnose_trans.py \
        --variant base \
        --vq-ckpt checkpoints/csl_lift3d/vq_csl_lift3d_base_v2/best.pt \
        --trans-ckpt src/output_sign/trans_csl_lift3d_base_cross_gloss_v2/best.pt \
        --tokens-dir checkpoints/csl_lift3d/vq_csl_lift3d_base_v2/tokens \
        --split dev --limit 200
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from models.t2m_trans_cross import CrossAttnText2MotionTransformer

import os as _os
LIFT3D_DATA_DIR = Path(_os.environ.get('LIFT3D_DATA_DIR',
                                         str(Path(__file__).resolve().parents[2] / 'data' / 'csl')))


def _resolve_stats(ckpt_path):
    p = Path(ckpt_path); stem = p.stem
    for m in [p.parent / 'stats' / f'{stem}_mean.npy',
              p.parent / f'{stem}_mean.npy',
              p.parent / 'mean.npy']:
        s = m.with_name(m.name.replace('mean', 'std'))
        if m.exists() and s.exists():
            return np.load(m), np.load(s)
    raise FileNotFoundError(f'mean/std for {ckpt_path}')


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


def decode_tokens_base(vq, tokens, mean, std, device):
    """tokens (1, T) int -> (T_motion, 178, 3) numpy. Filters EOS/PAD tokens."""
    # Filter out EOS (=nb_code) and PAD (=nb_code+1)
    tokens = tokens.cpu().numpy().flatten()
    valid = tokens < vq.code_dim if hasattr(vq, 'code_dim') else None
    # Actually use the args to know nb_code
    # tokens passed in should already be valid; we'll clip
    pass


def decode_base(vq, tokens_1d, mean, std, device, nb_code):
    """Take a 1D token tensor, filter EOS/PAD, decode to (T, 178, 3)."""
    t = tokens_1d.detach().cpu().numpy().flatten()
    # Drop tokens >= nb_code (EOS, PAD)
    t = t[t < nb_code]
    if len(t) == 0:
        return np.zeros((4, 178, 3), dtype=np.float32)
    tt = torch.from_numpy(t).long().unsqueeze(0).to(device)
    with torch.no_grad():
        motion = vq.forward_decoder(tt)              # (1, T_out, 534)
    m_np = motion[0].cpu().numpy() * std + mean      # (T_out, 534)
    return m_np.reshape(-1, 178, 3).astype(np.float32)


def decode_ms(vq, tokens_1d, mean, std, device, av):
    """MS: split interleaved tokens (body, face, hand sorted) and decode."""
    from models.vqvae_multistream import KP_SPLITS
    splits = KP_SPLITS[av.dataname]
    stream_names = sorted(splits.keys(), key=lambda n: splits[n][0])
    n_streams = len(stream_names)
    nb_per_stream = av.nb_code
    flat = tokens_1d.detach().cpu().numpy().flatten()
    # Filter out invalid (EOS, PAD: >= n_streams*nb_per_stream)
    flat = flat[flat < n_streams * nb_per_stream]
    T = (len(flat) // n_streams) * n_streams
    if T == 0:
        return np.zeros((4, 178, 3), dtype=np.float32)
    flat = flat[:T]
    tokens_dict = {}
    for s_idx, name in enumerate(stream_names):
        raw = flat[s_idx::n_streams] - s_idx * nb_per_stream
        raw = np.clip(raw, 0, nb_per_stream - 1)
        tokens_dict[name] = torch.tensor(raw, device=device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        motion = vq.forward_decoder(tokens_dict)
    m_np = motion[0].cpu().numpy() * std + mean
    return m_np.reshape(-1, 178, 3).astype(np.float32)


def pose_metrics(p_pred, p_gt):
    """p_pred, p_gt: (T, 178, 3) numpy. Returns dict."""
    T = min(p_pred.shape[0], p_gt.shape[0])
    if T < 2:
        return {'mpjpe': float('nan'), 'vel_err': float('nan'),
                'jerk': float('nan'), 'mean_vel_gen': float('nan'),
                'mean_vel_gt': float('nan')}
    pp = p_pred[:T]; pg = p_gt[:T]
    # MPJPE
    diff = pp - pg                                     # (T, 178, 3)
    mpjpe = float(np.linalg.norm(diff, axis=-1).mean())
    # Velocity
    v_pred = np.diff(pp, axis=0)                       # (T-1, 178, 3)
    v_gt = np.diff(pg, axis=0)
    vel_err = float(np.linalg.norm(v_pred - v_gt, axis=-1).mean())
    mean_vel_gen = float(np.linalg.norm(v_pred, axis=-1).mean())
    mean_vel_gt = float(np.linalg.norm(v_gt, axis=-1).mean())
    # Jerk (3rd derivative of generated motion)
    if T >= 4:
        j_pred = np.diff(np.diff(np.diff(pp, axis=0), axis=0), axis=0)  # (T-3, 178, 3)
        jerk = float(np.linalg.norm(j_pred, axis=-1).mean())
    else:
        jerk = float('nan')
    return {'mpjpe': mpjpe, 'vel_err': vel_err, 'jerk': jerk,
            'mean_vel_gen': mean_vel_gen, 'mean_vel_gt': mean_vel_gt}


@torch.no_grad()
def free_generate(trans, mem, mem_mask, max_len, num_vq, device,
                   temperature=0.9, top_k=20, greedy=False,
                   rep_penalty=0.0, max_run=0, rep_streams=1,
                   min_len=None, max_len_eff=None):
    """Run trans.sample with given config. Greedy = top_k=1 + T=0."""
    if greedy:
        tokens = trans.sample(mem, mem_mask, max_len=max_len,
                                temperature=0.001, top_k=1, top_p=1.0,
                                rep_penalty=rep_penalty, max_run=max_run,
                                rep_streams=rep_streams,
                                min_len=min_len, max_len_eff=max_len_eff)
    else:
        tokens = trans.sample(mem, mem_mask, max_len=max_len,
                                temperature=temperature, top_k=top_k, top_p=1.0,
                                rep_penalty=rep_penalty, max_run=max_run,
                                rep_streams=rep_streams,
                                min_len=min_len, max_len_eff=max_len_eff)
    return tokens  # (1, T_gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='base', choices=['base', 'ms', 'msr'])
    ap.add_argument('--vq-ckpt', required=True)
    ap.add_argument('--trans-ckpt', required=True)
    ap.add_argument('--tokens-dir', required=True)
    ap.add_argument('--split', default='dev')
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--temperature', type=float, default=0.9)
    ap.add_argument('--top-k', type=int, default=20)
    ap.add_argument('--rep-penalty', type=float, default=0.0,
                     help='Subtract this from logits of recent tokens (window=8)')
    ap.add_argument('--max-run', type=int, default=0,
                     help='Forbid same token after N consecutive picks (0 = off)')
    ap.add_argument('--rep-streams', type=int, default=1,
                     help='>1: per-stream rep tracking (3 for MS, 6 for MSR)')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    print(f'[*] loading VQ ({args.variant}) from {args.vq_ckpt}')
    if args.variant == 'base':
        vq, mean, std, av = load_vq_base(args.vq_ckpt, args.device)
        nb_code = av.nb_code
    elif args.variant == 'ms':
        from models.vqvae_multistream import MultiStreamVQVAE, KP_SPLITS
        ck = torch.load(args.vq_ckpt, map_location='cpu', weights_only=False)
        av = SimpleNamespace(**ck['args'])
        vq = MultiStreamVQVAE(av, dataset_name=av.dataname,
                              nb_code=av.nb_code, code_dim=av.code_dim,
                              output_emb_width=av.output_emb_width,
                              down_t=av.down_t, stride_t=av.stride_t,
                              width=av.width, depth=av.depth,
                              dilation_growth_rate=av.dilation_growth_rate,
                              activation=av.vq_act, norm=av.vq_norm)
        vq.load_state_dict(ck['model']); vq.to(args.device).eval()
        mean, std = _resolve_stats(args.vq_ckpt)
        nb_code = av.nb_code * 3
    else:  # msr
        from models.vqvae_multistream_residual import MultiStreamResidualVQVAE
        ck = torch.load(args.vq_ckpt, map_location='cpu', weights_only=False)
        av = SimpleNamespace(**ck['args'])
        stream_codes_base = {'body': av.nb_base_body, 'hand': av.nb_base_hand, 'face': av.nb_base_face}
        stream_codes_res  = {'body': av.nb_res_body,  'hand': av.nb_res_hand,  'face': av.nb_res_face}
        vq = MultiStreamResidualVQVAE(
            av, dataset_name=av.dataname, code_dim=av.code_dim,
            output_emb_width=av.output_emb_width, down_t=av.down_t,
            stride_t=av.stride_t, width=av.width, depth=av.depth,
            dilation_growth_rate=av.dilation_growth_rate,
            activation=av.vq_act, norm=av.vq_norm,
            stream_codes_base=stream_codes_base,
            stream_codes_residual=stream_codes_res)
        vq.load_state_dict(ck['model']); vq.to(args.device).eval()
        mean, std = _resolve_stats(args.vq_ckpt)
        nb_code = sum(stream_codes_base.values()) + sum(stream_codes_res.values())
    print(f'    nb_code (effective vocab for trans) = {nb_code}')

    print(f'[*] loading trans from {args.trans_ckpt}')
    ckt = torch.load(args.trans_ckpt, map_location='cpu', weights_only=False)
    at = SimpleNamespace(**ckt['args'])
    trans = CrossAttnText2MotionTransformer(
        num_vq=at.num_vq, text_dim=at.text_dim, embed_dim=at.embed_dim,
        block_size=at.block_size, num_layers=at.num_layers,
        n_head=at.n_head, drop_out_rate=0.0, fc_rate=at.fc_rate,
        align_dim=getattr(at, 'align_dim', 0))
    trans.load_state_dict(ckt['model']); trans.to(args.device).eval()
    PAD_ID = at.num_vq + 1
    BOS_ID = at.num_vq  # in our setup BOS == EOS slot == num_vq

    # Text encoder
    text_enc_kind = getattr(at, 'text_encoder', 'mbart')
    if text_enc_kind == 'char':
        from models.text_encoder_char import CharTokenizer, CharTextEncoder
        _vp = getattr(at, 'char_vocab_path', None)
        if _vp is None or not Path(_vp).exists():
            _vp = str(Path(__file__).resolve().parents[2] / 'data' / 'csl' / 'char_vocab' / 'txt.vocab')
        vocab_path = _vp
        tok = CharTokenizer(vocab_path, src_lang='zh_CN')
        char_enc_dim = getattr(at, 'char_enc_dim', 0) or at.embed_dim
        text_enc = CharTextEncoder(
            vocab_size=len(tok), embed_dim=char_enc_dim,
            num_layers=getattr(at, 'char_enc_layers', 2),
            n_head=getattr(at, 'char_enc_heads', 8),
            drop_out_rate=0.0,
            max_len=getattr(at, 'max_text_len', 128) + 8,
            pad_id=tok.pad_id)
        text_enc.load_state_dict(ckt['text_enc'])
        text_enc.to(args.device).eval()
    else:
        from transformers import MBart50TokenizerFast, MBartModel
        tok = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50', src_lang='zh_CN')
        text_enc = MBartModel.from_pretrained('facebook/mbart-large-50').encoder.to(args.device).eval()

    # Load GT tokens cache and GT 3D
    cache = torch.load(Path(args.tokens_dir) / f'{args.split}_tokens.pt',
                         map_location='cpu', weights_only=False)
    gt3d = torch.load(LIFT3D_DATA_DIR / f'csl_daily_lift3d.{args.split}.pt',
                       map_location='cpu', weights_only=False)
    sids = list(cache.keys())
    if args.limit and args.limit < len(sids):
        sids = sids[:args.limit]
    print(f'[*] running diagnostic on {len(sids)} samples ({args.split})')

    max_motion_len = at.block_size

    # Accumulators
    tf_ce_sum, tf_correct, tf_total = 0.0, 0, 0
    free_greedy_correct_prefix, free_greedy_prefix_n = 0, 0
    free_sample_correct_prefix, free_sample_prefix_n = 0, 0
    len_abs_err_sum = 0.0; len_rel_err_sum = 0.0
    len_gt_sum = 0; len_gen_g_sum = 0; len_gen_s_sum = 0
    # Token distribution: per-sample, then average
    dist_acc = {k: 0.0 for k in [
        'gt_unique', 'gt_top1', 'gt_top5', 'gt_repeat',
        'fg_unique', 'fg_top1', 'fg_top5', 'fg_repeat',
        'fs_unique', 'fs_top1', 'fs_top5', 'fs_repeat']}
    dist_n = 0
    pose_acc = {k: 0.0 for k in
        ['mpjpe_vq', 'mpjpe_tf', 'mpjpe_free_g', 'mpjpe_free_s',
         'vel_err_vq', 'vel_err_tf', 'vel_err_free_g', 'vel_err_free_s',
         'jerk_free_g', 'jerk_free_s', 'jerk_gt',
         'vel_mag_free_g', 'vel_mag_free_s', 'vel_mag_gt']}
    pose_n = 0

    ce_loss = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction='sum')

    t0 = time.time()
    for sid in tqdm(sids, desc=args.split):
        ex = cache[sid]
        text = ex.get('text', '') or ''
        gt_tokens = torch.from_numpy(ex['tokens'].astype('int64')).to(args.device)  # (T_gt,)
        # Truncate to fit in trans block_size (need room for EOS)
        if gt_tokens.size(0) > max_motion_len - 1:
            gt_tokens = gt_tokens[:max_motion_len - 1]
        gt_tokens_eos = torch.cat([gt_tokens, torch.tensor([BOS_ID], device=args.device)])  # append EOS slot
        T_gt = gt_tokens_eos.size(0)

        # text encode
        t_enc = tok(text, truncation=True, max_length=128, return_tensors='pt')
        ids = t_enc['input_ids'].to(args.device)
        am = t_enc['attention_mask'].to(args.device)
        with torch.no_grad():
            if text_enc_kind == 'char':
                mem = text_enc(ids, am)
            else:
                mem = text_enc(input_ids=ids, attention_mask=am).last_hidden_state

        # === TF: input = [BOS, t0..t_{T-2}], target = [t0..t_{T-1}, EOS] ===
        # Our target sequence is gt_tokens + EOS at the end (= gt_tokens_eos)
        # Input is BOS + gt_tokens_eos[:-1]
        inp = torch.cat([torch.tensor([BOS_ID], device=args.device), gt_tokens_eos[:-1]]).unsqueeze(0)
        tgt = gt_tokens_eos.unsqueeze(0)
        with torch.no_grad():
            logits = trans(inp, mem, am)             # (1, T, V)
        # CE (sum over valid positions)
        ce_val = ce_loss(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)).item()
        nvalid = T_gt  # all positions are valid (no PAD in single sample)
        tf_ce_sum += ce_val
        tf_total += nvalid
        pred = logits.argmax(-1).squeeze(0)         # (T,)
        tf_correct += int((pred == tgt.squeeze(0)).sum().item())

        # TF-predicted tokens (used for VQ decode comparison)
        tf_pred_tokens = pred.detach()              # (T,) of length T_gt (which includes EOS)

        # === Free-greedy ===
        gen_g = free_generate(trans, mem, am, max_motion_len, at.num_vq,
                               args.device, greedy=True,
                               rep_penalty=args.rep_penalty, max_run=args.max_run,
                               rep_streams=args.rep_streams)
        gen_g_1d = gen_g.squeeze(0)
        T_g = gen_g_1d.size(0)

        # === Free-sampled ===
        gen_s = free_generate(trans, mem, am, max_motion_len, at.num_vq,
                               args.device, temperature=args.temperature,
                               top_k=args.top_k, greedy=False,
                               rep_penalty=args.rep_penalty, max_run=args.max_run,
                               rep_streams=args.rep_streams)
        gen_s_1d = gen_s.squeeze(0)
        T_s = gen_s_1d.size(0)

        # Prefix-aligned acc (position-wise match against gt_tokens up to min length)
        Tmin_g = min(T_g, gt_tokens.size(0))
        Tmin_s = min(T_s, gt_tokens.size(0))
        if Tmin_g > 0:
            free_greedy_correct_prefix += int((gen_g_1d[:Tmin_g] == gt_tokens[:Tmin_g]).sum().item())
            free_greedy_prefix_n += Tmin_g
        if Tmin_s > 0:
            free_sample_correct_prefix += int((gen_s_1d[:Tmin_s] == gt_tokens[:Tmin_s]).sum().item())
            free_sample_prefix_n += Tmin_s

        # === Token distribution stats ===
        def _dist_stats(t):
            """t: 1D long tensor. Returns (unique_ratio, top1, top5, repeat_ratio)."""
            if t.numel() == 0:
                return 0.0, 0.0, 0.0, 0.0
            arr = t.detach().cpu().numpy()
            n = len(arr)
            unique = len(set(arr.tolist()))
            from collections import Counter
            c = Counter(arr.tolist())
            top1 = c.most_common(1)[0][1] / n
            top5 = sum(v for _, v in c.most_common(5)) / n
            if n >= 2:
                rep = float((arr[1:] == arr[:-1]).mean())
            else:
                rep = 0.0
            return unique / n, top1, top5, rep

        u, t1, t5, r = _dist_stats(gt_tokens)
        dist_acc['gt_unique'] += u; dist_acc['gt_top1'] += t1
        dist_acc['gt_top5'] += t5; dist_acc['gt_repeat'] += r
        u, t1, t5, r = _dist_stats(gen_g_1d)
        dist_acc['fg_unique'] += u; dist_acc['fg_top1'] += t1
        dist_acc['fg_top5'] += t5; dist_acc['fg_repeat'] += r
        u, t1, t5, r = _dist_stats(gen_s_1d)
        dist_acc['fs_unique'] += u; dist_acc['fs_top1'] += t1
        dist_acc['fs_top5'] += t5; dist_acc['fs_repeat'] += r
        dist_n += 1

        # Length error (use greedy as canonical "model thinks")
        len_abs_err_sum += abs(T_g - gt_tokens.size(0))
        if gt_tokens.size(0) > 0:
            len_rel_err_sum += abs(T_g - gt_tokens.size(0)) / gt_tokens.size(0)
        len_gt_sum += gt_tokens.size(0)
        len_gen_g_sum += T_g
        len_gen_s_sum += T_s

        # === Pose decode 4 ways ===
        # (a) GT 3D
        v = gt3d.get(sid)
        if v is None:
            continue
        pose_gt = v['poses_3d']
        if torch.is_tensor(pose_gt): pose_gt = pose_gt.numpy()
        pose_gt = pose_gt.astype(np.float32)              # (T_gt_frames, 178, 3)

        # (b) GT tokens through VQ
        if args.variant == 'base':
            pose_vq = decode_base(vq, gt_tokens, mean, std, args.device, nb_code)
            pose_tf = decode_base(vq, tf_pred_tokens, mean, std, args.device, nb_code)
            pose_g  = decode_base(vq, gen_g_1d, mean, std, args.device, nb_code)
            pose_s  = decode_base(vq, gen_s_1d, mean, std, args.device, nb_code)
        elif args.variant == 'ms':
            pose_vq = decode_ms(vq, gt_tokens, mean, std, args.device, av)
            pose_tf = decode_ms(vq, tf_pred_tokens, mean, std, args.device, av)
            pose_g  = decode_ms(vq, gen_g_1d, mean, std, args.device, av)
            pose_s  = decode_ms(vq, gen_s_1d, mean, std, args.device, av)
        else:  # msr
            from dataset.dataset_TM_sign_msr import SUB_ORDER
            sub_codes = {'body_base': av.nb_base_body, 'body_res': av.nb_res_body,
                         'hand_base': av.nb_base_hand, 'hand_res': av.nb_res_hand,
                         'face_base': av.nb_base_face, 'face_res': av.nb_res_face}
            offsets = {}; off = 0
            for s in SUB_ORDER:
                offsets[s] = off; off += sub_codes[s]
            def _dec_msr(toks):
                t = toks.detach().cpu().numpy().flatten()
                t = t[t < nb_code]
                n_sub = len(SUB_ORDER)
                T = (len(t) // n_sub) * n_sub
                if T == 0: return np.zeros((4, 178, 3), dtype=np.float32)
                t = t[:T]
                td = {}
                for i, s in enumerate(SUB_ORDER):
                    raw = t[i::n_sub] - offsets[s]
                    raw = np.clip(raw, 0, sub_codes[s] - 1)
                    td[s] = torch.tensor(raw, device=args.device, dtype=torch.long).unsqueeze(0)
                with torch.no_grad():
                    motion = vq.forward_decoder(td)
                m_np = motion[0].cpu().numpy() * std + mean
                return m_np.reshape(-1, 178, 3).astype(np.float32)
            pose_vq = _dec_msr(gt_tokens)
            pose_tf = _dec_msr(tf_pred_tokens)
            pose_g  = _dec_msr(gen_g_1d)
            pose_s  = _dec_msr(gen_s_1d)

        m_vq = pose_metrics(pose_vq, pose_gt)
        m_tf = pose_metrics(pose_tf, pose_gt)
        m_g  = pose_metrics(pose_g,  pose_gt)
        m_s  = pose_metrics(pose_s,  pose_gt)
        # jerk on GT (reference)
        v_gt_diff = np.diff(pose_gt, axis=0)
        gt_mean_vel = float(np.linalg.norm(v_gt_diff, axis=-1).mean()) if v_gt_diff.size > 0 else 0
        if pose_gt.shape[0] >= 4:
            j_gt = np.diff(np.diff(np.diff(pose_gt, axis=0), axis=0), axis=0)
            gt_jerk = float(np.linalg.norm(j_gt, axis=-1).mean())
        else:
            gt_jerk = float('nan')

        pose_acc['mpjpe_vq']     += m_vq['mpjpe']
        pose_acc['mpjpe_tf']     += m_tf['mpjpe']
        pose_acc['mpjpe_free_g'] += m_g['mpjpe']
        pose_acc['mpjpe_free_s'] += m_s['mpjpe']
        pose_acc['vel_err_vq']     += m_vq['vel_err']
        pose_acc['vel_err_tf']     += m_tf['vel_err']
        pose_acc['vel_err_free_g'] += m_g['vel_err']
        pose_acc['vel_err_free_s'] += m_s['vel_err']
        pose_acc['jerk_free_g'] += m_g['jerk']
        pose_acc['jerk_free_s'] += m_s['jerk']
        pose_acc['jerk_gt']     += gt_jerk
        pose_acc['vel_mag_free_g'] += m_g['mean_vel_gen']
        pose_acc['vel_mag_free_s'] += m_s['mean_vel_gen']
        pose_acc['vel_mag_gt']     += gt_mean_vel
        pose_n += 1

    elapsed = time.time() - t0

    print('\n' + '=' * 80)
    print(f'Diagnostic summary — {len(sids)} samples, elapsed {elapsed:.0f}s')
    print('=' * 80)

    tf_ce = tf_ce_sum / max(tf_total, 1)
    tf_acc = tf_correct / max(tf_total, 1)
    fg_acc = free_greedy_correct_prefix / max(free_greedy_prefix_n, 1)
    fs_acc = free_sample_correct_prefix / max(free_sample_prefix_n, 1)
    len_mae = len_abs_err_sum / max(len(sids), 1)
    len_rel = len_rel_err_sum / max(len(sids), 1) * 100
    len_gt_mean = len_gt_sum / max(len(sids), 1)
    len_gen_g_mean = len_gen_g_sum / max(len(sids), 1)
    len_gen_s_mean = len_gen_s_sum / max(len(sids), 1)

    print(f'\n[TOKEN-LEVEL]')
    print(f'  TF CE:                  {tf_ce:.4f}')
    print(f'  TF acc:                 {tf_acc*100:.2f}%   (model gets next token right given GT prefix)')
    print(f'  Free-greedy acc:        {fg_acc*100:.2f}%   (AR argmax: prefix-aligned position match)')
    print(f'  Free-sample (T={args.temperature} k={args.top_k}) acc: {fs_acc*100:.2f}%')

    print(f'\n[LENGTH]')
    print(f'  GT mean length:         {len_gt_mean:.1f} tokens')
    print(f'  Free-greedy mean len:   {len_gen_g_mean:.1f} tokens')
    print(f'  Free-sample mean len:   {len_gen_s_mean:.1f} tokens')
    print(f'  Length MAE (vs greedy): {len_mae:.1f} tokens   ({len_rel:.1f}% relative)')

    print(f'\n[POSE — averaged over {pose_n} samples]')
    print(f'  MPJPE GT-tokens (VQ ceiling): {pose_acc["mpjpe_vq"]/pose_n:.5f}')
    print(f'  MPJPE TF-tokens:              {pose_acc["mpjpe_tf"]/pose_n:.5f}')
    print(f'  MPJPE Free-greedy:            {pose_acc["mpjpe_free_g"]/pose_n:.5f}')
    print(f'  MPJPE Free-sample:            {pose_acc["mpjpe_free_s"]/pose_n:.5f}')
    print(f'  Vel-err VQ-tokens:            {pose_acc["vel_err_vq"]/pose_n:.5f}')
    print(f'  Vel-err TF-tokens:            {pose_acc["vel_err_tf"]/pose_n:.5f}')
    print(f'  Vel-err Free-greedy:          {pose_acc["vel_err_free_g"]/pose_n:.5f}')
    print(f'  Vel-err Free-sample:          {pose_acc["vel_err_free_s"]/pose_n:.5f}')
    print(f'  Mean vel-mag GT:              {pose_acc["vel_mag_gt"]/pose_n:.5f}')
    print(f'  Mean vel-mag Free-greedy:     {pose_acc["vel_mag_free_g"]/pose_n:.5f}')
    print(f'  Mean vel-mag Free-sample:     {pose_acc["vel_mag_free_s"]/pose_n:.5f}')
    print(f'  Jerk GT:                      {pose_acc["jerk_gt"]/pose_n:.6f}')
    print(f'  Jerk Free-greedy:             {pose_acc["jerk_free_g"]/pose_n:.6f}')
    print(f'  Jerk Free-sample:             {pose_acc["jerk_free_s"]/pose_n:.6f}')

    print(f'\n[TOKEN DISTRIBUTION — averaged per sample]')
    print(f'  GT unique-ratio:        {dist_acc["gt_unique"]/dist_n*100:.1f}%')
    print(f'  Free-greedy unique:     {dist_acc["fg_unique"]/dist_n*100:.1f}%')
    print(f'  Free-sample unique:     {dist_acc["fs_unique"]/dist_n*100:.1f}%')
    print(f'  GT top-1 share:         {dist_acc["gt_top1"]/dist_n*100:.1f}%')
    print(f'  Free-greedy top-1:      {dist_acc["fg_top1"]/dist_n*100:.1f}%')
    print(f'  Free-sample top-1:      {dist_acc["fs_top1"]/dist_n*100:.1f}%')
    print(f'  GT top-5 share:         {dist_acc["gt_top5"]/dist_n*100:.1f}%')
    print(f'  Free-greedy top-5:      {dist_acc["fg_top5"]/dist_n*100:.1f}%')
    print(f'  Free-sample top-5:      {dist_acc["fs_top5"]/dist_n*100:.1f}%')
    print(f'  GT consecutive repeat:  {dist_acc["gt_repeat"]/dist_n*100:.1f}%')
    print(f'  Free-greedy repeat:     {dist_acc["fg_repeat"]/dist_n*100:.1f}%')
    print(f'  Free-sample repeat:     {dist_acc["fs_repeat"]/dist_n*100:.1f}%')

    print('\n[INTERPRETATION HINT]')
    if tf_acc > 0.30 and fg_acc < tf_acc * 0.5:
        print('  >> TF acc >> Free acc — EXPOSURE BIAS dominant. Fix: code corruption, scheduled sampling, masked-bidirectional.')
    elif tf_acc < 0.10:
        print('  >> TF acc itself low — text→token mapping not learned. Fix: better text encoder, alignment loss, more data.')
    else:
        print('  >> Mixed — investigate length and pose metrics for clues.')
    if len_rel > 30:
        print(f'  >> Length error {len_rel:.0f}% — duration not learned. Fix: duration head (T2S-GPT).')
    if pose_acc['jerk_free_s']/pose_n > 1.5 * pose_acc['jerk_gt']/pose_n:
        print(f'  >> Jerk-free-sample > 1.5× GT — motion is noisy / unstable.')


if __name__ == '__main__':
    main()
