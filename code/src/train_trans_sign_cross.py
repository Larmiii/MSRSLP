"""Stage 2 training for cross-attention text-to-motion Transformer.

Unified script — handles baseline (single VQ), M1 (multi-stream), M2 (multi-stream
residual), and M2+M3 (M2 + InfoNCE alignment). The model `CrossAttnText2MotionTransformer`
is variant-agnostic; the only difference is `--num-vq` and (optionally) `--lambda-align`.

Replaces the legacy `train_trans_sign.py` / `train_trans_sign_msr_align.py` etc., which
all used pooled mBART feature as a single condition vector. Here the full mBART encoder
sequence is used as cross-attn memory.

Usage:
    # Baseline (single VQ, e.g. csl_lift3d base v2 with num_vq=4096)
    python train_trans_sign_cross.py --dataname csl_lift3d \
        --tokens-dir checkpoints/csl_lift3d/vq_csl_lift3d_base_v2/tokens \
        --vq-ckpt checkpoints/csl_lift3d/vq_csl_lift3d_base_v2/best.pt \
        --exp-name trans_csl_lift3d_base_cross \
        --num-vq 4096 --block-size 160 \
        --num-layers 6 --embed-dim 512 \
        --batch-size 16 --total-iter 30000 \
        --lr 1e-4 --lr-scheduler 15000 25000 --gamma 0.2

    # M2 (multi-stream + residual, num_vq = 896)
    python train_trans_sign_cross.py --dataname csl_lift3d \
        --tokens-dir checkpoints/csl_lift3d/vq_csl_lift3d_msr/tokens \
        --vq-ckpt checkpoints/csl_lift3d/vq_csl_lift3d_msr/best.pt \
        --exp-name trans_csl_lift3d_msr_cross \
        --num-vq 896 --block-size 320 \
        --num-layers 6 --embed-dim 512 \
        --batch-size 16 --total-iter 30000

    # M2+M3 (with alignment)
    python train_trans_sign_cross.py ... --align-dim 256 --lambda-align 0.1
"""
from __future__ import annotations
import argparse, json, os, sys, time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset.dataset_TM_sign import TMSignDataset, collate_tm_sign
from models.t2m_trans_cross import CrossAttnText2MotionTransformer

LANG_DEFAULTS = {'phix': 'de_DE', 'csl': 'zh_CN', 'phix14t': 'de_DE', 'csl_lift3d': 'zh_CN', 'phix_lift3d': 'de_DE'}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataname', choices=['phix', 'csl', 'phix14t', 'csl_lift3d', 'phix_lift3d'], required=True)
    p.add_argument('--tokens-dir', required=True)
    p.add_argument('--vq-ckpt', required=True, help='only used for sanity check')
    p.add_argument('--exp-name', default='trans_cross_debug')
    p.add_argument('--out-dir', default='output_sign/')
    p.add_argument('--mbart-name', default='facebook/mbart-large-50')
    p.add_argument('--lang', default=None)
    p.add_argument('--text-encoder', choices=['mbart', 'char'], default='mbart',
                    help='mbart: frozen mBART-50 (default). '
                          'char: train a small custom char-level Transformer encoder from scratch '
                          '(inspired by Walsh et al. FG 2024 — domain-specific text features).')
    p.add_argument('--char-vocab-path', default=None,
                    help='Char vocab file (one token per line, first 4 = <unk>,<pad>,<s>,</s>). '
                          'Required if --text-encoder=char. Default: CSL-Daily SLT char vocab.')
    p.add_argument('--char-enc-layers', type=int, default=2)
    p.add_argument('--char-enc-dim', type=int, default=0,
                    help='Char encoder hidden dim. 0 = use embed_dim from trans.')
    p.add_argument('--char-enc-heads', type=int, default=8)
    p.add_argument('--char-enc-dropout', type=float, default=0.1)

    p.add_argument('--num-vq', type=int, required=True,
                    help='size of motion VQ vocabulary (excluding BOS/EOS)')
    p.add_argument('--embed-dim', type=int, default=512)
    p.add_argument('--text-dim', type=int, default=1024,
                    help='mBART encoder hidden dim')
    p.add_argument('--block-size', type=int, default=320)
    p.add_argument('--num-layers', type=int, default=6)
    p.add_argument('--n-head', type=int, default=8)
    p.add_argument('--drop-out-rate', type=float, default=0.1)
    p.add_argument('--fc-rate', type=int, default=4)
    p.add_argument('--align-dim', type=int, default=0,
                    help='InfoNCE align head dim. 0 = no align loss (i.e. no M3)')
    p.add_argument('--lambda-align', type=float, default=0.1)

    # Length head: predicts total motion-token length from text mem (pooled),
    # used as an auxiliary loss and at inference time to constrain EOS.
    p.add_argument('--predict-length', type=int, default=0,
                    help='If 1: add length head (pool text -> MLP -> log-length scalar) '
                         'and add SmoothL1 loss against log(gt_len).')
    p.add_argument('--lambda-length', type=float, default=0.2)

    # Gloss-guided text encoder supervision (part of the strong baseline, NOT a M-module)
    p.add_argument('--gloss-supervised', type=int, default=0,
                    help='If 1: add a small gloss decoder on top of text_memory and train it '
                          'with CE loss. Forces text encoder to encode sign-language semantics. '
                          'Used only at training; not invoked at inference.')
    p.add_argument('--gloss-vocab-path', default=None,
                    help='Path to gls.vocab. Required if --gloss-supervised=1. Default: CSL gloss vocab.')
    p.add_argument('--gloss-dec-layers', type=int, default=2)
    p.add_argument('--gloss-dec-dim', type=int, default=256)
    p.add_argument('--gloss-dec-heads', type=int, default=4)
    p.add_argument('--gloss-max-len', type=int, default=48)
    p.add_argument('--lambda-gloss', type=float, default=0.5)

    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--total-iter', type=int, default=30000)
    p.add_argument('--warm-up-iter', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr-scheduler', type=int, nargs='+', default=[15000, 25000])
    p.add_argument('--gamma', type=float, default=0.2)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--max-text-len', type=int, default=128)
    p.add_argument('--freeze-text', type=int, default=1)
    p.add_argument('--label-smoothing', type=float, default=0.0)
    p.add_argument('--motion-token-mask-prob', type=float, default=0.0,
                    help='Probability to replace each input motion token with a random valid token. '
                          'BERT-style regularization, forces model to attend to text and prevents '
                          'memorization of full motion sequences. 0 = off (default).')

    p.add_argument('--print-iter', type=int, default=200)
    p.add_argument('--eval-iter', type=int, default=1000)
    p.add_argument('--save-iter', type=int, default=5000)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    p.add_argument('--early-stop-patience', type=int, default=8)
    p.add_argument('--early-stop-min-delta', type=float, default=0.001)
    p.add_argument('--min-iter-before-early-stop', type=int, default=3000)
    p.add_argument('--max-no-improve-iter', type=int, default=0,
                    help='FIX C: hard cap on (it - best_iter). Triggers stop regardless of '
                          'lr resets when iters_since_best > this. 0 = auto = 3×patience×eval_iter.')

    return p.parse_args()


def warmup_lr(opt, it, warm_up_iter, max_lr):
    lr = max_lr * (it + 1) / (warm_up_iter + 1)
    for g in opt.param_groups:
        g['lr'] = lr
    return lr


def encode_text_full(text_enc, input_ids, attn_mask, kind='mbart'):
    """Return text encoder output sequence + attention_mask.

    Supports both mBART (HuggingFace API) and custom CharTextEncoder.

    Returns:
      last_hidden:  (B, T_text, text_dim)
      attn_mask:    (B, T_text) — 1=valid, 0=pad (returned as-is for downstream)
    """
    if kind == 'mbart':
        out = text_enc(input_ids=input_ids, attention_mask=attn_mask).last_hidden_state
    else:  # char (CharTextEncoder)
        out = text_enc(input_ids, attn_mask)
    return out, attn_mask


def prepend_bos(motion_tokens, bos_id, pad_id):
    """Prepend BOS token to motion_tokens, drop last (preserve length).

    motion_tokens: (B, T)  — t0..t_{T-2}, EOS
    Returns:
      input_tokens:  (B, T) — BOS, t0, ..., t_{T-2}   (last EOS dropped, BOS prepended)
      target_tokens: (B, T) — t0, t1, ..., t_{T-2}, EOS   (= motion_tokens)
    """
    B, T = motion_tokens.shape
    bos = torch.full((B, 1), bos_id, device=motion_tokens.device, dtype=motion_tokens.dtype)
    inp = torch.cat([bos, motion_tokens[:, :-1]], dim=1)
    return inp, motion_tokens


@torch.no_grad()
def eval_dev_loss(model, text_enc, loader, device, ce, num_vq, args, text_enc_kind='mbart',
                   gloss_dec=None, ce_gloss=None):
    model.eval()
    if text_enc_kind != 'mbart':
        text_enc.eval()
    if gloss_dec is not None:
        gloss_dec.eval()
    total_ce = 0.0
    total_align = 0.0
    total_gloss = 0.0
    total_gloss_correct = 0
    total_gloss_count = 0
    n = 0
    for batch in loader:
        ids = batch['input_ids'].to(device)
        am = batch['attention_mask'].to(device)
        mt = batch['motion_tokens'].to(device)
        mem, mem_mask = encode_text_full(text_enc, ids, am, kind=text_enc_kind)

        inp, tgt = prepend_bos(mt, bos_id=num_vq, pad_id=num_vq + 1)
        if args.align_dim > 0:
            logits, mot_feat, text_mem = model.forward_with_features(inp, mem, mem_mask)
        else:
            logits = model(inp, mem, mem_mask)
        loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

        bs = mt.size(0); n += bs
        total_ce += loss.item() * bs

        if args.align_dim > 0:
            align_loss = compute_align_loss(model, mot_feat, text_mem, mem_mask, mt, num_vq)
            total_align += align_loss.item() * bs

        if gloss_dec is not None and 'gloss_input_ids' in batch:
            g_ids = batch['gloss_input_ids'].to(device)
            g_inp = g_ids[:, :-1]
            g_tgt = g_ids[:, 1:]
            g_logits = gloss_dec(g_inp, mem, mem_mask)
            g_loss = ce_gloss(g_logits.reshape(-1, g_logits.size(-1)), g_tgt.reshape(-1))
            total_gloss += g_loss.item() * bs
            # token-level accuracy (ignore pad)
            pred = g_logits.argmax(-1)
            mask = (g_tgt != ce_gloss.ignore_index)
            total_gloss_correct += ((pred == g_tgt) & mask).sum().item()
            total_gloss_count += mask.sum().item()
    model.train()
    out = (total_ce / max(n, 1), total_align / max(n, 1))
    if gloss_dec is not None:
        gloss_acc = total_gloss_correct / max(total_gloss_count, 1)
        out = out + (total_gloss / max(n, 1), gloss_acc)
    return out


def compute_align_loss(model, mot_feat, text_mem, mem_mask, motion_tokens, num_vq):
    """InfoNCE alignment loss: pool motion and text features, contrastive on batch.

    Args:
      mot_feat:        (B, T_m, embed_dim) — decoder hidden (pre-LM head)
      text_mem:        (B, T_text, embed_dim) — already projected mBART memory
      mem_mask:        (B, T_text)  — 1=valid, 0=pad
      motion_tokens:   (B, T_m)  — used to mask out PAD positions in motion pool
      num_vq:          motion vocab size (PAD = num_vq+1)
    """
    # Mean-pool motion (mask out PAD positions which are pad_id = num_vq+1)
    mot_mask = (motion_tokens != num_vq + 1).float().unsqueeze(-1)
    mot_pool = (mot_feat * mot_mask).sum(1) / mot_mask.sum(1).clamp(min=1)
    # Mean-pool text (mask out padding)
    txt_mask = mem_mask.float().unsqueeze(-1)
    txt_pool = (text_mem * txt_mask).sum(1) / txt_mask.sum(1).clamp(min=1)
    # Project to align space
    mot_z = F.normalize(model.motion_align(mot_pool), dim=-1)
    txt_z = F.normalize(model.text_align(txt_pool), dim=-1)
    # InfoNCE: symmetric contrastive
    temp = 0.1
    logits_m2t = mot_z @ txt_z.t() / temp
    logits_t2m = logits_m2t.t()
    B = mot_z.size(0)
    labels = torch.arange(B, device=mot_z.device)
    loss = (F.cross_entropy(logits_m2t, labels) + F.cross_entropy(logits_t2m, labels)) / 2
    return loss


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.lang is None:
        args.lang = LANG_DEFAULTS[args.dataname]

    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'train.log'
    log_f = open(log_path, 'w', encoding='utf-8')
    def log(m): print(m); log_f.write(m + '\n'); log_f.flush()
    log(f'[*] args: {vars(args)}')

    # ---- Text encoder (mBART OR custom char) ----
    if args.text_encoder == 'mbart':
        log('[*] loading mBART-50 tokenizer + encoder...')
        from transformers import MBart50TokenizerFast, MBartModel
        tokenizer = MBart50TokenizerFast.from_pretrained(args.mbart_name, src_lang=args.lang)
        mbart_full = MBartModel.from_pretrained(args.mbart_name)
        mbart_enc = mbart_full.encoder.to(args.device)
        if args.freeze_text:
            for p in mbart_enc.parameters():
                p.requires_grad = False
            mbart_enc.eval()
            log('[*] mBART encoder FROZEN')
        assert mbart_full.config.hidden_size == args.text_dim, \
            f"text_dim={args.text_dim} but mBART hidden={mbart_full.config.hidden_size}"
        text_enc = mbart_enc
        text_enc_trainable = (not args.freeze_text)
    else:  # char
        log('[*] building char-level text encoder from scratch (Walsh-style)')
        from models.text_encoder_char import CharTokenizer, CharTextEncoder
        if args.char_vocab_path is None:
            # Default: <repo>/data/<dataset>/char_vocab/txt.vocab (project-relative)
            # Override with --char-vocab-path or env CHAR_VOCAB_PATH
            _root = Path(__file__).resolve().parents[2]
            _ds = 'csl' if 'csl' in args.dataname else ('phix' if 'phix' in args.dataname else args.dataname)
            args.char_vocab_path = str(_root / 'data' / _ds / 'char_vocab' / 'txt.vocab')
            import os as _os
            _env = _os.environ.get('CHAR_VOCAB_PATH')
            if _env: args.char_vocab_path = _env
        tokenizer = CharTokenizer(args.char_vocab_path, src_lang=args.lang)
        log(f'[*] CharTokenizer: vocab_size={len(tokenizer)} from {args.char_vocab_path}')
        char_enc_dim = args.char_enc_dim if args.char_enc_dim > 0 else args.embed_dim
        # Override text_dim to match char encoder output (no projection bloat)
        args.text_dim = char_enc_dim
        char_text_enc = CharTextEncoder(
            vocab_size=len(tokenizer),
            embed_dim=char_enc_dim,
            num_layers=args.char_enc_layers,
            n_head=args.char_enc_heads,
            drop_out_rate=args.char_enc_dropout,
            max_len=args.max_text_len + 8,
            pad_id=tokenizer.pad_id,
        ).to(args.device)
        n_params = sum(p.numel() for p in char_text_enc.parameters())
        log(f'[*] CharTextEncoder: {n_params/1e6:.2f}M params, '
            f'{args.char_enc_layers}L embed {char_enc_dim} heads {args.char_enc_heads}')
        text_enc = char_text_enc
        text_enc_trainable = True

    # ---- Optional gloss tokenizer + decoder (aux supervision for text encoder) ----
    gloss_tokenizer = None
    gloss_dec = None
    if args.gloss_supervised:
        from models.text_encoder_char import GlossTokenizer
        from models.gloss_decoder import GlossDecoder
        if args.gloss_vocab_path is None:
            _root = Path(__file__).resolve().parents[2]
            _ds = 'csl' if 'csl' in args.dataname else ('phix' if 'phix' in args.dataname else args.dataname)
            args.gloss_vocab_path = str(_root / 'data' / _ds / 'char_vocab' / 'gls.vocab')
            import os as _os
            _env = _os.environ.get('GLOSS_VOCAB_PATH')
            if _env: args.gloss_vocab_path = _env
        gloss_tokenizer = GlossTokenizer(args.gloss_vocab_path)
        log(f'[*] GlossTokenizer: effective_vocab_size={len(gloss_tokenizer)} from {args.gloss_vocab_path}')
        log(f'    bos={gloss_tokenizer.bos_id} eos={gloss_tokenizer.eos_id} '
            f'pad={gloss_tokenizer.pad_id} unk={gloss_tokenizer.unk_id}')

    tokens_dir = Path(args.tokens_dir)
    train_set = TMSignDataset(tokens_dir / 'train_tokens.pt', tokenizer,
                                num_vq=args.num_vq, max_text_len=args.max_text_len,
                                max_motion_len=args.block_size, lang_code=args.lang,
                                gloss_tokenizer=gloss_tokenizer,
                                max_gloss_len=args.gloss_max_len)
    dev_set = TMSignDataset(tokens_dir / 'dev_tokens.pt', tokenizer,
                              num_vq=args.num_vq, max_text_len=args.max_text_len,
                              max_motion_len=args.block_size, lang_code=args.lang,
                              gloss_tokenizer=gloss_tokenizer,
                              max_gloss_len=args.gloss_max_len)
    gloss_pad = gloss_tokenizer.pad_id if gloss_tokenizer is not None else None
    collate = partial(collate_tm_sign, motion_pad_id=args.num_vq + 1, gloss_pad_id=gloss_pad)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, drop_last=True,
                                collate_fn=collate)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate)
    log(f'[*] train: {len(train_set)}, dev: {len(dev_set)}')

    model = CrossAttnText2MotionTransformer(
        num_vq=args.num_vq,
        text_dim=args.text_dim,
        embed_dim=args.embed_dim,
        block_size=args.block_size,
        num_layers=args.num_layers,
        n_head=args.n_head,
        drop_out_rate=args.drop_out_rate,
        fc_rate=args.fc_rate,
        align_dim=args.align_dim,
        predict_length=bool(args.predict_length),
    ).to(args.device)
    log(f'[*] CrossAttn Trans params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

    # Build gloss decoder (after text_enc and model so we know text_dim)
    if args.gloss_supervised:
        gloss_dec = GlossDecoder(
            gloss_vocab_size=len(gloss_tokenizer),
            text_dim=args.text_dim,
            embed_dim=args.gloss_dec_dim,
            num_layers=args.gloss_dec_layers,
            n_head=args.gloss_dec_heads,
            drop_out_rate=args.drop_out_rate,
            max_len=args.gloss_max_len,
            pad_id=gloss_tokenizer.pad_id,
        ).to(args.device)
        log(f'[*] GlossDecoder: {sum(p.numel() for p in gloss_dec.parameters())/1e6:.2f}M params, '
            f'{args.gloss_dec_layers}L embed {args.gloss_dec_dim} heads {args.gloss_dec_heads}')

    # Include text encoder params in optimizer if trainable (char encoder always; mBART if not frozen)
    optimizable_params = list(model.parameters())
    if text_enc_trainable:
        optimizable_params += list(text_enc.parameters())
        log(f'[*] including text_enc params in optimizer (text_encoder={args.text_encoder})')
    if gloss_dec is not None:
        optimizable_params += list(gloss_dec.parameters())
        log(f'[*] including gloss_dec params in optimizer (lambda_gloss={args.lambda_gloss})')
    optimizer = optim.AdamW(optimizable_params, lr=args.lr, betas=(0.9, 0.99),
                              weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler,
                                                 gamma=args.gamma)
    # Loss: ignore PAD (= num_vq + 1)
    ce = nn.CrossEntropyLoss(ignore_index=args.num_vq + 1,
                              label_smoothing=args.label_smoothing)
    # Gloss loss: ignore gloss pad
    ce_gloss = None
    if gloss_dec is not None:
        ce_gloss = nn.CrossEntropyLoss(ignore_index=gloss_tokenizer.pad_id)
    # Length loss: SmoothL1 on log-length
    smooth_l1 = nn.SmoothL1Loss()

    # Effective min_iter_before_early_stop: must be ≥ last lr milestone so
    # the model gets meaningful training time at the final lr level (FIX B).
    last_lr_milestone = max(args.lr_scheduler) if args.lr_scheduler else 0
    effective_min_iter = max(args.min_iter_before_early_stop, last_lr_milestone + 1000)
    if effective_min_iter != args.min_iter_before_early_stop:
        log(f'[*] FIX B: min_iter_before_early_stop bumped {args.min_iter_before_early_stop} '
            f'→ {effective_min_iter} (last lr milestone {last_lr_milestone} + 1000 buffer)')

    # FIX C: hard cap on (it - best_iter). Prevents FIX A from indefinitely resetting
    # patience on every lr drop, which can let training drag on forever past the point
    # where the model has stopped improving globally.
    if args.max_no_improve_iter > 0:
        max_no_improve = args.max_no_improve_iter
    else:
        max_no_improve = 3 * args.early_stop_patience * args.eval_iter
    log(f'[*] FIX C: max_no_improve_iter = {max_no_improve} '
        f'(hard stop if no new best for this many iters, regardless of lr resets)')

    it = 0
    t0 = time.time()
    best_dev = float('inf')
    best_iter = 0
    no_improve = 0
    prev_lr = args.lr   # FIX A: track lr to reset patience on lr drop
    train_iter = iter(train_loader)
    while it < args.total_iter:
        try: batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader); batch = next(train_iter)

        ids = batch['input_ids'].to(args.device)
        am = batch['attention_mask'].to(args.device)
        mt = batch['motion_tokens'].to(args.device)

        if it < args.warm_up_iter:
            cur_lr = warmup_lr(optimizer, it, args.warm_up_iter, args.lr)
        else:
            cur_lr = optimizer.param_groups[0]['lr']

        # FIX A (v2): on lr drop, HALF-reset patience instead of full reset.
        # Rationale: lr drop deserves a partial second chance, but full reset lets
        # already-diverging training drag forever (observed: gloss run iter 14K best,
        # then drifted 14K→38K with patience reset hiding the divergence).
        # Half-reset = "give lr drop a chance, but cap the damage".
        if cur_lr < prev_lr - 1e-12:
            if no_improve > 0:
                new_no_improve = no_improve // 2
                log(f'    [FIX A] lr dropped {prev_lr:.6f} → {cur_lr:.6f} at iter {it}: '
                    f'HALF-reset patience ({no_improve}/{args.early_stop_patience} → '
                    f'{new_no_improve}/{args.early_stop_patience})')
                no_improve = new_no_improve
        prev_lr = cur_lr

        if args.text_encoder == 'mbart' and args.freeze_text:
            with torch.no_grad():
                mem, mem_mask = encode_text_full(text_enc, ids, am, kind='mbart')
        else:
            if args.text_encoder != 'mbart':
                text_enc.train()
            mem, mem_mask = encode_text_full(text_enc, ids, am, kind=args.text_encoder)

        inp, tgt = prepend_bos(mt, bos_id=args.num_vq, pad_id=args.num_vq + 1)

        # Motion-token mask regularization: randomly replace some input tokens with
        # random valid tokens. Only applied during training and only to non-special
        # positions (skip BOS at index 0; skip PAD positions).
        if args.motion_token_mask_prob > 0 and model.training:
            mask = torch.rand(inp.shape, device=inp.device) < args.motion_token_mask_prob
            mask[:, 0] = False                                       # never mask BOS
            mask = mask & (inp != args.num_vq + 1)                   # never mask PAD
            mask = mask & (inp != args.num_vq)                       # never mask BOS/EOS
            random_tokens = torch.randint(0, args.num_vq, inp.shape, device=inp.device)
            inp = torch.where(mask, random_tokens, inp)

        if args.align_dim > 0:
            logits, mot_feat, text_mem = model.forward_with_features(inp, mem, mem_mask)
        else:
            logits = model(inp, mem, mem_mask)

        ce_loss = ce(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        total_loss = ce_loss

        align_loss_val = 0.0
        if args.align_dim > 0:
            align_loss = compute_align_loss(model, mot_feat, text_mem, mem_mask, mt, args.num_vq)
            total_loss = total_loss + args.lambda_align * align_loss
            align_loss_val = align_loss.item()

        gloss_loss_val = 0.0
        if gloss_dec is not None and 'gloss_input_ids' in batch:
            g_ids = batch['gloss_input_ids'].to(args.device)
            g_am  = batch['gloss_attention_mask'].to(args.device)
            # Teacher forcing: input = g_ids[:, :-1] (BOS..token_{T-2}); target = g_ids[:, 1:]
            g_inp = g_ids[:, :-1]
            g_tgt = g_ids[:, 1:]
            gloss_dec.train()
            g_logits = gloss_dec(g_inp, mem, mem_mask)
            gloss_loss = ce_gloss(g_logits.reshape(-1, g_logits.size(-1)), g_tgt.reshape(-1))
            total_loss = total_loss + args.lambda_gloss * gloss_loss
            gloss_loss_val = gloss_loss.item()

        # Length head: predict log(gt_len) from pooled text memory
        length_loss_val = 0.0
        if model.predict_length:
            # mem (B, T_text, embed_dim) was already projected inside model.forward;
            # we need to re-project here (text mem was not exposed). Easiest:
            # project on-the-fly using model's text_proj+text_norm.
            mem_proj = model.text_norm(model.text_proj(mem))
            mem_kpm = (mem_mask == 0)
            log_pred_len = model.predict_motion_length(mem_proj, mem_kpm)        # (B,)
            # gt length (motion_len) excludes BOS but includes EOS; we want the number
            # of actual motion tokens, so subtract 1 for the appended EOS slot.
            gt_len = (batch['motion_len'].to(args.device).float() - 1.0).clamp(min=1.0)
            log_gt_len = torch.log(gt_len)
            length_loss = smooth_l1(log_pred_len, log_gt_len)
            total_loss = total_loss + args.lambda_length * length_loss
            length_loss_val = length_loss.item()

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(optimizable_params, 1.0)
        optimizer.step()
        if it >= args.warm_up_iter:
            scheduler.step()

        if it % args.print_iter == 0:
            extra = f' | align {align_loss_val:.4f}' if args.align_dim > 0 else ''
            if gloss_dec is not None:
                extra += f' | gloss {gloss_loss_val:.4f}'
            if model.predict_length:
                extra += f' | len {length_loss_val:.4f}'
            log(f'iter {it:>6d} | lr {cur_lr:.6f} | ce {ce_loss.item():.4f}{extra} | '
                f'elapsed {time.time()-t0:.0f}s')

        if it > 0 and it % args.eval_iter == 0:
            ev = eval_dev_loss(model, text_enc, dev_loader, args.device, ce, args.num_vq,
                                 args, text_enc_kind=args.text_encoder,
                                 gloss_dec=gloss_dec, ce_gloss=ce_gloss)
            dev_ce, dev_align = ev[0], ev[1]
            extra = f' | dev_align {dev_align:.4f}' if args.align_dim > 0 else ''
            if gloss_dec is not None:
                dev_gloss, dev_gloss_acc = ev[2], ev[3]
                extra += f' | dev_gloss {dev_gloss:.4f} | gloss_acc {dev_gloss_acc:.3f}'
            log(f'    >> DEV iter {it} ce={dev_ce:.4f}{extra}')
            improved = dev_ce < best_dev - args.early_stop_min_delta
            if improved:
                best_dev = dev_ce; best_iter = it; no_improve = 0
                ckpt_dict = {'model': model.state_dict(), 'iter': it,
                              'args': vars(args), 'dev_ce': dev_ce}
                if text_enc_trainable:
                    ckpt_dict['text_enc'] = text_enc.state_dict()
                if gloss_dec is not None:
                    ckpt_dict['gloss_dec'] = gloss_dec.state_dict()
                torch.save(ckpt_dict, out_dir / 'best.pt')
                log(f'    >> SAVED best @ dev ce {best_dev:.4f} (iter {best_iter})')
            else:
                no_improve += 1
                log(f'    >> no improvement ({no_improve}/{args.early_stop_patience}), '
                    f'best still {best_dev:.4f} @ iter {best_iter}')
                # FIX B: use effective_min_iter (≥ last lr milestone)
                if (it >= effective_min_iter and
                        no_improve >= args.early_stop_patience):
                    log(f'    >> EARLY STOP at iter {it} (patience exhausted)')
                    break

                # FIX C: hard cap on iters since best, regardless of FIX A resets
                iters_since_best = it - best_iter
                if (it >= effective_min_iter and
                        iters_since_best > max_no_improve):
                    log(f'    >> HARD STOP at iter {it} (iters_since_best={iters_since_best} '
                        f'> max_no_improve={max_no_improve}; FIX C — best @ iter {best_iter})')
                    break

        if it > 0 and it % args.save_iter == 0:
            iter_ckpt = {'model': model.state_dict(), 'iter': it, 'args': vars(args)}
            if text_enc_trainable:
                iter_ckpt['text_enc'] = text_enc.state_dict()
            if gloss_dec is not None:
                iter_ckpt['gloss_dec'] = gloss_dec.state_dict()
            torch.save(iter_ckpt, out_dir / f'iter{it}.pt')

        it += 1

    final_ckpt = {'model': model.state_dict(), 'iter': it, 'args': vars(args)}
    if text_enc_trainable:
        final_ckpt['text_enc'] = text_enc.state_dict()
    if gloss_dec is not None:
        final_ckpt['gloss_dec'] = gloss_dec.state_dict()
    torch.save(final_ckpt, out_dir / 'final.pt')
    log(f'[OK] done in {(time.time()-t0)/60:.1f} min, best dev ce {best_dev:.4f}')
    log_f.close()


if __name__ == '__main__':
    main()
