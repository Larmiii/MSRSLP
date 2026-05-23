"""1:1 SLRTP-official eval wrapper.

For each v2 ckpt: gen poses → SLRTP dict format → run official main.py → collect JSON.

Generation matches the SLRTP-style protocol:
- greedy argmax (no sampling — matches old eval_t2mgpt_phix_bt_msr.py)
- per-stream masked logits (enforce stream order)
- max_len = trans.block_size - 1
- decode 6 streams (for msr) / 3 streams (for ms) / 2 streams (for rvq) / 1 stream (baseline)
- output poses at original 25fps (SLRTP main.py will subsample by 2 if --fps 25)

Usage:
    python _slrtp_run_official.py \
        --variant msr --vq-ckpt ... --trans-ckpt ... \
        --gt-pt bt_eval_kit/slrtp_official/data_official/dev.pt \
        --tag v2_M1M2_dev
"""
from __future__ import annotations
import argparse, sys, json, subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm

RELEASE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RELEASE_ROOT / "code" / "src"))

# Greedy decoding helpers -----------------------------------------------------

@torch.no_grad()
def encode_text_mbart(enc, ids, mask):
    """Pool mBART encoder output (mean over valid positions)."""
    out = enc(input_ids=ids, attention_mask=mask).last_hidden_state
    m = mask.unsqueeze(-1).float()
    return out, m   # also return per-position output for cross-attn


@torch.no_grad()
def greedy_sample_cross_attn(trans, mem, am, stream_ranges, num_vq, max_len):
    """Greedy argmax sampling for our CrossAttnText2MotionTransformer.

    stream_ranges: list of (lo, hi) per token position; range[k] = allowed token IDs.
    num_vq: BOS = num_vq, EOS = num_vq (we use num_vq as the end-of-seq).
    Actually our trans uses end_token = num_vq (and pad = num_vq+1).
    """
    device = mem.device
    end_id = num_vq
    seq = []                                     # generated token IDs (no BOS in output)
    for step in range(max_len):
        # Build input: BOS + previous tokens
        inp_ids = [end_id] + seq                 # use end_id as BOS too (our convention)
        # Actually our trans uses BOS = num_vq, EOS = num_vq same value; or num_vq+1 as pad.
        # Check via trans.sample method to mirror its convention.
        # Use trans.forward with current sequence and read last logit.
        x = torch.tensor([inp_ids], device=device, dtype=torch.long)
        logits = trans(x, mem, am)                # (1, L, num_vq + 2 or num_vq + 1)
        last = logits[0, -1, :].clone()
        # Enforce stream range
        lo, hi = stream_ranges[step]
        mask = torch.full_like(last, -float('inf'))
        mask[lo:hi] = 0.0
        mask[end_id] = 0.0                       # allow EOS anywhere
        last = last + mask
        nxt = last.argmax(-1).item()
        if nxt == end_id:
            break
        seq.append(nxt)
    return seq


# Variant-specific decoding (use trans.sample with temperature=epsilon for near-greedy)
@torch.no_grad()
def sample_via_trans(trans, mem, am, stream_ranges, num_vq, max_len, rep_streams,
                       greedy=True):
    """Use trans.sample with greedy-equivalent settings."""
    if greedy:
        tokens = trans.sample(
            mem, am,
            max_len=max_len,
            temperature=1e-6,                    # near-deterministic argmax
            top_k=1, top_p=1.0,                  # only top-1
            stream_ranges=stream_ranges,
            rep_penalty=0.0,                     # no rep penalty
            max_run=0,                           # no max-run
            rep_streams=rep_streams,
        )
    else:
        tokens = trans.sample(
            mem, am, max_len=max_len,
            temperature=0.9, top_k=20,
            stream_ranges=stream_ranges,
            rep_penalty=1.5, max_run=4,
            rep_streams=rep_streams,
        )
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=['base', 'ms', 'rvq', 'msr'])
    ap.add_argument('--vq-ckpt', required=True)
    ap.add_argument('--trans-ckpt', required=True)
    ap.add_argument('--gt-pt', required=True, help='SLRTP-format gt .pt (dev/test) with {sid: {text, poses_3d}}')
    ap.add_argument('--bt-model-dir', default=str(RELEASE_ROOT / 'bt_eval_kit/slrtp_official/backTranslation_PHIX_model'))
    ap.add_argument('--slrtp-repo', default=str(RELEASE_ROOT / 'bt_eval_kit/slrtp_official'))
    ap.add_argument('--out-dir', required=True, help='where to save predictions .pt and SLRTP result.json')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--decoding', default='greedy', choices=['greedy', 'sample'])
    ap.add_argument('--max-len', type=int, default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load VQ + trans ---
    print(f'[*] Loading VQ ({args.variant}) from {args.vq_ckpt}')
    # Reuse eval_cross_slt_lift3d helpers via direct import
    sys.path.insert(0, str(RELEASE_ROOT / "code" / "eval"))
    from eval_cross_slt_lift3d import (
        load_vq_base, load_vq_ms, load_vq_msr, load_vq_rvq,
        get_stream_ranges_ms, get_stream_ranges_msr, get_stream_ranges_rvq,
        decode_motion_base, decode_motion_ms, decode_motion_rvq, decode_motion_msr,
    )
    if args.variant == 'base':
        vq, mean, std, av = load_vq_base(args.vq_ckpt, args.device)
    elif args.variant == 'ms':
        vq, mean, std, av = load_vq_ms(args.vq_ckpt, args.device)
    elif args.variant == 'rvq':
        vq, mean, std, av = load_vq_rvq(args.vq_ckpt, args.device)
    else:
        vq, mean, std, av = load_vq_msr(args.vq_ckpt, args.device)

    print(f'[*] Loading trans from {args.trans_ckpt}')
    from models.t2m_trans_cross import CrossAttnText2MotionTransformer
    ck = torch.load(args.trans_ckpt, map_location='cpu', weights_only=False)
    ta = SimpleNamespace(**ck['args'])
    max_len = args.max_len if args.max_len else (ta.block_size - 1)

    # Stream ranges + per-variant decode setup
    if args.variant == 'base':
        stream_ranges = [(0, ta.num_vq)] * max_len
        rep_streams = 1
        sub_codes = offsets = sub_order = None
    elif args.variant == 'ms':
        stream_ranges, stream_names, nb_per_stream = get_stream_ranges_ms(av, max_len)
        rep_streams = 3
        sub_codes = offsets = sub_order = None
    elif args.variant == 'rvq':
        stream_ranges, sub_order, sub_codes, offsets = get_stream_ranges_rvq(av, max_len)
        rep_streams = 2
    else:  # msr
        stream_ranges, sub_order, sub_codes, offsets = get_stream_ranges_msr(av, max_len)
        rep_streams = 6

    # Build trans
    trans = CrossAttnText2MotionTransformer(
        num_vq=ta.num_vq, embed_dim=ta.embed_dim, text_dim=ta.text_dim,
        block_size=ta.block_size, num_layers=ta.num_layers, n_head=ta.n_head,
        drop_out_rate=ta.drop_out_rate, fc_rate=ta.fc_rate,
        align_dim=getattr(ta, 'align_dim', 0),
        predict_length=getattr(ta, 'predict_length', 0),
        gloss_supervised=getattr(ta, 'gloss_supervised', 0),
        gloss_vocab=None,
        gloss_dec_layers=getattr(ta, 'gloss_dec_layers', 2),
        gloss_dec_dim=getattr(ta, 'gloss_dec_dim', 256),
        gloss_dec_heads=getattr(ta, 'gloss_dec_heads', 4),
        gloss_max_len=getattr(ta, 'gloss_max_len', 48),
    )
    trans.load_state_dict(ck['model'])
    trans.to(args.device).eval()

    # --- Load mBART encoder ---
    print(f'[*] Loading mBART ({ta.mbart_name}, lang={ta.lang})')
    from transformers import MBart50TokenizerFast, MBartModel
    tok = MBart50TokenizerFast.from_pretrained(ta.mbart_name, src_lang=ta.lang)
    mbart_enc = MBartModel.from_pretrained(ta.mbart_name).encoder.to(args.device).eval()

    # --- Load GT and generate ---
    print(f'[*] Loading GT from {args.gt_pt}')
    gt = torch.load(args.gt_pt, weights_only=True)
    sids = list(gt.keys())
    print(f'[*] {len(sids)} samples')

    predictions = {}
    for sid in tqdm(sids, desc='gen'):
        txt = gt[sid]['text']
        enc = tok(txt, truncation=True, max_length=128, return_tensors='pt')
        ids = enc['input_ids'].to(args.device)
        am = enc['attention_mask'].to(args.device)
        with torch.no_grad():
            mem = mbart_enc(input_ids=ids, attention_mask=am).last_hidden_state    # (1, T_text, 1024)

        tokens = sample_via_trans(
            trans, mem, am, stream_ranges, ta.num_vq, max_len, rep_streams,
            greedy=(args.decoding == 'greedy'))                                     # (1, T_out)

        if args.variant == 'base':
            motion = decode_motion_base(vq, tokens, mean, std, args.device)
        elif args.variant == 'ms':
            motion = decode_motion_ms(vq, tokens, mean, std, args.device, av)
        elif args.variant == 'rvq':
            motion = decode_motion_rvq(vq, tokens, mean, std, args.device, av)
        else:
            motion = decode_motion_msr(vq, tokens, mean, std, args.device,
                                          sub_codes, offsets, sub_order)
        # motion is (T, 534); reshape to (T, 178, 3)
        pose = motion.reshape(motion.shape[0], 178, 3).astype(np.float32)
        predictions[sid] = torch.from_numpy(pose)

    pred_path = out_dir / f'{args.tag}_preds.pt'
    torch.save(predictions, pred_path)
    print(f'[OK] saved {pred_path} ({len(predictions)} samples)')

    # --- Run SLRTP main.py ---
    cmd = ['python', 'main.py', str(pred_path), str(args.gt_pt), str(args.bt_model_dir),
            '--tag', args.tag, '--fps', str(args.fps)]
    print(f'[*] Running SLRTP eval: {" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=args.slrtp_repo, capture_output=True, text=True)
    print('===STDOUT===')
    print(result.stdout)
    if result.returncode != 0:
        print('===STDERR===')
        print(result.stderr)


if __name__ == '__main__':
    main()
