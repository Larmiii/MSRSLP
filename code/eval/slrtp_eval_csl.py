"""SLRTP-style canonical eval for CSL: uses SLRTP back_translate() + sacrebleu directly.

CSL signjoey BT model has skeleton_subsample=1, so we DO NOT subsample poses.
Otherwise identical to SLRTP main.py.
"""
from __future__ import annotations
import argparse, gzip, pickle, sys, json
from pathlib import Path

import numpy as np
import torch

RELEASE_ROOT = Path(__file__).resolve().parents[2]
SLRTP_REPO = RELEASE_ROOT / 'bt_eval_kit' / 'slrtp_official'
sys.path.insert(0, str(SLRTP_REPO))

from back_translation.back_translate import back_translate, make_back_translation_model
from metrics import bleu, chrf, rouge, wer, pose_dtw_mje, pose_distance, pose_length


def load_pred_pickle_or_dict(path):
    """Accept gzip pickle [{name, sign}, ...] OR .pt dict {sid: pose}."""
    p = Path(path)
    if p.suffix in ('.pt',):
        d = torch.load(p, weights_only=True)
        return {k: (v if torch.is_tensor(v) else torch.from_numpy(np.asarray(v))) for k, v in d.items()}
    with gzip.open(p, 'rb') as f:
        preds = pickle.load(f)
    out = {}
    for s in preds:
        sign = s['sign']
        if torch.is_tensor(sign):
            x = sign
        else:
            x = torch.from_numpy(np.asarray(sign))
        if x.ndim == 2:
            x = x.reshape(x.shape[0], 178, 3)
        out[s['name']] = x.float()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True, help='gzip pickle or .pt dict')
    ap.add_argument('--gt-pt', required=True, help='GT .pt with {sid: {text, poses_3d}}')
    ap.add_argument('--bt-model-dir', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--no-subsample', action='store_true',
                     help='CSL: do not subsample (BT config has skeleton_subsample=1)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    pred_dict = load_pred_pickle_or_dict(args.pred)
    print(f'[*] {len(pred_dict)} predictions loaded')

    gt = torch.load(args.gt_pt, weights_only=False)
    sids = list(gt.keys())
    gt_text = [gt[s]['text'] for s in sids]
    gt_pose = [gt[s]['poses_3d'] for s in sids]
    gt_pose = [torch.from_numpy(np.asarray(p)) if not torch.is_tensor(p) else p for p in gt_pose]

    assert len(pred_dict) == len(sids), f'pred={len(pred_dict)} vs gt={len(sids)}'

    # Subsample? CSL: skeleton_subsample=1 → no subsample. PHIX/SLRTP: subsample by 2 if fps=25.
    if not args.no_subsample:
        pred_dict = {k: v[::2, ...] for k, v in pred_dict.items()}
        gt_pose = [v[::2, ...] for v in gt_pose]
        print(f'[*] subsampled by 2 (fps=25 → 12.5)')

    pose_preds = [pred_dict[s] for s in sids]

    print(f'[*] Loading BT model from {args.bt_model_dir}')
    bt = make_back_translation_model(model_dir=args.bt_model_dir)
    print(f'[*] Running BT...')
    text_pred = back_translate(model=bt, poses=pose_preds)

    # For Chinese (CSL): char-tokenize (space-separate each character) before sacrebleu,
    # so BLEU and other metrics see per-char tokens. Detect by checking ref text.
    def is_chinese(s):
        return any('一' <= c <= '鿿' for c in s)
    if any(is_chinese(t) for t in gt_text):
        print('[*] Chinese detected — char-tokenizing for BLEU/CHRF/ROUGE/WER')
        def tok_zh(t):
            return ' '.join(list(t.replace(' ', '')))
        text_pred_eval = [tok_zh(t) for t in text_pred]
        gt_text_eval = [tok_zh(t) for t in gt_text]
    else:
        text_pred_eval = text_pred
        gt_text_eval = gt_text

    bleu_s = bleu(hypotheses=text_pred_eval, references=gt_text_eval)
    chrf_s = chrf(hypotheses=text_pred_eval, references=gt_text_eval)
    rouge_s = rouge(hypotheses=text_pred_eval, references=gt_text_eval)
    wer_s = wer(hypotheses=text_pred_eval, references=gt_text_eval)
    dtw_s = pose_dtw_mje(hyps=pose_preds, gt_pose=gt_pose)
    td_s = pose_distance(hyps=pose_preds, gt_pose=gt_pose)
    dur_s = pose_length(hyps=pose_preds, gt_pose=gt_pose)

    results = {
        'bleu': bleu_s, 'chrf': chrf_s, 'rouge': rouge_s, 'wer': wer_s,
        'dtw_mje': dtw_s, 'total_distance': td_s, 'avg_duration': dur_s,
        'n_samples': len(sids),
    }
    print(json.dumps(results, indent=2, default=str))
    save_path = out_dir / f'{args.tag}.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    # Save text preds for bucket analysis later
    torch.save({'sids': sids, 'gt_text': gt_text, 'pred_text': text_pred},
               out_dir / f'{args.tag}_text_preds.pt')
    print(f'[OK] saved {save_path}')


if __name__ == '__main__':
    main()
