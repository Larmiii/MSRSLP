"""Convert existing v2 SLP pickle → SLRTP dict format → run SLRTP main.py."""
from __future__ import annotations
import argparse, gzip, pickle, subprocess, json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred-pickle', required=True, help='gzip pickle with [{name, sign (T,534), ...}]')
    ap.add_argument('--gt-pt', required=True)
    ap.add_argument('--bt-model-dir', required=True)
    ap.add_argument('--slrtp-repo', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--py', default='python', help='Python to run SLRTP main.py')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load existing pickle
    with gzip.open(args.pred_pickle, 'rb') as f:
        preds = pickle.load(f)
    print(f'[*] loaded {len(preds)} predictions from {args.pred_pickle}')

    # Convert to {sid: pose (N, 178, 3)} dict
    pred_dict = {}
    for p in preds:
        sign = p['sign']
        if torch.is_tensor(sign):
            sign_np = sign.cpu().numpy()
        else:
            sign_np = np.asarray(sign)
        if sign_np.ndim == 2:
            sign_np = sign_np.reshape(sign_np.shape[0], 178, 3)
        pred_dict[p['name']] = torch.from_numpy(sign_np.astype(np.float32))

    pred_path = out_dir / f'{args.tag}_preds.pt'
    torch.save(pred_dict, pred_path)
    print(f'[OK] saved SLRTP-format preds: {pred_path}')

    cmd = [args.py, 'main.py', str(pred_path), str(args.gt_pt), str(args.bt_model_dir),
            '--tag', args.tag, '--fps', str(args.fps)]
    print(f'[*] Running: cwd={args.slrtp_repo} cmd={" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=args.slrtp_repo, capture_output=True, text=True)
    print('===STDOUT===')
    print(result.stdout)
    if result.returncode != 0:
        print('===STDERR===')
        print(result.stderr[-2000:])
        return

    # Parse results.json
    json_path = Path(args.slrtp_repo) / 'results' / f'{args.tag}.json'
    if json_path.exists():
        with open(json_path) as f:
            results = json.load(f)
        print('===RESULTS===')
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
