"""Build char-level + gloss-level vocab from a lift3d pose dataset.

For PHIX (or any new dataset): produces txt.vocab and gls.vocab compatible with
CharTokenizer / GlossTokenizer in models/text_encoder_char.py.

Usage:
    python scripts/build_char_vocab.py \
        --data-dir ../data/phix \
        --out-dir ../data/phix/char_vocab

Assumes the data dir contains <prefix>_lift3d.{train,dev,test}.pt files where each
sample is a dict with 'text' (str) and 'gloss' (str) keys.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True, help='dir with <prefix>_lift3d.{train,dev,test}.pt')
    ap.add_argument('--out-dir', required=True, help='output dir for {txt,gls}.vocab')
    ap.add_argument('--prefix', default='', help='filename prefix before _lift3d. '
                                                    '(empty means autodetect)')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Detect prefix
    if not args.prefix:
        files = list(data_dir.glob('*_lift3d.train.pt'))
        if not files:
            raise FileNotFoundError(f"No *_lift3d.train.pt in {data_dir}")
        args.prefix = files[0].name.replace('_lift3d.train.pt', '')
        print(f'[*] autodetected prefix: {args.prefix}')

    char_counter = Counter()
    gloss_counter = Counter()
    n_samples = 0

    # Build vocab from train only
    train_path = data_dir / f'{args.prefix}_lift3d.train.pt'
    print(f'[*] loading {train_path}')
    cache = torch.load(train_path, map_location='cpu', weights_only=False)
    for sid, v in cache.items():
        txt = (v.get('text') or '').strip()
        gls = (v.get('gloss') or '').strip()
        # char-level for text
        for c in txt:
            char_counter[c] += 1
        # word-level for gloss (space-separated)
        for w in gls.split():
            gloss_counter[w] += 1
        n_samples += 1
    print(f'[*] {n_samples} train samples')
    print(f'    unique chars: {len(char_counter)}')
    print(f'    unique gloss words: {len(gloss_counter)}')

    # Write txt.vocab: 4 special tokens first, then chars by frequency
    txt_vocab_path = out_dir / 'txt.vocab'
    with open(txt_vocab_path, 'w', encoding='utf-8') as f:
        f.write('<unk>\n')
        f.write('<pad>\n')
        f.write('<s>\n')
        f.write('</s>\n')
        for c, _ in char_counter.most_common():
            f.write(f'{c}\n')
    print(f'[OK] wrote {txt_vocab_path} ({4 + len(char_counter)} entries)')

    # Write gls.vocab: 3 special tokens (matching CSL gls.vocab format: <si>, <unk>, <pad>)
    gls_vocab_path = out_dir / 'gls.vocab'
    with open(gls_vocab_path, 'w', encoding='utf-8') as f:
        f.write('<si>\n')
        f.write('<unk>\n')
        f.write('<pad>\n')
        for w, _ in gloss_counter.most_common():
            f.write(f'{w}\n')
    print(f'[OK] wrote {gls_vocab_path} ({3 + len(gloss_counter)} entries)')


if __name__ == '__main__':
    main()
