"""Stage 2 dataset for M1+M2 combined: Multi-Stream Residual tokens (6-stream interleaved).

Encoding: 6 sub-streams arranged in canonical order per timestep:
  body_base, body_res, hand_base, hand_res, face_base, face_res
Global id offsets:
  body_base: [0,                                                          nb_bb)
  body_res:  [nb_bb,                                                      nb_bb+nb_br)
  hand_base: [nb_bb+nb_br,                                                nb_bb+nb_br+nb_hb)
  ...
END = total
PAD = total + 1
"""
from __future__ import annotations
import numpy as np
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence


SUB_ORDER = ('body_base', 'body_res', 'hand_base', 'hand_res', 'face_base', 'face_res')


class TMSignMSRDataset(data.Dataset):
    def __init__(self, tokens_cache_path, tokenizer, stream_codes: dict,
                 max_text_len=128, max_motion_len=200, lang_code=None):
        """
        stream_codes: dict with keys SUB_ORDER → nb_code per sub-stream.
        """
        cache = torch.load(tokens_cache_path, map_location='cpu', weights_only=False)
        self.items = []
        for sid, v in cache.items():
            if 'tokens_body_base' not in v: continue
            T = len(v['tokens_body_base'])
            self.items.append({
                'sid': sid, 'text': v['text'] or '', 'gloss': v.get('gloss', ''),
                'T_tok': T,
                'sub_tokens': {s: v[f'tokens_{s}'].astype('int64') for s in SUB_ORDER},
            })
        self.tokenizer = tokenizer
        if lang_code is not None and hasattr(tokenizer, 'src_lang'):
            tokenizer.src_lang = lang_code
        self.stream_codes = stream_codes
        # Build offsets
        self.offsets = {}; off = 0
        for s in SUB_ORDER:
            self.offsets[s] = off; off += stream_codes[s]
        self.num_total = off
        self.END = self.num_total
        self.PAD = self.num_total + 1
        self.max_text_len = max_text_len
        self.max_motion_len = max_motion_len

    def __len__(self):
        return len(self.items)

    def encode_interleaved(self, sub_tokens):
        T = len(sub_tokens[SUB_ORDER[0]])
        n_sub = len(SUB_ORDER)
        out = np.zeros(T * n_sub, dtype=np.int64)
        for i, s in enumerate(SUB_ORDER):
            out[i::n_sub] = sub_tokens[s] + self.offsets[s]
        return out

    def decode_interleaved(self, flat_ids):
        flat = np.asarray(flat_ids); n_sub = len(SUB_ORDER)
        T = len(flat) // n_sub * n_sub
        flat = flat[:T]
        out = {}
        for i, s in enumerate(SUB_ORDER):
            raw = flat[i::n_sub] - self.offsets[s]
            out[s] = np.clip(raw, 0, self.stream_codes[s] - 1)
        return out

    def __getitem__(self, idx):
        ex = self.items[idx]
        tok = self.tokenizer(ex['text'], truncation=True,
                                max_length=self.max_text_len, return_tensors='pt')
        flat = self.encode_interleaved(ex['sub_tokens'])
        if len(flat) > self.max_motion_len - 1:
            flat = flat[:self.max_motion_len - 1]
        seq = np.concatenate([flat, [self.END]]).astype(np.int64)
        return {'sid': ex['sid'],
                 'input_ids': tok['input_ids'].squeeze(0),
                 'attention_mask': tok['attention_mask'].squeeze(0),
                 'motion_tokens': torch.from_numpy(seq),
                 'motion_len': seq.shape[0],
                 'text': ex['text'], 'gloss': ex['gloss']}


def collate_msr(batch, pad_id):
    input_ids = pad_sequence([b['input_ids'] for b in batch],
                               batch_first=True, padding_value=1)
    attn_mask = pad_sequence([b['attention_mask'] for b in batch],
                               batch_first=True, padding_value=0)
    motion_tokens = pad_sequence([b['motion_tokens'] for b in batch],
                                   batch_first=True, padding_value=pad_id)
    motion_len = torch.tensor([b['motion_len'] for b in batch], dtype=torch.long)
    return {'sids': [b['sid'] for b in batch],
             'texts': [b['text'] for b in batch],
             'glosses': [b['gloss'] for b in batch],
             'input_ids': input_ids, 'attention_mask': attn_mask,
             'motion_tokens': motion_tokens, 'motion_len': motion_len}
