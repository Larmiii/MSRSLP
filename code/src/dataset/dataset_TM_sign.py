"""Stage 2 (Text → motion-token) dataset for sign-language SLP.

Loads {sample_id: {text, gloss, tokens, T_orig, T_tok}} cache file produced by
tokenize_sign.py, returns batches of (text_input_ids, attention_mask, token_seq,
token_len) suitable for mBART-conditioned AR Transformer training.

Special tokens:
  MOTION_END_ID = num_vq      (used as EOS for motion sequence)
  MOTION_PAD_ID = num_vq + 1  (used to pad to max_len in batch)

The AR Transformer should ignore loss on PAD tokens (use ignore_index=num_vq+1).
"""
from __future__ import annotations
import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence


class TMSignDataset(data.Dataset):
    def __init__(self, tokens_cache_path, tokenizer, num_vq=512,
                 max_text_len=128, max_motion_len=64, lang_code=None,
                 gloss_tokenizer=None, max_gloss_len=48):
        """
        Args:
            tokens_cache_path: path to {split}_tokens.pt from tokenize_sign.py
            tokenizer: HuggingFace mBART tokenizer (already configured with src_lang)
            num_vq: codebook size, used to define END/PAD token ids
            max_text_len: truncate text to this many tokens
            max_motion_len: truncate motion seq to this many tokens (block_size of transformer)
            lang_code: e.g. 'de_DE' for PHIX, 'zh_CN' for CSL — passed via tokenizer.src_lang
        """
        cache = torch.load(tokens_cache_path, map_location='cpu', weights_only=False)
        self.items = []
        for sid, v in cache.items():
            txt = v['text'] or ''
            tokens = v['tokens']
            if len(tokens) == 0:
                continue
            self.items.append({
                'sid': sid,
                'text': txt,
                'tokens': tokens.astype('int64'),
                'gloss': v.get('gloss', ''),
            })
        self.tokenizer = tokenizer
        if lang_code is not None and hasattr(tokenizer, 'src_lang'):
            tokenizer.src_lang = lang_code
        self.num_vq = num_vq
        self.MOTION_END = num_vq
        self.MOTION_PAD = num_vq + 1
        self.max_text_len = max_text_len
        self.max_motion_len = max_motion_len
        # Optional gloss tokenizer (e.g. for gloss-supervised training)
        self.gloss_tokenizer = gloss_tokenizer
        self.max_gloss_len = max_gloss_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ex = self.items[idx]
        tok_text = self.tokenizer(
            ex['text'], truncation=True, max_length=self.max_text_len,
            return_tensors='pt')
        input_ids = tok_text['input_ids'].squeeze(0)
        attn_mask = tok_text['attention_mask'].squeeze(0)

        tokens = ex['tokens'][:self.max_motion_len - 1]   # leave room for END
        tokens = torch.tensor(list(tokens) + [self.MOTION_END], dtype=torch.long)
        out = {
            'sid': ex['sid'],
            'input_ids': input_ids,
            'attention_mask': attn_mask,
            'motion_tokens': tokens,
            'motion_len': tokens.size(0),
            'text': ex['text'],
            'gloss': ex['gloss'],
        }
        # Tokenize gloss if requested (for gloss-supervised aux loss)
        if self.gloss_tokenizer is not None and ex['gloss']:
            gtok = self.gloss_tokenizer(
                ex['gloss'], truncation=True, max_length=self.max_gloss_len,
                return_tensors='pt')
            out['gloss_input_ids'] = gtok['input_ids'].squeeze(0)
            out['gloss_attention_mask'] = gtok['attention_mask'].squeeze(0)
        return out


def collate_tm_sign(batch, motion_pad_id, gloss_pad_id=None):
    input_ids = pad_sequence([b['input_ids'] for b in batch],
                               batch_first=True, padding_value=1)  # mBART pad = 1
    attn_mask = pad_sequence([b['attention_mask'] for b in batch],
                               batch_first=True, padding_value=0)
    motion_tokens = pad_sequence([b['motion_tokens'] for b in batch],
                                   batch_first=True, padding_value=motion_pad_id)
    motion_len = torch.tensor([b['motion_len'] for b in batch], dtype=torch.long)
    out = {
        'sids': [b['sid'] for b in batch],
        'texts': [b['text'] for b in batch],
        'glosses': [b['gloss'] for b in batch],
        'input_ids': input_ids,
        'attention_mask': attn_mask,
        'motion_tokens': motion_tokens,
        'motion_len': motion_len,
    }
    # Optional gloss tokens (if dataset was constructed with gloss_tokenizer)
    if 'gloss_input_ids' in batch[0]:
        g_pad = gloss_pad_id if gloss_pad_id is not None else 0
        g_ids = pad_sequence([b['gloss_input_ids'] for b in batch],
                              batch_first=True, padding_value=g_pad)
        g_am = pad_sequence([b['gloss_attention_mask'] for b in batch],
                              batch_first=True, padding_value=0)
        out['gloss_input_ids'] = g_ids
        out['gloss_attention_mask'] = g_am
    return out
