"""Character-level text encoder trained from scratch (no pretrained LM).

Inspired by Walsh et al. (FG 2024) Sign-VQ-Transformer: use a small
Transformer encoder over character embeddings rather than relying on
a frozen multilingual LM. For CSL-Daily (vocab 2281 chars) and PHIX-14T
(vocab ~3000 words), the SLP task is narrow enough that a custom encoder
trained jointly with the motion decoder converges fast and provides
task-specific representations.

Two classes:
  - CharTokenizer: drop-in replacement for HuggingFace tokenizer
                   (same __call__ signature, returns dict with input_ids/attention_mask).
  - CharTextEncoder: standard Transformer encoder over character embeddings.
"""
from __future__ import annotations
from pathlib import Path
from typing import Union, List

import torch
import torch.nn as nn


class CharTokenizer:
    """Drop-in replacement for HuggingFace tokenizer.

    Reads a vocab file where each line is a token. First 4 are
    reserved: <unk>, <pad>, <s>, </s>. Then real chars / words.
    """

    def __init__(self, vocab_path: Union[str, Path], src_lang: str = None):
        self.vocab_path = Path(vocab_path)
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.itos = [line.strip() for line in f if line.strip()]
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.unk_id = self.stoi.get('<unk>', 0)
        self.pad_id = self.stoi.get('<pad>', 1)
        self.bos_id = self.stoi.get('<s>', 2)
        self.eos_id = self.stoi.get('</s>', 3)
        # HF compat
        self.src_lang = src_lang
        self.pad_token_id = self.pad_id
        self.bos_token_id = self.bos_id
        self.eos_token_id = self.eos_id

    def __len__(self):
        return len(self.itos)

    def _tokenize_one(self, text: str, max_length: int) -> List[int]:
        # Char-level tokenization (each unicode char is a token)
        ids = [self.bos_id] + [self.stoi.get(c, self.unk_id) for c in text] + [self.eos_id]
        if len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.eos_id]
        return ids

    def __call__(self, text, truncation=True, max_length=128, return_tensors='pt',
                  padding=False):
        """Matches the HuggingFace tokenizer __call__ signature we use.

        Returns dict with input_ids (1, T) and attention_mask (1, T) tensors.
        Batch mode: text can be a list of strings → returns (B, T_max) with PAD.
        """
        if isinstance(text, str):
            ids = self._tokenize_one(text, max_length)
            input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            if return_tensors == 'pt':
                return {'input_ids': input_ids, 'attention_mask': attention_mask}
            return {'input_ids': ids, 'attention_mask': [1] * len(ids)}
        elif isinstance(text, (list, tuple)):
            # Batch tokenization with padding
            all_ids = [self._tokenize_one(t, max_length) for t in text]
            T_max = max(len(ids) for ids in all_ids)
            B = len(all_ids)
            input_ids = torch.full((B, T_max), self.pad_id, dtype=torch.long)
            attention_mask = torch.zeros((B, T_max), dtype=torch.long)
            for i, ids in enumerate(all_ids):
                L = len(ids)
                input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, :L] = 1
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            raise TypeError(f'Unsupported text type: {type(text)}')

    def decode(self, ids):
        """For debugging / inspection."""
        return ''.join(
            self.itos[i] if 0 <= i < len(self.itos) and i not in (self.bos_id, self.eos_id, self.pad_id) else ''
            for i in (ids.tolist() if isinstance(ids, torch.Tensor) else ids)
        )


class GlossTokenizer:
    """Word-level tokenizer for CSL gloss vocab.

    gls.vocab format: first 3 lines are special (<si>, <unk>, <pad>), then gloss words.
    Gloss strings are space-separated, e.g. "他 今年 4".

    Because gls.vocab does not include <s>/</s>, we synthesize BOS/EOS as extra
    indices beyond the file's vocab. Effective vocab size = len(itos) + 2.
    """

    def __init__(self, vocab_path: Union[str, Path]):
        self.vocab_path = Path(vocab_path)
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            self.itos = [line.strip() for line in f if line.strip()]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.unk_id = self.stoi.get('<unk>', 1)
        self.pad_id = self.stoi.get('<pad>', 2)
        # Synthesize BOS/EOS at the end
        self.bos_id = len(self.itos)        # = len(vocab_file)
        self.eos_id = len(self.itos) + 1    # = len(vocab_file) + 1
        # Effective vocab size includes BOS/EOS
        self._effective_vocab_size = len(self.itos) + 2

    def __len__(self):
        return self._effective_vocab_size

    def _tokenize_one(self, gloss_str: str, max_length: int):
        words = gloss_str.strip().split()
        ids = [self.bos_id] + [self.stoi.get(w, self.unk_id) for w in words] + [self.eos_id]
        if len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.eos_id]
        return ids

    def __call__(self, text, truncation=True, max_length=48, return_tensors='pt',
                  padding=False):
        if isinstance(text, str):
            ids = self._tokenize_one(text, max_length)
            input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        elif isinstance(text, (list, tuple)):
            all_ids = [self._tokenize_one(t, max_length) for t in text]
            T_max = max(len(ids) for ids in all_ids)
            B = len(all_ids)
            input_ids = torch.full((B, T_max), self.pad_id, dtype=torch.long)
            attention_mask = torch.zeros((B, T_max), dtype=torch.long)
            for i, ids in enumerate(all_ids):
                L = len(ids)
                input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, :L] = 1
            return {'input_ids': input_ids, 'attention_mask': attention_mask}
        else:
            raise TypeError(type(text))

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        out_words = []
        for i in ids:
            if i == self.bos_id or i == self.eos_id or i == self.pad_id:
                continue
            if 0 <= i < len(self.itos):
                out_words.append(self.itos[i])
        return ' '.join(out_words)


class CharTextEncoder(nn.Module):
    """Transformer encoder over character embeddings.

    Output is (B, T_text, embed_dim), drop-in replacement for mBART encoder's
    last_hidden_state. Designed to be trained jointly with the motion decoder
    (no pretraining).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 384,
        num_layers: int = 2,
        n_head: int = 8,
        drop_out_rate: float = 0.1,
        max_len: int = 256,
        pad_id: int = 1,
        fc_rate: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.tok_emb.weight, std=0.02)

        self.input_drop = nn.Dropout(drop_out_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=embed_dim * fc_rate,
            dropout=drop_out_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
          input_ids:       (B, T_text)
          attention_mask:  (B, T_text) — 1 = valid, 0 = pad

        Returns:
          last_hidden:     (B, T_text, embed_dim)
        """
        B, T = input_ids.shape
        x = self.tok_emb(input_ids) + self.pos_emb[:, :T]
        x = self.input_drop(x)

        # PyTorch convention: True = ignore
        kpm = (attention_mask == 0)
        out = self.encoder(x, src_key_padding_mask=kpm)
        return out


class MBartCompatWrapper(nn.Module):
    """Wraps CharTextEncoder so its forward matches mBART encoder's interface:
    returns object with `.last_hidden_state` attribute."""

    def __init__(self, char_enc: CharTextEncoder):
        super().__init__()
        self.char_enc = char_enc

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.char_enc(input_ids, attention_mask)

        class _Out:
            pass
        o = _Out()
        o.last_hidden_state = last_hidden_state
        return o
