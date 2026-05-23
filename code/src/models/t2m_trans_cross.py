"""Cross-attention Text-to-Motion Transformer (proper seq2seq, not pooled-feat).

Replaces T2M-GPT's `Text2Motion_Transformer` (which conditions on a single
mean-pooled mBART vector) with a standard seq2seq decoder that **cross-attends
to the full mBART encoder output sequence**. This preserves sentence structure
that pooled-feat loses on long / open-domain inputs (CSL-Daily).

Architecture
------------
text → mBART encoder (frozen) → (B, T_text, 1024) "memory"
                                                          ↓ (cross-attn K, V)
[BOS] mot₀ mot₁ … motₜ-1 → emb + pos → TransformerDecoder
                                          - causal self-attn over motion tokens
                                          - cross-attn over text memory
                                          ↓
                                      LM head → next-token logits (vocab = num_vq + 1)

The +1 vocab slot is shared BOS/EOS. BOS is the first input position; EOS predicted
to terminate generation. Padded targets use a separate ignore_index (num_vq + 1)
that is reserved by `train_trans_sign_cross.py` — i.e., model vocab is num_vq+1,
loss-side pad id is num_vq+1 (out of vocab range, never predicted, just ignored).

This model is variant-agnostic: baseline, M1 (multi-stream interleaved), M2
(multi-stream + residual), and M2+M3 (with align) all use the same model, just
with different num_vq values. Multi-stream-specific token-range masking happens
during sampling (not in the model itself).
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnText2MotionTransformer(nn.Module):
    """Standard transformer decoder with cross-attn to text memory.

    Args
    ----
    num_vq:     size of motion token vocabulary (excluding BOS/EOS slot).
                The model's effective vocab = num_vq + 1 (the extra slot serves as
                both BOS input index and EOS prediction target).
    text_dim:   mBART encoder hidden dim (1024 for mbart-large-50).
    embed_dim:  decoder hidden dim.
    block_size: max motion-token sequence length (includes BOS).
    num_layers: transformer decoder layers.
    n_head:     attention heads.
    drop_out_rate: dropout in decoder.
    fc_rate:    feed-forward multiplier (dim_feedforward = embed_dim * fc_rate).
    align_dim:  if > 0, projection dim for InfoNCE alignment head (M3).
    """

    def __init__(
        self,
        num_vq: int,
        text_dim: int = 1024,
        embed_dim: int = 512,
        block_size: int = 320,
        num_layers: int = 6,
        n_head: int = 8,
        drop_out_rate: float = 0.1,
        fc_rate: int = 4,
        align_dim: int = 0,
        predict_length: bool = False,
    ):
        super().__init__()
        self.num_vq = num_vq
        self.embed_dim = embed_dim
        self.block_size = block_size

        # Embedding vocab = num_vq + 2 (covers regular codes 0..num_vq-1,
        # BOS/EOS slot at index num_vq, PAD slot at index num_vq+1).
        # LM head outputs num_vq + 1 (predicts 0..num_vq incl. EOS, never PAD).
        # Loss uses ignore_index=num_vq+1 to drop padded target positions.
        self.token_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Project mBART memory to decoder dim
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)

        self.input_drop = nn.Dropout(drop_out_rate)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=n_head,
            dim_feedforward=embed_dim * fc_rate,
            dropout=drop_out_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers,
                                              norm=nn.LayerNorm(embed_dim))

        # LM head — same +1 vocab (last index = EOS)
        self.lm_head = nn.Linear(embed_dim, num_vq + 1)

        # Optional InfoNCE alignment head (M3): pools motion features and projects to
        # a shared text/motion space.
        self.align_dim = align_dim
        if align_dim > 0:
            self.motion_align = nn.Sequential(
                nn.Linear(embed_dim, align_dim),
                nn.GELU(),
                nn.Linear(align_dim, align_dim),
            )
            self.text_align = nn.Sequential(
                nn.Linear(embed_dim, align_dim),
                nn.GELU(),
                nn.Linear(align_dim, align_dim),
            )

        # Optional length head: predicts total motion token length from pooled text memory.
        # Output is a single scalar (log-space, then exp() at inference to ensure positive).
        self.predict_length = predict_length
        if predict_length:
            self.length_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 1),
            )
        else:
            self.length_head = None

        self._init_weights()

    def predict_motion_length(self, mem: torch.Tensor, mem_key_pad: torch.Tensor) -> torch.Tensor:
        """Pool text memory (mean over valid positions) and predict log-length.

        Args:
          mem:          (B, T_text, embed_dim)  — already-projected text memory
          mem_key_pad:  (B, T_text)  — True = pad

        Returns:
          log_len:      (B,)  — predicted log motion length (use exp() to get length)
        """
        if self.length_head is None:
            raise RuntimeError("length_head not enabled in this model")
        valid = (~mem_key_pad).float().unsqueeze(-1)
        pooled = (mem * valid).sum(1) / valid.sum(1).clamp(min=1)   # (B, embed_dim)
        return self.length_head(pooled).squeeze(-1)                  # (B,)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _causal_mask(T: int, device) -> torch.Tensor:
        """Causal mask: (T, T) bool, True = forbidden (upper triangle excl. diag)."""
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def encode_text_memory(self, mbart_last_hidden: torch.Tensor,
                            text_attn_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project mBART encoder outputs and compute key padding mask for cross-attn.

        Args:
          mbart_last_hidden: (B, T_text, text_dim)
          text_attn_mask:    (B, T_text) — 1 = valid token, 0 = padding

        Returns:
          mem:               (B, T_text, embed_dim)
          mem_key_pad:       (B, T_text) — True = pad (PyTorch convention: True = ignore)
        """
        mem = self.text_norm(self.text_proj(mbart_last_hidden))
        # nn.Transformer expects True for padding positions
        mem_key_pad = (text_attn_mask == 0)
        return mem, mem_key_pad

    def forward(
        self,
        motion_tokens: torch.Tensor,
        mbart_last_hidden: torch.Tensor,
        text_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Args:
          motion_tokens:      (B, T_m) — must include BOS (= num_vq) as first token.
          mbart_last_hidden:  (B, T_text, text_dim)
          text_attn_mask:     (B, T_text) — 1 = valid token, 0 = padding

        Returns:
          logits:             (B, T_m, num_vq + 1) — predicts next token at each position.
                              Position t predicts token at original-seq index (t).
                              So usual training: feed [BOS, m0, m1, ..., m_{T-1}], target
                              [m0, m1, ..., m_{T-1}, EOS].
        """
        B, T = motion_tokens.shape
        assert T <= self.block_size, f"T={T} > block_size={self.block_size}"

        x = self.token_emb(motion_tokens) + self.pos_emb[:, :T]
        x = self.input_drop(x)

        mem, mem_key_pad = self.encode_text_memory(mbart_last_hidden, text_attn_mask)

        tgt_mask = self._causal_mask(T, x.device)

        out = self.decoder(
            tgt=x,
            memory=mem,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=mem_key_pad,
        )                                                     # (B, T, embed_dim)
        logits = self.lm_head(out)                            # (B, T, num_vq + 1)
        return logits

    def forward_with_features(
        self,
        motion_tokens: torch.Tensor,
        mbart_last_hidden: torch.Tensor,
        text_attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like forward() but also returns the pre-LM-head motion features and
        text memory features (for the InfoNCE align head).
        """
        B, T = motion_tokens.shape
        x = self.token_emb(motion_tokens) + self.pos_emb[:, :T]
        x = self.input_drop(x)
        mem, mem_key_pad = self.encode_text_memory(mbart_last_hidden, text_attn_mask)
        tgt_mask = self._causal_mask(T, x.device)
        out = self.decoder(
            tgt=x, memory=mem,
            tgt_mask=tgt_mask, memory_key_padding_mask=mem_key_pad,
        )
        logits = self.lm_head(out)
        return logits, out, mem  # mem already projected, (B, T_text, embed_dim)

    @torch.no_grad()
    def sample(
        self,
        mbart_last_hidden: torch.Tensor,
        text_attn_mask: torch.Tensor,
        max_len: int,
        temperature: float = 0.9,
        top_k: int = 20,
        top_p: float = 1.0,
        # For multi-stream variants: per-step allowed token ranges (lo, hi).
        # If provided, must be list of length max_len with (lo, hi) tuples.
        # Tokens outside [lo, hi) (and != EOS=num_vq) are masked to -inf.
        stream_ranges: Optional[list] = None,
        # Length-window control (use with length_head): only allow EOS within
        # [min_len, max_len_eff]. None = no constraint.
        min_len: Optional[int] = None,
        max_len_eff: Optional[int] = None,
        # Repetition penalty: at each step, subtract `rep_penalty` from logits of
        # any token that appeared in the last `rep_window` generated tokens.
        # rep_penalty = 0 => disabled.
        rep_penalty: float = 0.0,
        rep_window: int = 8,
        # No-repeat-run: forbid the same token from being chosen if it has been
        # chosen `max_run` times in a row. 0 = disabled.
        max_run: int = 0,
        # Per-stream repetition tracking (for multi-stream interleaved tokens).
        # If rep_streams > 1: rep_penalty and max_run are computed against the
        # subset of previously-emitted tokens that share the same stream id
        # (where stream_id = step % rep_streams). Default 1 = global tracking.
        rep_streams: int = 1,
    ) -> torch.Tensor:
        """Autoregressive sampling. Returns (1, T_out) token sequence, BOS-stripped,
        EOS-stripped (just the generated motion tokens)."""
        device = mbart_last_hidden.device
        eos_id = self.num_vq          # last index = EOS (and BOS)
        bos_id = self.num_vq

        cur = torch.tensor([[bos_id]], device=device, dtype=torch.long)
        out_tokens = []
        # For global no-repeat-run tracking
        last_token = None
        run_count = 0
        # Per-stream tracking: per-stream last token + consec count
        per_stream_last = [None] * rep_streams
        per_stream_run = [0] * rep_streams

        # Effective max length is min(max_len_eff, max_len) if length-window enabled
        if max_len_eff is not None:
            eff_max = min(max_len_eff, max_len)
        else:
            eff_max = max_len
        min_len_v = min_len if min_len is not None else 0

        # Pre-project text memory (don't recompute every step)
        mem, mem_key_pad = self.encode_text_memory(mbart_last_hidden, text_attn_mask)
        tgt_mask_full = self._causal_mask(self.block_size, device)

        for step in range(eff_max):
            T = cur.size(1)
            if T > self.block_size:
                cur = cur[:, -self.block_size:]
                T = self.block_size
            x = self.token_emb(cur) + self.pos_emb[:, :T]
            x = self.input_drop(x)
            tgt_mask = tgt_mask_full[:T, :T]
            out = self.decoder(
                tgt=x, memory=mem,
                tgt_mask=tgt_mask, memory_key_padding_mask=mem_key_pad,
            )
            logits = self.lm_head(out[:, -1, :])    # (1, num_vq + 1)

            # Stream-range masking (for M1/M2 multi-substream tokens)
            if stream_ranges is not None and step < len(stream_ranges):
                lo, hi = stream_ranges[step]
                allow = torch.full_like(logits, float('-inf'))
                allow[:, lo:hi] = 0.0
                allow[:, eos_id] = 0.0                # always allow EOS
                logits = logits + allow

            # Length-window: forbid EOS before min_len_v
            if step < min_len_v:
                logits[:, eos_id] = float('-inf')

            stream_id = step % rep_streams if rep_streams > 1 else 0

            # Repetition penalty: subtract from logits of recently-seen tokens
            if rep_penalty != 0.0 and len(out_tokens) > 0:
                if rep_streams > 1:
                    # Per-stream history: only tokens emitted at same stream-id positions
                    same_stream_hist = out_tokens[stream_id::rep_streams]
                    recent = same_stream_hist[-rep_window:]
                else:
                    recent = out_tokens[-rep_window:]
                for tok in set(recent):
                    logits[:, tok] -= rep_penalty

            # No-repeat-run: forbid same token if already at max_run consecutive count
            if max_run > 0:
                if rep_streams > 1:
                    ps_last = per_stream_last[stream_id]
                    ps_run = per_stream_run[stream_id]
                    if ps_run >= max_run and ps_last is not None:
                        logits[:, ps_last] = float('-inf')
                else:
                    if run_count >= max_run and last_token is not None:
                        logits[:, last_token] = float('-inf')

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-6)

            if top_k > 0:
                tv, ti = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
                nl = torch.full_like(logits, float('-inf'))
                nl.scatter_(-1, ti, tv)
                logits = nl

            if top_p < 1.0:
                sv, si = torch.sort(logits, dim=-1, descending=True)
                sp = F.softmax(sv, dim=-1)
                cp = torch.cumsum(sp, dim=-1)
                smask = cp > top_p
                smask[..., 0] = False
                sv = sv.masked_fill(smask, float('-inf'))
                nl = torch.full_like(logits, float('-inf'))
                nl.scatter_(-1, si, sv)
                logits = nl

            probs = F.softmax(logits, dim=-1)
            idx = torch.multinomial(probs, num_samples=1)     # (1, 1)
            tok_int = idx.item()
            if tok_int == eos_id:
                # Always break on EOS (we've already masked it out for step < min_len_v above)
                break
            out_tokens.append(tok_int)
            # Update global run counter
            if tok_int == last_token:
                run_count += 1
            else:
                run_count = 1
                last_token = tok_int
            # Update per-stream run counter
            if rep_streams > 1:
                if tok_int == per_stream_last[stream_id]:
                    per_stream_run[stream_id] += 1
                else:
                    per_stream_run[stream_id] = 1
                    per_stream_last[stream_id] = tok_int
            cur = torch.cat([cur, idx], dim=1)

        if not out_tokens:
            # Degenerate: predicted EOS at step 0. Return empty (caller should handle).
            return torch.zeros((1, 0), device=device, dtype=torch.long)
        return torch.tensor(out_tokens, device=device, dtype=torch.long).unsqueeze(0)
