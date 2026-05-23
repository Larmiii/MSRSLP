"""Small Transformer decoder predicting gloss sequence from text_memory.

Used ONLY during training as auxiliary supervision for the text encoder.
NOT invoked at inference — the path is: text → text_encoder → motion decoder
(no gloss involvement).

Architecture:
    gloss_tokens (BOS-prepended) → embed + pos → TransformerDecoder
                                                      ↑ cross-attn
                                              text_memory (from text encoder)
                                                      ↓
                                                  LM head → gloss next-token logits

By being a small decoder (1-2 layers), it cannot solve the gloss task on its own
and must force the text_encoder to encode sign-language-relevant features into
text_memory.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class GlossDecoder(nn.Module):
    """AR Transformer decoder that predicts gloss tokens conditioned on text memory.

    Args:
      gloss_vocab_size:  effective size (including synthesized BOS/EOS).
                         LM head outputs this many classes.
      text_dim:          dim of the text_memory (from text encoder).
      embed_dim:         this decoder's hidden dim.
      num_layers:        kept small (1-2) intentionally — see module docstring.
      n_head:            attention heads.
      drop_out_rate:     dropout.
      max_len:           max gloss-token sequence length (for pos emb).
      pad_id:            padding token id in gloss vocab.
    """

    def __init__(
        self,
        gloss_vocab_size: int,
        text_dim: int = 384,
        embed_dim: int = 256,
        num_layers: int = 2,
        n_head: int = 4,
        drop_out_rate: float = 0.1,
        max_len: int = 64,
        pad_id: int = 2,
        fc_rate: int = 4,
    ):
        super().__init__()
        self.gloss_vocab_size = gloss_vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(gloss_vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.trunc_normal_(self.tok_emb.weight, std=0.02)

        # Project text memory into our embed_dim
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
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )

        self.lm_head = nn.Linear(embed_dim, gloss_vocab_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _causal_mask(T: int, device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        gloss_input_ids: torch.Tensor,
        text_memory: torch.Tensor,
        text_attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          gloss_input_ids:   (B, T_gloss) — must include BOS at position 0.
          text_memory:       (B, T_text, text_dim) — from text encoder.
          text_attn_mask:    (B, T_text) — 1 = valid, 0 = pad.

        Returns:
          logits:            (B, T_gloss, gloss_vocab_size)
        """
        B, T = gloss_input_ids.shape
        assert T <= self.max_len, f"T={T} > max_len={self.max_len}"

        x = self.tok_emb(gloss_input_ids) + self.pos_emb[:, :T]
        x = self.input_drop(x)

        mem = self.text_norm(self.text_proj(text_memory))
        mem_kpm = (text_attn_mask == 0)
        tgt_mask = self._causal_mask(T, x.device)

        out = self.decoder(
            tgt=x, memory=mem,
            tgt_mask=tgt_mask, memory_key_padding_mask=mem_kpm,
        )
        return self.lm_head(out)
