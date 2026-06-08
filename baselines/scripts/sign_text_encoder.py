"""Drop-in text encoders for the sign-language MoMask reproduction.

Replaces MoMask's English CLIP text encoder so the baseline uses the SAME
text representation as the MSRSLP method (fair comparison):
  - 'mbart' : frozen mBART-50 encoder, mean-pooled over tokens -> (B, 1024)   [PHIX, German]
  - 'char'  : (placeholder for CSL char-level encoder; added when we do CSL)

MoMask conditions on a single pooled text vector, so we mean-pool mBART's
token features (masked by attention) into one vector per sentence.
"""
import torch
import torch.nn as nn


class MBartTextEncoder(nn.Module):
    out_dim = 1024
    def __init__(self, name='facebook/mbart-large-50', src_lang='de_DE', device='cpu'):
        super().__init__()
        from transformers import MBart50TokenizerFast, MBartModel
        self.tokenizer = MBart50TokenizerFast.from_pretrained(name, src_lang=src_lang)
        self.encoder = MBartModel.from_pretrained(name).encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        # keep the frozen mBART encoder in eval (no dropout) even when parent .train()
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, raw_text):
        device = next(self.encoder.parameters()).device
        enc = self.tokenizer(list(raw_text), return_tensors='pt', padding=True,
                             truncation=True, max_length=128)
        ids = enc['input_ids'].to(device)
        am = enc['attention_mask'].to(device)
        hid = self.encoder(input_ids=ids, attention_mask=am).last_hidden_state  # (B,T,1024)
        m = am.unsqueeze(-1).float()
        pooled = (hid * m).sum(1) / m.sum(1).clamp(min=1e-6)                     # mean-pool -> (B,1024)
        return pooled.float()


def build_sign_text_encoder(kind, device, src_lang='de_DE'):
    if kind == 'mbart':
        enc = MBartTextEncoder(src_lang=src_lang, device=device)
        enc.encoder.to(device)
        return enc, MBartTextEncoder.out_dim
    raise ValueError(f'unknown text encoder kind: {kind}')


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc, dim = build_sign_text_encoder('mbart', dev)
    feats = enc(['da haben wir morgen schon die dreißig grad',
                 'guten abend liebe zuschauer'])
    print('out dim', dim, '| feats', tuple(feats.shape), '| device', feats.device)
