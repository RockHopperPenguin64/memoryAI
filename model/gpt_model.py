import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from config.gpt_config import GPTConfig

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)

class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv_proj = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask=None):
        B, T, C = x.size()
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) for t in qkv]

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=(attn_mask is None)
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.out_proj(attn_output))

class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.norm1(x), attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.max_seq_len, cfg.d_model))
        self.dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])
        self.norm_final = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
        self.register_buffer("causal_mask", mask, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.cfg.max_seq_len
        tok = self.token_emb(idx)
        pos = self.pos_emb[:, :T, :]
        x = self.dropout(tok + pos)

        attn_mask = self.causal_mask[:T, :T]
        attn_mask = attn_mask == 0
        attn_mask = attn_mask.float().masked_fill(attn_mask, float("-inf"))

        for block in self.blocks:
            if self.cfg.use_checkpoint and self.training:
                x = checkpoint(lambda inp: block(inp, attn_mask), x, use_reentrant=False)
            else:
                x = block(x, attn_mask)

        x = self.norm_final(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            targets[:, 1:].contiguous().view(-1),
            label_smoothing=0.05,
            ignore_index = self.cfg.pad_token_id
        )
        return logits, loss
