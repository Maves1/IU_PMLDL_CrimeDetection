import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")

# A helper config class that contains model parameters
class Config:
    embed_dropout = 0.1
    ff_dropout = 0.1
    attn_dropout = 0.1

    num_embed = 768
    num_heads = 12
    num_blocks = 12

    batch_size = 32

    def __init__(self, vocab_size, max_seq_len) -> None:
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

class SelfAttention(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        if config.num_embed % config.num_heads != 0:
            raise ValueError("num_embed % num_heads != 0")

        self.num_embed = config.num_embed
        self.num_heads = config.num_heads

        self.c_attn = nn.Linear(config.num_embed, 3 * config.num_embed)  # key, query, value
        self.c_proj = nn.Linear(config.num_embed, config.num_embed)

        # regularization
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.ff_dropout)

        # Mask that makes sure that attention only affects left tokens (previous, not future ones)
        self.register_buffer("bias", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                                     .view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x):
        B, T, C = x.size()  # batch size, seq len, num_embed

        # query, key, value for every head in batch
        query, key, value = self.c_attn(x).split(self.num_embed, dim=2)

        key = key.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        query = query.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        value = value.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        # Causal self attention
        atn = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        atn = atn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        atn = F.softmax(atn, dim=-1)
        atn = self.attn_dropout(atn)

        y = atn @ value
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y

class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.num_embed)

        self.attention = SelfAttention(config)

        self.ln_2 = nn.LayerNorm(config.num_embed)

        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.num_embed, 4 * config.num_embed),
            c_proj  = nn.Linear(4 * config.num_embed, config.num_embed),
            act     = nn.GELU(),
            dropout = nn.Dropout(config.ff_dropout),
        ))

        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))

        return x

class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()

        self.max_seq_len = config.max_seq_len
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.num_embed),
            wpe = nn.Embedding(config.max_seq_len, config.num_embed),
            dropout = nn.Dropout(config.embed_dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.num_blocks)]),
            ln_f = nn.LayerNorm(config.num_embed)
        ))

        self.head = nn.Linear(config.num_embed, config.vocab_size)

    def forward(self, x, targets=None):
        # x.shape = (Batches, Seq length)

        seq_len = x.size(1)

        if seq_len > self.max_seq_len:
            raise ValueError("Sequence length is > max allowed length")

        token_emb = self.transformer.wte(x)  # Batch size, seq length, num_embed

        positions = torch.arange(0, seq_len,
                               dtype=torch.long,
                               device=device).unsqueeze(0)  # (1, max_seq_len)

        pos_emb = self.transformer.wpe(positions)  # 1, max_seq_len, num_embed

        x = self.transformer.dropout(token_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.head(x)
        # print(f"logits.shape: {logits.shape}")
        # print(f"targets.shape: {targets.shape}")

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)
            # print(f"LOGITS: {logits}\nTargets: {targets}")

        return logits, loss

    def generate(self, xs, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):

        for _ in range(max_new_tokens):
            if xs.size(1) > self.max_seq_len:
                x = xs[:, -self.max_seq_len:]
            else:
                x = xs

            logits, _ = self(x)

            # Taking last logits
            logits = logits[:, -1, :] / temperature  # Also scaling by temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            # We can either sample from distribution or choose using top_k
            if do_sample:
                x_next = torch.multinomial(probs, num_samples=1)
            else:
                _, x_next = torch.topk(probs, k=1, dim=-1)

            xs = torch.cat((xs, x_next), dim=1)  # Adding a chosen token to the sequence
        return xs
