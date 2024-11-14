import math

import torch
from torch import nn
from torch.nn import functional as F


class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):        
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.embedding = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        return self.embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=5000, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

        # Create a tensor for position indices # Shape: [max_len, 1]
        positions = torch.arange(0, self.max_len).unsqueeze(1)

        # Create a tensor for the even indices
        div_term = 10000 ** (torch.arange(0, self.emb_size, 2).float() / self.emb_size)

        # Apply sine and cosine functions on the entire tensor in one go
        pe = torch.zeros(self.max_len, self.emb_size)

        # Sine for even indices
        pe[:, 0::2] = torch.sin(positions / div_term)

        # Cosine for odd indices
        pe[:, 1::2] = torch.cos(positions / div_term)

        self.pe = pe.unsqueeze(0)  # Shape: [1, max_len, emb_size]

    def forward(self, x):
        x = x * math.sqrt(self.emb_size)
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)

class Embeddings(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.emb = InputEmbeddings(vocab_size, emb_size)
        self.pe = PositionalEncoding(emb_size)

    def forward(self, x):
        return self.pe(self.emb(x))

class SelfAttentionBlock(nn.Module):
    def __init__(self, emb_size, head_dim):
        super().__init__()
        self.query = nn.Linear(emb_size, head_dim, bias=False)
        self.key = nn.Linear(emb_size, head_dim, bias=False)
        self.value = nn.Linear(emb_size, head_dim, bias=False)
        self.scale = head_dim ** -0.5

    def forward(self, x, mask=None):
        query = self.query(x)  # Shape: [batch_size, seq_len, head_dim]
        key = self.key(x)      # Shape: [batch_size, seq_len, head_dim]
        value = self.value(x)  # Shape: [batch_size, seq_len, head_dim]

        scores  = torch.bmm(query, key.transpose(1, 2)) * self.scale  # Shape: [batch_size, seq_len, seq_len]


        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))  # Mask padding positions

        attention_weights = torch.softmax(scores, dim=-1) # Shape: [batch_size, seq_len, seq_len]
        context = torch.bmm(attention_weights, value)  # Shape: [batch_size, seq_len, head_dim]
        return context
        

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, emb_size, n_heads, dropout=0.1):
        super().__init__()
        assert emb_size % n_heads == 0
        self.head_dim = emb_size // n_heads
        self.heads = nn.ModuleList(
            [SelfAttentionBlock(emb_size, self.head_dim) for _ in range(n_heads)]
        )
        self.linear = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        x = torch.cat([head(x, mask) for head in self.heads], dim=-1)
        x = self.linear(x)
        return self.dropout(x)

class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size, expansion_factor):
        super().__init__()
        self.emb_size = emb_size
        self.expansion_factor = expansion_factor
        
        self.linear1 = nn.Linear(emb_size, expansion_factor * emb_size)
        self.linear2 = nn.Linear(expansion_factor * emb_size, emb_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.linear2(x)
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, emb_size, n_heads, expansion_factor):
        super().__init__()
        self.attention = MultiHeadAttentionBlock(emb_size, n_heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.feed_forward = FeedForwardBlock(emb_size, expansion_factor)
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attention(x, mask))
        x = self.norm2(x + self.feed_forward(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, vocab_size, emb_size, n_heads, n_layers, expansion_factor=4):
        super().__init__()

        self.embeddings = Embeddings(vocab_size, emb_size)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(emb_size, n_heads, expansion_factor) for _ in range(n_layers)
        ])

    def forward(self, x, mask=None):
        x = self.embeddings(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)

        return x
