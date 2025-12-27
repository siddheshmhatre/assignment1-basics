import math
import torch
import torch.nn as nn

from jaxtyping import Float
from einops import einsum, reduce, rearrange

class Linear(nn.Module):
    def __init__(self, in_features, out_features, weights=None, device=None, dtype=None):
        super().__init__()

        if weights is None:
            weights = torch.empty(out_features, in_features, dtype=dtype, device=device)
            stddev = (2 / (in_features + out_features)) ** 0.5
            nn.init.trunc_normal_(weights, std = stddev, a = -3 * stddev, b = 3 * stddev)

        self.W = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.W, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, weights=None, device=None, dtype=None):
        super().__init__()

        if weights is None:
            weights = torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
            nn.init.trunc_normal_(weights, std = 1, a = -3, b = -3)

        self.embeddings = nn.Parameter(weights)	

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, weights=None, device=None, dtype=None):
        super().__init__()

        if weights is None:
            weights = torch.ones(d_model, dtype=dtype, device=device)

        self.eps = eps
        self.d_model = d_model

        self.gain = nn.Parameter(weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        old_dtype = x.dtype
        x = x.to(torch.float32)

        # Ensure rms is computed for each token in each batch
        rms = reduce(x ** 2, "... dim -> ... 1", "mean")
        rms = torch.rsqrt(rms + self.eps)
        
        x = x * rms * self.gain

        return x.to(old_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, 
                    w1_weight: Float[torch.Tensor, " d_ff d_model"] = None, 
                    w2_weight: Float[torch.Tensor, " d_model d_ff"] = None, 
                    w3_weight: Float[torch.Tensor, " d_ff d_model"] = None):
        super().__init__()

        if d_ff is None:
            d_ff = int((8 / 3) * d_model)
            # Setting to a multiple of 64
            d_ff = 64 * math.ceil(d_ff / 64)

        self.W1 = Linear(d_model, d_ff, w1_weight)
        self.W2 = Linear(d_ff, d_model, w2_weight)
        self.W3 = Linear(d_model, d_ff, w3_weight)

    def forward(self, in_features: torch.Tensor):
        w3 = self.W3(in_features)
        w1 = self.W1(in_features)
        w1 = w1 * torch.sigmoid(w1)
        w2 = self.W2(w1 * w3)
        
        return w2

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        k = (2 * (torch.arange(1, int(d_k / 2) + 1) - 1)) / d_k
        angles = 1.0 / (theta ** k)
        positions = torch.arange(max_seq_len)
        angles = torch.outer(positions, angles)
        sines = angles.sin().to(device)
        cosines = angles.cos().to(device)

        # shape -> seq_len x d_model / 2
        self.register_buffer("sines_cached", sines)
        self.register_buffer("cosines_cached", cosines)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        sine = self.sines_cached[token_positions]
        cosine = self.cosines_cached[token_positions]

        x_evens = x[..., 0::2]
        x_odds = x[..., 1::2]

        # Add them up
        rot1 = cosine * x_evens - sine * x_odds
        rot2 = sine * x_evens + cosine * x_odds

        # Populate output tensor
        op = torch.empty_like(x)
        op[..., 0::2] = rot1
        op[..., 1::2] = rot2

        return op

class SoftMax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        max_val = x.max(dim=dim, keepdim=True)[0]
        x_shifted = x - max_val
        exp = x_shifted.exp() 
        return exp / exp.sum(dim=dim, keepdim=True)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = SoftMax()

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        d_k = Q.shape[-1]
        attn_scores = einsum(Q, K, "... n d_k, ... m d_k -> ... n m")
        attn_scores = attn_scores / math.sqrt(d_k)
        if mask is not None:
            attn_scores.masked_fill_(~mask, -math.inf)
        attn_scores = self.softmax(attn_scores, dim=-1)

        return einsum(attn_scores, V, "... n m, ... m d_v -> ... n d_v")

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, 
                q_proj_weight: torch.Tensor = None, 
                k_proj_weight: torch.Tensor = None, 
                v_proj_weight: torch.Tensor = None, 
                o_proj_weight: torch.Tensor = None,
                # RoPE Params
                theta: float = None, 
                max_seq_len: int = None):

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        weights = None
        if q_proj_weight is not None: # Assuming that if q weights are given then so are the rest
            weights = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=-2)
        self.QKV = Linear(d_model, d_model * 3, weights=weights)
        self.spda = ScaledDotProductAttention()
        self.o_proj = Linear(d_model, d_model, weights=o_proj_weight)

        if theta is not None: # Again assuming all RoPE params are provided
            self.rope = RoPE(theta, self.d_head, max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        seq_len = x.shape[1]
        x = self.QKV(x)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        Q, K, V = rearrange(x, "b s (qkv heads d_head) -> qkv b heads s d_head", qkv=3, heads=self.num_heads, d_head=self.d_head)

        if token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        x = self.spda(Q, K, V, mask)
        x = rearrange(x, "b heads s d_head -> b s (heads d_head)", heads=self.num_heads, d_head=self.d_head)

        x = self.o_proj(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()

        self.ln1 = RMSNorm(d_model)        
        self.ln2 = RMSNorm(d_model)        
        self.attn = MultiHeadSelfAttention(d_model, num_heads, theta=theta, max_seq_len=max_seq_len)
        self.ffn = SwiGLU(d_model, d_ff)

    def initialize_weights(self, weights: dict[str, torch.Tensor]):
        weights = {k: nn.Parameter(v) for k,v in weights.items()}

        self.ln1.gain = weights['ln1.weight']
        self.ln2.gain = weights['ln2.weight']
        self.attn.QKV.W = nn.Parameter(torch.cat([weights['attn.q_proj.weight'], 
                                weights['attn.k_proj.weight'], 
                                weights['attn.v_proj.weight']], dim=0))
        self.attn.o_proj.W = weights['attn.output_proj.weight']
        self.ffn.W1.W = weights['ffn.w1.weight']
        self.ffn.W2.W = weights['ffn.w2.weight']
        self.ffn.W3.W = weights['ffn.w3.weight']

    def forward(self, x):
        token_positions = torch.arange(x.shape[1])
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)

        self.transformer_lm = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def initialize_weights(self, weights: dict[str, torch.Tensor]):
        weights = {k: nn.Parameter(v) for k, v in weights.items()}
        self.embed.embeddings = weights['token_embeddings.weight']
        
        for idx, transformer_block in enumerate(self.transformer_lm):
            block_weights = {'.'.join(k.split('.')[2:]): v for k,v in weights.items() if k.split('.')[1] == str(idx)}
            transformer_block.initialize_weights(block_weights)

        self.ln_final.gain = weights['ln_final.weight']
        self.lm_head.W = weights['lm_head.weight']

    def forward(self, x):
        x = self.embed(x)

        for layer in self.transformer_lm:
            x = layer(x)

        return self.lm_head(self.ln_final(x))