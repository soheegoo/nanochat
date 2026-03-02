"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (half context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    peri_ln: bool = False  # Peri-LN: add output normalization after each sublayer (Kim et al., ICML 2025)
    diff_attn: bool = False  # Differential Attention: noise-cancelling via softmax difference (Ye et al., ICLR 2025)


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        # Differential attention (Ye et al., ICLR 2025): attention = softmax(Q1K1) - λ·softmax(Q2K2)
        self.diff_attn = config.diff_attn
        if self.diff_attn:
            assert self.n_head % 2 == 0, "n_head must be even for differential attention"
            assert self.n_kv_head % 2 == 0, "n_kv_head must be even for differential attention"
            self.num_diff_heads = self.n_head // 2
            self.num_kv_diff_heads = self.n_kv_head // 2
            # V keeps standard shape; split into halves happens in forward (Variant 2)
            self.n_v_heads = self.n_kv_head
            self.v_head_dim = self.head_dim
            # Lambda reparameterization vectors (shared across heads, per layer)
            self.lambda_q1 = nn.Parameter(torch.empty(self.head_dim))
            self.lambda_k1 = nn.Parameter(torch.empty(self.head_dim))
            self.lambda_q2 = nn.Parameter(torch.empty(self.head_dim))
            self.lambda_k2 = nn.Parameter(torch.empty(self.head_dim))
            # Per-head RMSNorm with learnable affine weight
            self.diff_norm_weight = nn.Parameter(torch.empty(2 * self.head_dim))
            # Lambda init constant (not learnable, depends on layer depth)
            self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * layer_idx)
        else:
            self.n_v_heads = self.n_kv_head
            self.v_head_dim = self.head_dim
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_v_heads, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project Q and K (same reshape for both standard and differential attention)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        # Project V (always n_kv_head × head_dim; diff_attn splits in forward)
        v = self.c_v(x).view(B, T, self.n_v_heads, self.v_head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_v_heads, self.v_head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_v_heads), range (0, 2)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        if self.diff_attn:
            # Differential attention (Variant 2): split Q, K, V into sub-head pairs
            # All tensors keep the same head_dim for FA3 compatibility
            q = q.reshape(B, T, self.num_diff_heads, 2, self.head_dim)
            q1, q2 = q[:, :, :, 0, :].contiguous(), q[:, :, :, 1, :].contiguous()
            k = k.reshape(B, T, self.num_kv_diff_heads, 2, self.head_dim)
            k1, k2 = k[:, :, :, 0, :].contiguous(), k[:, :, :, 1, :].contiguous()
            v = v.reshape(B, T, self.num_kv_diff_heads, 2, self.head_dim)
            v1, v2 = v[:, :, :, 0, :].contiguous(), v[:, :, :, 1, :].contiguous()

            if kv_cache is None:
                # Training: 4 flash_attn calls (mathematically identical to Variant 1)
                attn11 = flash_attn.flash_attn_func(q1, k1, v1, causal=True, window_size=window_size)
                attn12 = flash_attn.flash_attn_func(q1, k1, v2, causal=True, window_size=window_size)
                attn21 = flash_attn.flash_attn_func(q2, k2, v1, causal=True, window_size=window_size)
                attn22 = flash_attn.flash_attn_func(q2, k2, v2, causal=True, window_size=window_size)
            else:
                # Inference: 4 flash_attn_with_kvcache calls (k insertions are idempotent)
                k1_cache, k2_cache, v1_cache, v2_cache = kv_cache.get_layer_cache(self.layer_idx)
                attn11 = flash_attn.flash_attn_with_kvcache(
                    q1, k1_cache, v1_cache, k=k1, v=v1,
                    cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=window_size,
                )
                attn12 = flash_attn.flash_attn_with_kvcache(
                    q1, k1_cache, v2_cache, k=k1, v=v2,
                    cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=window_size,
                )
                attn21 = flash_attn.flash_attn_with_kvcache(
                    q2, k2_cache, v1_cache, k=k2, v=v1,
                    cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=window_size,
                )
                attn22 = flash_attn.flash_attn_with_kvcache(
                    q2, k2_cache, v2_cache, k=k2, v=v2,
                    cache_seqlens=kv_cache.cache_seqlens, causal=True, window_size=window_size,
                )
                # Advance position after last layer processes
                if self.layer_idx == kv_cache.n_layers - 1:
                    kv_cache.advance(T)

            # Recombine V halves: (B, T, num_diff_heads, 2*head_dim)
            attn1 = torch.cat([attn11, attn12], dim=-1)
            attn2 = torch.cat([attn21, attn22], dim=-1)

            # Compute lambda (float32 for numerical stability, clamped to prevent exp overflow)
            dot1 = torch.dot(self.lambda_q1.float(), self.lambda_k1.float()).clamp(-10, 10)
            dot2 = torch.dot(self.lambda_q2.float(), self.lambda_k2.float()).clamp(-10, 10)
            lambda_1 = torch.exp(dot1)
            lambda_2 = torch.exp(dot2)
            lambda_val = (lambda_1 - lambda_2 + self.lambda_init).to(attn1.dtype)

            # Differential subtraction: (B, T, num_diff_heads, 2*head_dim)
            y = attn1 - lambda_val * attn2

            # Per-head RMSNorm with learnable weight, scaled by (1 - lambda_init) for gradient flow
            y = F.rms_norm(y, (2 * self.head_dim,)) * self.diff_norm_weight
            y = y * (1 - self.lambda_init)
        else:
            # Standard attention
            # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
            # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
            if kv_cache is None:
                # Training: causal attention with optional sliding window
                y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
            else:
                # Inference: use flash_attn_with_kvcache which handles cache management
                k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
                y = flash_attn.flash_attn_with_kvcache(
                    q, k_cache, v_cache,
                    k=k, v=v,
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True,
                    window_size=window_size,
                )
                # Advance position after last layer processes
                if self.layer_idx == kv_cache.n_layers - 1:
                    kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.peri_ln = config.peri_ln

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        if self.peri_ln:
            # Peri-LN: norm both before and after each sublayer (Kim et al., ICML 2025)
            x = x + norm(self.attn(norm(x), ve, cos_sin, window_size, kv_cache))
            x = x + norm(self.mlp(norm(x)))
        else:
            x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
            x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)   # 1.0 => typical residual connections at init
        self.x0_lambdas.fill_(0.1)      # 0.1 => small initial weight for skip connection to input embedding

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init to zero so gates start at sigmoid(0) = 0.5, scaled by 2 -> 1.0 (neutral)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)

        # Differential attention parameters (lambda vectors + per-head norm weight)
        if self.config.diff_attn:
            for block in self.transformer.h:
                attn = block.attn
                # Lambda vectors: normal(0, 0.1) so lambda starts near lambda_init
                torch.nn.init.normal_(attn.lambda_q1, mean=0.0, std=0.1)
                torch.nn.init.normal_(attn.lambda_k1, mean=0.0, std=0.1)
                torch.nn.init.normal_(attn.lambda_q2, mean=0.0, std=0.1)
                torch.nn.init.normal_(attn.lambda_k2, mean=0.0, std=0.1)
                # Per-head RMSNorm weight: ones (neutral at init)
                attn.diff_norm_weight.fill_(1.0)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
            for ve in self.value_embeds.values():
                ve.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (half context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings, per-layer scalars, and 1D diff_attn params
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        diff_attn_1d_numel = sum(p.numel() for p in self.transformer.h.parameters() if p.dim() == 1)
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          diff_attn_1d_numel)
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window.
        # Standard attention: 12 * h * q * seq  (QK^T forward+backward: 6, AV forward+backward: 6)
        # Diff attention (Variant 2): 4 flash calls, each with h/2 heads and head_dim q
        #   = 4 * (6*(h/2)*q + 6*(h/2)*q) * seq = 4 * 12*(h/2)*q*seq = 24*h*q*seq
        #   But h/2 heads means: 4 * 12 * (h/2) = 24h/2 = ... simplified: 24*(h/2)*q = 12*h*q per call pair
        #   Total: 4 calls × 6*(h/2)*q*seq each for QK^T + AV = 4*12*(h/2)*q*seq = 24*h*q*seq
        attn_multiplier = 24 if self.config.diff_attn else 12
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += attn_multiplier * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        # Diff attn adds 1D params (lambda vectors, norm weight) that need AdamW, not Muon
        all_block_params = list(self.transformer.h.parameters())
        matrix_params = [p for p in all_block_params if p.dim() >= 2]
        diff_attn_params = [p for p in all_block_params if p.dim() == 1]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == len(matrix_params) + len(diff_attn_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        # Diff attn 1D params (lambda vectors, per-head norm weight) with AdamW
        # Lambda vectors go through exp(dot(q,k)) so they're exponentially sensitive —
        # need a much lower LR than other scalars to prevent divergence
        if diff_attn_params:
            param_groups.append(dict(kind='adamw', params=diff_attn_params, lr=scalar_lr * 0.02, betas=adam_betas, eps=1e-10, weight_decay=0.0))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx) # embed current token
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
