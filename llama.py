import os
import torch
import torch.nn as nn
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = nn.functional.silu(x_fc1) * x_fc2
        return self.fc3(x)

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
            freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
            (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    positions = torch.arange(context_length)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # (context_length, head_dim // 2)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin


def compute_rope(x, cos, sin, offset=0):
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Reshape into interleaved pairs: (x[0],x[1]), (x[2],x[3]), ...
    x_pairs = x.float().reshape(batch_size, num_heads, seq_len, head_dim // 2, 2)
    x1 = x_pairs[..., 0]  # even dims
    x2 = x_pairs[..., 1]  # odd dims

    cos = cos[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[offset:offset + seq_len, :].unsqueeze(0).unsqueeze(0)

    new_x1 = x1 * cos - x2 * sin
    new_x2 = x1 * sin + x2 * cos

    # Interleave back: [new_x1[0], new_x2[0], new_x1[1], new_x2[1], ...]
    x_rotated = torch.stack([new_x1, new_x2], dim=-1).reshape(batch_size, num_heads, seq_len, head_dim)

    return x_rotated.to(dtype=x.dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        b, num_tokens, d_in = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        keys = compute_rope(keys, cos, sin, offset=start_pos)
        queries = compute_rope(queries, cos, sin, offset=start_pos)

        if cache is not None:
            prev_k, prev_v = cache
            keys = torch.cat([prev_k, keys], dim=2)
            values = torch.cat([prev_v, values], dim=2)

        next_cache = (keys, values)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec, next_cache

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            dtype=cfg["dtype"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.norm2 = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])

    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        shortcut = x
        x = self.norm1(x)
        x, next_cache = self.att(x, mask, cos, sin, start_pos=start_pos, cache=cache)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x, next_cache

class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = nn.RMSNorm(cfg["emb_dim"], eps=1e-5, dtype=cfg["dtype"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        cos, sin = precompute_rope_params(
            head_dim=cfg["emb_dim"] // cfg["n_heads"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            freq_config=cfg["rope_freq"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.cfg = cfg
        self.current_pos = 0

    def forward(self, in_idx, cache=None):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        num_tokens = x.shape[1]

        if cache is not None:
            pos_start = self.current_pos
            pos_end = self.current_pos + num_tokens

            # [pos_start:pos_end, :pos_end] slices out rows of new tokens, because they are the only queries being computed.
            mask = torch.triu(
                torch.ones(pos_end, pos_end, device=x.device, dtype=torch.bool), diagonal=1
            )[pos_start:pos_end, :pos_end]

        else:
            pos_start = 0
            mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.bool), diagonal=1)

        for i, block in enumerate(self.trf_blocks):
            blk_cache = cache.get(i) if cache else None
            x, new_blk_cache = block(x, mask, self.cos, self.sin, start_pos=pos_start, cache=blk_cache)
            if cache is not None:
                cache.update(i, new_blk_cache)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg["dtype"]))
        return logits
    
    def reset_kv_cache(self):
        self.current_pos = 0

class Tokenizer:
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        mergeable = load_tiktoken_bpe(model_path)

        self.special = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<think>": 128002,
            "</think>": 128003,
            "<answer>": 128004,
            "</answer>": 128005,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special.update({
            f"<|reserved_{i}|>": 128002 + i
            for i in range(256)
            if 128002 + i not in self.special.values()
        })
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=(
                r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                r"|\p{N}{1,3}"
                r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+"
                r"|\s+(?!\S)"
                r"|\s+"
            ),
            mergeable_ranks=mergeable,
            special_tokens=self.special,
        )

    def encode(self, text, bos=True, eos=False):
        ids = ([self.special["<|begin_of_text|>"]] if bos else []) \
              + self.model.encode(text)
        if eos:
            ids.append(self.special["<|end_of_text|>"])
        return ids

    def decode(self, ids):
        return self.model.decode(ids)

def assign(left, right, tensor_name="unknown"):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

    with torch.no_grad():
        if isinstance(right, torch.Tensor):
            left.copy_(right)
        else:
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))
    return left


def load_weights_into_llama(model, param_config, params):
    model.tok_emb.weight = assign(model.tok_emb.weight, params["tok_embeddings.weight"], "tok_embeddings.weight")

    for l in range(param_config["n_layers"]):
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"layers.{l}.attention.wq.weight"],
            f"layers.{l}.attention.wq.weight"
        )
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"layers.{l}.attention.wk.weight"],
            f"layers.{l}.attention.wk.weight"
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"layers.{l}.attention.wv.weight"],
            f"layers.{l}.attention.wv.weight"
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"layers.{l}.attention.wo.weight"],
            f"layers.{l}.attention.wo.weight"
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"layers.{l}.attention_norm.weight"],
            f"layers.{l}.attention_norm.weight"
        )
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"layers.{l}.feed_forward.w1.weight"],
            f"layers.{l}.feed_forward.w1.weight"
        )
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"layers.{l}.feed_forward.w3.weight"],
            f"layers.{l}.feed_forward.w3.weight"
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"layers.{l}.feed_forward.w2.weight"],
            f"layers.{l}.feed_forward.w2.weight"
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"layers.{l}.ffn_norm.weight"],
            f"layers.{l}.ffn_norm.weight"
        )

    model.final_norm.weight = assign(model.final_norm.weight, params["norm.weight"], "norm.weight")

    if "output.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["output.weight"], "output.weight")
    else:
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")


LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,
    "context_length": 131_072,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 16,
    "hidden_dim": 8192,
    "n_kv_groups": 8,
    "rope_base": 500_000.0,
    "dtype": torch.bfloat16,
    "rope_freq": {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

LLAMA32_CONFIG_3B = {
    "vocab_size": 128_256,          
    "context_length": 131_072,       
    "emb_dim": 3072,                 
    "n_heads": 24,                   
    "n_layers": 28,                  
    "hidden_dim": 8192,              
    "n_kv_groups": 8,                
    "rope_base": 500_000.0,          
    "dtype": torch.bfloat16,         
    "rope_freq": {                   
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}

LLAMA31_CONFIG_8B = {
    "vocab_size": 128_256,
    "context_length": 131_072,
    "emb_dim": 4096,
    "n_heads": 32,
    "n_layers": 32,
    "hidden_dim": 14_336,
    "n_kv_groups": 8,
    "rope_base": 500_000.0,
    "dtype": torch.bfloat16,
    "rope_freq": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }
}
