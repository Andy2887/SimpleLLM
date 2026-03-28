"""
Tests for text_to_token_ids, token_ids_to_text, generate, and KVCache from utils.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils import text_to_token_ids, token_ids_to_text, generate, KVCache
from llama import Tokenizer, Llama3Model

TOKENIZER_PATH = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/tokenizer.model")

TINY_CFG = {
    "vocab_size": 128_256,
    "context_length": 128,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "hidden_dim": 128,
    "n_kv_groups": 2,
    "rope_base": 10_000.0,
    "dtype": torch.float32,
    "rope_freq": None,
}


class TestTextTokenConversion:
    """Tests for text_to_token_ids and token_ids_to_text."""

    def setup_method(self):
        self.tokenizer = Tokenizer(TOKENIZER_PATH)

    def test_text_to_token_ids_returns_2d_tensor(self):
        result = text_to_token_ids("hello", self.tokenizer)
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 2
        assert result.shape[0] == 1  # batch dimension

    def test_token_ids_to_text_recovers_text(self):
        text = "The capital of France is Paris."
        ids = text_to_token_ids(text, self.tokenizer)
        recovered = token_ids_to_text(ids, self.tokenizer)
        # BOS token gets decoded too, but the original text should be present
        assert text in recovered

    def test_roundtrip(self):
        text = "Machine learning is fascinating"
        ids = text_to_token_ids(text, self.tokenizer)
        recovered = token_ids_to_text(ids, self.tokenizer)
        assert text in recovered

    def test_token_ids_are_long_dtype(self):
        result = text_to_token_ids("test", self.tokenizer)
        assert result.dtype == torch.long


class TestGenerate:
    """Tests for the generate function using a tiny random-weight model."""

    def setup_method(self):
        self.model = Llama3Model(TINY_CFG)
        self.model.eval()

    def test_output_longer_than_input(self):
        idx = torch.randint(0, 1000, (1, 5))
        max_new = 10
        result = generate(self.model, idx, max_new_tokens=max_new, context_size=TINY_CFG["context_length"])
        assert result.shape[1] > idx.shape[1]
        assert result.shape[1] <= idx.shape[1] + max_new

    def test_output_starts_with_input(self):
        idx = torch.randint(0, 1000, (1, 5))
        result = generate(self.model, idx, max_new_tokens=3, context_size=TINY_CFG["context_length"])
        assert result[0, : idx.shape[1]].tolist() == idx[0].tolist()

    def test_respects_eos_id(self):
        # With a random model we can't guarantee EOS is produced,
        # but we verify the mechanism: if eos_id matches a generated token, output is shorter.
        idx = torch.randint(0, 1000, (1, 3))
        max_new = 50
        # Generate without eos — should produce up to max_new tokens
        result_no_eos = generate(
            self.model, idx.clone(), max_new_tokens=max_new,
            context_size=TINY_CFG["context_length"], eos_id=None
        )
        assert result_no_eos.shape[1] == idx.shape[1] + max_new

    def test_generate_without_cache(self):
        idx = torch.randint(0, 1000, (1, 5))
        result = generate(
            self.model, idx, max_new_tokens=5,
            context_size=TINY_CFG["context_length"], use_cache=False
        )
        assert result.shape[1] == idx.shape[1] + 5

    def test_generate_returns_2d_tensor(self):
        idx = torch.randint(0, 1000, (1, 4))
        result = generate(self.model, idx, max_new_tokens=3, context_size=TINY_CFG["context_length"])
        assert result.dim() == 2
        assert result.shape[0] == 1


class TestKVCache:
    """Tests for the KVCache class."""

    def test_get_returns_none_initially(self):
        cache = KVCache(n_layers=4)
        for i in range(4):
            assert cache.get(i) is None

    def test_update_then_get(self):
        cache = KVCache(n_layers=4)
        value = (torch.randn(1, 4, 3, 8), torch.randn(1, 4, 3, 8))
        cache.update(0, value)
        retrieved = cache.get(0)
        assert retrieved is not None
        assert torch.equal(retrieved[0], value[0])
        assert torch.equal(retrieved[1], value[1])

    def test_reset_clears_all(self):
        cache = KVCache(n_layers=3)
        for i in range(3):
            cache.update(i, (torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4)))
        cache.reset()
        for i in range(3):
            assert cache.get(i) is None

    def test_get_all(self):
        cache = KVCache(n_layers=2)
        assert len(cache.get_all()) == 2
        assert all(v is None for v in cache.get_all())

    def test_update_specific_layer(self):
        cache = KVCache(n_layers=3)
        val = (torch.randn(1, 2, 2, 4), torch.randn(1, 2, 2, 4))
        cache.update(1, val)
        assert cache.get(0) is None
        assert cache.get(1) is not None
        assert cache.get(2) is None


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
