"""
Tests for Tokenizer and gradient checkpointing from llama.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from llama import Tokenizer, Llama3Model

TOKENIZER_PATH = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/tokenizer.model")


class TestTokenizer:
    """Tests for the Tokenizer class."""

    def setup_method(self):
        self.tokenizer = Tokenizer(TOKENIZER_PATH)

    def test_encode_decode_roundtrip(self):
        text = "Hello, world! This is a test."
        ids = self.tokenizer.encode(text, bos=False)
        decoded = self.tokenizer.decode(ids)
        assert decoded == text

    def test_special_token_ids(self):
        assert self.tokenizer.special["<|begin_of_text|>"] == 128000
        assert self.tokenizer.special["<|end_of_text|>"] == 128001
        assert self.tokenizer.special["<think>"] == 128002
        assert self.tokenizer.special["</think>"] == 128003
        assert self.tokenizer.special["<answer>"] == 128004
        assert self.tokenizer.special["</answer>"] == 128005

    def test_encode_with_bos(self):
        ids_with_bos = self.tokenizer.encode("hi", bos=True)
        ids_without_bos = self.tokenizer.encode("hi", bos=False)
        assert ids_with_bos[0] == 128000
        assert ids_with_bos[1:] == ids_without_bos

    def test_encode_without_bos(self):
        ids = self.tokenizer.encode("hi", bos=False)
        assert ids[0] != 128000

    def test_encode_with_eos(self):
        ids = self.tokenizer.encode("hi", bos=False, eos=True)
        assert ids[-1] == 128001  # <|end_of_text|>

    def test_encode_without_eos(self):
        ids = self.tokenizer.encode("hi", bos=False, eos=False)
        assert ids[-1] != 128001

    def test_encode_empty_string(self):
        ids = self.tokenizer.encode("", bos=False, eos=False)
        assert ids == []

    def test_encode_empty_string_with_bos(self):
        ids = self.tokenizer.encode("", bos=True, eos=False)
        assert ids == [128000]

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Tokenizer("/nonexistent/path/tokenizer.model")

    def test_encode_returns_list_of_ints(self):
        ids = self.tokenizer.encode("test", bos=True)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)

    def test_roundtrip_with_bos_stripped(self):
        text = "The quick brown fox"
        ids = self.tokenizer.encode(text, bos=True)
        # Strip BOS before decoding
        decoded = self.tokenizer.decode(ids[1:])
        assert decoded == text


class TestGradientCheckpointing:
    """Tests for gradient checkpointing in Llama3Model."""

    TINY_CFG = {
        "vocab_size": 256,
        "context_length": 64,
        "emb_dim": 64,
        "n_heads": 4,
        "n_layers": 2,
        "hidden_dim": 128,
        "n_kv_groups": 2,
        "rope_base": 10_000.0,
        "dtype": torch.float32,
        "rope_freq": None,
    }

    def test_gradient_checkpointing_default_off(self):
        model = Llama3Model(self.TINY_CFG)
        assert model.gradient_checkpointing is False

    def test_gradient_checkpointing_toggle(self):
        model = Llama3Model(self.TINY_CFG)
        model.gradient_checkpointing = True
        assert model.gradient_checkpointing is True

    def test_forward_with_checkpointing(self):
        model = Llama3Model(self.TINY_CFG)
        model.gradient_checkpointing = True
        model.train()
        x = torch.randint(0, 256, (1, 8))
        logits = model(x)
        assert logits.shape == (1, 8, 256)

    def test_forward_without_checkpointing(self):
        model = Llama3Model(self.TINY_CFG)
        model.gradient_checkpointing = False
        x = torch.randint(0, 256, (1, 8))
        logits = model(x)
        assert logits.shape == (1, 8, 256)

    def test_checkpointing_produces_same_output(self):
        model = Llama3Model(self.TINY_CFG)
        model.train()
        x = torch.randint(0, 256, (1, 8))

        model.gradient_checkpointing = False
        logits_no_ckpt = model(x)

        model.gradient_checkpointing = True
        logits_ckpt = model(x)

        assert torch.allclose(logits_no_ckpt, logits_ckpt, atol=1e-5)

    def test_gradients_match_without_checkpointing(self):
        """Gradients with checkpointing should match gradients without it."""
        torch.manual_seed(42)
        model = Llama3Model(self.TINY_CFG)
        x = torch.randint(0, 256, (1, 8))
        targets = torch.randint(0, 256, (1, 8))
        loss_fn = torch.nn.CrossEntropyLoss()

        # Without checkpointing
        model.gradient_checkpointing = False
        model.train()
        model.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        grads_no_ckpt = {
            name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
        }

        # With checkpointing
        model.gradient_checkpointing = True
        model.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        grads_ckpt = {
            name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
        }

        assert grads_no_ckpt.keys() == grads_ckpt.keys()
        for name in grads_no_ckpt:
            assert torch.allclose(grads_no_ckpt[name], grads_ckpt[name], atol=1e-5), \
                f"Gradient mismatch in {name}"

    def test_backward_pass_succeeds(self):
        """Full backward pass with checkpointing should not error."""
        model = Llama3Model(self.TINY_CFG)
        model.gradient_checkpointing = True
        model.train()
        x = torch.randint(0, 256, (1, 8))
        targets = torch.randint(0, 256, (1, 8))

        logits = model(x)
        loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()

        # Every parameter should have a gradient
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"

    def test_optimizer_step_with_checkpointing(self):
        """Simulate a full SFT training step: forward, loss, backward, optimizer.step()."""
        model = Llama3Model(self.TINY_CFG)
        model.gradient_checkpointing = True
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        x = torch.randint(0, 256, (2, 16))  # batch_size=2
        targets = torch.randint(0, 256, (2, 16))

        # Capture weight before step
        param = next(model.parameters())
        weight_before = param.data.clone()

        logits = model(x)
        loss = torch.nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

        # Weights should have changed
        assert not torch.equal(param.data, weight_before), "Weights did not update after optimizer step"

    def test_checkpointing_disabled_during_eval(self):
        """Checkpointing should only activate in training mode, not eval."""
        model = Llama3Model(self.TINY_CFG)
        model.gradient_checkpointing = True
        model.eval()
        x = torch.randint(0, 256, (1, 8))
        # Should run without error even though checkpointing is flagged on
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 8, 256)

    def test_checkpointing_skipped_with_kv_cache(self):
        """When cache is provided, checkpointing should be skipped (per the guard in forward)."""
        from utils import KVCache
        model = Llama3Model(self.TINY_CFG)
        model.gradient_checkpointing = True
        model.train()

        cache = KVCache(self.TINY_CFG["n_layers"])
        x = torch.randint(0, 256, (1, 8))
        # Forward with cache — should not error (checkpointing guard: cache is None)
        logits = model(x, cache=cache)
        assert logits.shape == (1, 8, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
