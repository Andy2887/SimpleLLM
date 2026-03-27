"""
Tests for CoTDataset and cot_collate_fn from cot_dataset.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from data_prep.cot_dataset import CoTDataset, cot_collate_fn


# Helper: fake tokenized data as (token_ids, prompt_len) tuples
def make_sample(total_len, prompt_len):
    """Create a fake (token_ids, prompt_len) tuple with sequential token IDs."""
    tokens = list(range(1, total_len + 1))
    return (tokens, prompt_len)


class TestCoTDataset:
    """Tests for the CoTDataset class."""

    def test_len(self):
        data = [make_sample(10, 4), make_sample(12, 5), make_sample(8, 3)]
        ds = CoTDataset(data, max_seq_len=20)
        assert len(ds) == 3

    def test_getitem_returns_tensors(self):
        data = [make_sample(10, 4)]
        ds = CoTDataset(data, max_seq_len=20)
        input_ids, target_ids = ds[0]
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(target_ids, torch.Tensor)
        assert input_ids.dtype == torch.long
        assert target_ids.dtype == torch.long

    def test_input_target_shifted_by_one(self):
        tokens = [10, 20, 30, 40, 50]
        data = [(tokens, 2)]
        ds = CoTDataset(data, max_seq_len=20)
        input_ids, target_ids = ds[0]
        # input_ids = tokens[:-1], target_ids = tokens[1:]
        assert input_ids.tolist() == [10, 20, 30, 40]
        assert target_ids[2:].tolist() == [40, 50]  # unmasked portion from tokens[1:]

    def test_prompt_masking(self):
        tokens = list(range(1, 11))  # [1, 2, ..., 10]
        prompt_len = 5
        data = [(tokens, prompt_len)]
        ds = CoTDataset(data, max_seq_len=20)
        _, target_ids = ds[0]
        # target_ids[:prompt_len-1] should all be -100
        assert (target_ids[: prompt_len - 1] == -100).all()
        # positions from prompt_len-1 onward should be real token IDs (not -100)
        assert (target_ids[prompt_len - 1 :] != -100).all()

    def test_samples_exceeding_max_seq_len_filtered(self):
        data = [make_sample(10, 4), make_sample(25, 8), make_sample(8, 3)]
        ds = CoTDataset(data, max_seq_len=15)
        assert len(ds) == 2  # the 25-token sample is filtered out

    def test_all_filtered_gives_empty(self):
        data = [make_sample(100, 30)]
        ds = CoTDataset(data, max_seq_len=10)
        assert len(ds) == 0

    def test_sequence_lengths(self):
        tokens = list(range(1, 11))  # 10 tokens
        data = [(tokens, 3)]
        ds = CoTDataset(data, max_seq_len=20)
        input_ids, target_ids = ds[0]
        assert len(input_ids) == 9  # tokens[:-1]
        assert len(target_ids) == 9  # tokens[1:]


class TestCotCollateFn:
    """Tests for the cot_collate_fn function."""

    def test_single_sample_batch(self):
        input_ids = torch.tensor([1, 2, 3], dtype=torch.long)
        target_ids = torch.tensor([2, 3, 4], dtype=torch.long)
        batch = [(input_ids, target_ids)]
        inputs, targets = cot_collate_fn(batch)
        assert inputs.shape == (1, 3)
        assert targets.shape == (1, 3)
        assert inputs[0].tolist() == [1, 2, 3]
        assert targets[0].tolist() == [2, 3, 4]

    def test_padding_different_lengths(self):
        batch = [
            (torch.tensor([1, 2, 3], dtype=torch.long), torch.tensor([2, 3, 4], dtype=torch.long)),
            (torch.tensor([5, 6], dtype=torch.long), torch.tensor([6, 7], dtype=torch.long)),
        ]
        inputs, targets = cot_collate_fn(batch)
        assert inputs.shape == (2, 3)
        assert targets.shape == (2, 3)

    def test_input_padding_uses_pad_token_id(self):
        batch = [
            (torch.tensor([1, 2, 3], dtype=torch.long), torch.tensor([2, 3, 4], dtype=torch.long)),
            (torch.tensor([5, 6], dtype=torch.long), torch.tensor([6, 7], dtype=torch.long)),
        ]
        inputs, targets = cot_collate_fn(batch, pad_token_id=128001)
        # Second sample should be padded with 128001
        assert inputs[1, 2].item() == 128001

    def test_target_padding_uses_ignore_index(self):
        batch = [
            (torch.tensor([1, 2, 3], dtype=torch.long), torch.tensor([2, 3, 4], dtype=torch.long)),
            (torch.tensor([5, 6], dtype=torch.long), torch.tensor([6, 7], dtype=torch.long)),
        ]
        inputs, targets = cot_collate_fn(batch, ignore_index=-100)
        # Second sample target should be padded with -100
        assert targets[1, 2].item() == -100

    def test_output_shapes(self):
        batch = [
            (torch.tensor([1, 2, 3, 4], dtype=torch.long), torch.tensor([2, 3, 4, 5], dtype=torch.long)),
            (torch.tensor([10, 20], dtype=torch.long), torch.tensor([20, 30], dtype=torch.long)),
            (torch.tensor([7, 8, 9], dtype=torch.long), torch.tensor([8, 9, 10], dtype=torch.long)),
        ]
        inputs, targets = cot_collate_fn(batch)
        assert inputs.shape == (3, 4)
        assert targets.shape == (3, 4)

    def test_no_padding_when_same_length(self):
        batch = [
            (torch.tensor([1, 2, 3], dtype=torch.long), torch.tensor([2, 3, 4], dtype=torch.long)),
            (torch.tensor([5, 6, 7], dtype=torch.long), torch.tensor([6, 7, 8], dtype=torch.long)),
        ]
        inputs, targets = cot_collate_fn(batch, pad_token_id=128001, ignore_index=-100)
        # No padding needed — no pad tokens should appear
        assert (inputs != 128001).all()
        assert (targets != -100).all()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
