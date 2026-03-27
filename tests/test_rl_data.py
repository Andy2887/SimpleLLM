"""
Tests for load_and_process_rl_data from data_prep/prepare_cot_data.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from data_prep.prepare_cot_data import load_and_process_rl_data
from llama import Tokenizer

TOKENIZER_PATH = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/tokenizer.model")


def make_fake_dataset(rows):
    """Create a mock HuggingFace dataset from a list of dicts."""
    ds = MagicMock()
    ds.__iter__ = lambda self: iter(rows)
    ds.__len__ = lambda self: len(rows)
    ds.filter = lambda fn: make_fake_dataset([r for r in rows if fn(r)])
    return ds


SAMPLE_ROWS = [
    {
        "input": "What is 3 + 5?",
        "output": "8",
        "source": "gsm8k",
    },
    {
        "input": "If a train travels 60 miles per hour for 2 hours, how far does it go?",
        "output": "120 miles",
        "source": "gsm8k",
    },
    {
        "input": "This should be filtered out",
        "output": "Ignored",
        "source": "other_source",
    },
]


class TestLoadAndProcessRlData:
    """Tests for load_and_process_rl_data."""

    def setup_method(self):
        self.tokenizer = Tokenizer(TOKENIZER_PATH)

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_filters_to_gsm8k(self, mock_load):
        mock_load.return_value = make_fake_dataset(SAMPLE_ROWS)
        result = load_and_process_rl_data(self.tokenizer, max_seq_len=2048)
        # Only 2 of 3 rows have source == "gsm8k"
        assert len(result) == 2

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_returns_list_of_dicts(self, mock_load):
        mock_load.return_value = make_fake_dataset(SAMPLE_ROWS)
        result = load_and_process_rl_data(self.tokenizer, max_seq_len=2048)
        for item in result:
            assert isinstance(item, dict)
            assert "prompt_tokens" in item
            assert "answer" in item

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_prompt_tokens_are_valid_ints(self, mock_load):
        mock_load.return_value = make_fake_dataset(SAMPLE_ROWS)
        result = load_and_process_rl_data(self.tokenizer, max_seq_len=2048)
        for item in result:
            assert isinstance(item["prompt_tokens"], list)
            assert all(isinstance(t, int) for t in item["prompt_tokens"])
            assert all(t >= 0 for t in item["prompt_tokens"])

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_respects_max_seq_len(self, mock_load):
        mock_load.return_value = make_fake_dataset(SAMPLE_ROWS)
        # Very short max_seq_len — all prompts should be skipped
        result = load_and_process_rl_data(self.tokenizer, max_seq_len=5)
        assert len(result) == 0

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_empty_dataset(self, mock_load):
        mock_load.return_value = make_fake_dataset([])
        result = load_and_process_rl_data(self.tokenizer, max_seq_len=2048)
        assert result == []

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_no_matching_source(self, mock_load):
        rows = [{"input": "x", "output": "y", "source": "not_gsm8k"}]
        mock_load.return_value = make_fake_dataset(rows)
        result = load_and_process_rl_data(self.tokenizer, max_seq_len=2048)
        assert result == []

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_answer_field_matches_output(self, mock_load):
        mock_load.return_value = make_fake_dataset(SAMPLE_ROWS)
        result = load_and_process_rl_data(self.tokenizer, max_seq_len=2048)
        assert result[0]["answer"] == "8"
        assert result[1]["answer"] == "120 miles"

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_prompt_tokens_start_with_bos(self, mock_load):
        mock_load.return_value = make_fake_dataset(SAMPLE_ROWS)
        result = load_and_process_rl_data(self.tokenizer, max_seq_len=2048)
        bos_id = self.tokenizer.special["<|begin_of_text|>"]
        for item in result:
            assert item["prompt_tokens"][0] == bos_id


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
