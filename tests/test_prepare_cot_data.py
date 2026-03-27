"""
Tests for load_and_process_sft_data from data_prep/prepare_cot_data.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
from data_prep.prepare_cot_data import (
    load_and_process_sft_data,
    format_chat_tokens,
    THINK_START_ID,
    THINK_END_ID,
    ANSWER_START_ID,
    ANSWER_END_ID,
)
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
        "input": "What is 2+2?",
        "think": "2 plus 2 equals 4.",
        "output": "4",
        "source": "databricks_thinking",
    },
    {
        "input": "What is the capital of France?",
        "think": "France is a country in Europe. Its capital is Paris.",
        "output": "Paris",
        "source": "databricks_thinking",
    },
    {
        "input": "Should be filtered out",
        "think": "This row has a different source.",
        "output": "Ignored",
        "source": "other_source",
    },
]


class TestFormatChatTokens:
    """Tests for format_chat_tokens (the tokenization helper)."""

    def setup_method(self):
        self.tokenizer = Tokenizer(TOKENIZER_PATH)

    def test_returns_tuple(self):
        tokens, prompt_len = format_chat_tokens(
            self.tokenizer, "Hello", "thinking", "answer"
        )
        assert isinstance(tokens, list)
        assert isinstance(prompt_len, int)

    def test_prompt_len_less_than_total(self):
        tokens, prompt_len = format_chat_tokens(
            self.tokenizer, "Hello", "thinking", "answer"
        )
        assert 0 < prompt_len < len(tokens)

    def test_contains_reasoning_tokens(self):
        tokens, _ = format_chat_tokens(
            self.tokenizer, "Hello", "step by step", "42"
        )
        assert THINK_START_ID in tokens
        assert THINK_END_ID in tokens
        assert ANSWER_START_ID in tokens
        assert ANSWER_END_ID in tokens

    def test_reasoning_tokens_in_response_not_prompt(self):
        tokens, prompt_len = format_chat_tokens(
            self.tokenizer, "Hello", "step by step", "42"
        )
        prompt_tokens = tokens[:prompt_len]
        response_tokens = tokens[prompt_len:]
        assert THINK_START_ID not in prompt_tokens
        assert THINK_START_ID in response_tokens

    def test_starts_with_bos(self):
        tokens, _ = format_chat_tokens(
            self.tokenizer, "Hello", "thinking", "answer"
        )
        bos_id = self.tokenizer.special["<|begin_of_text|>"]
        assert tokens[0] == bos_id

    def test_ends_with_eot(self):
        tokens, _ = format_chat_tokens(
            self.tokenizer, "Hello", "thinking", "answer"
        )
        eot_id = self.tokenizer.special["<|eot_id|>"]
        assert tokens[-1] == eot_id

    def test_ordering_think_before_answer(self):
        tokens, _ = format_chat_tokens(
            self.tokenizer, "Hello", "reasoning here", "final answer"
        )
        think_pos = tokens.index(THINK_START_ID)
        answer_pos = tokens.index(ANSWER_START_ID)
        assert think_pos < answer_pos


class TestLoadAndProcessSftData:
    """Tests for load_and_process_sft_data (the full pipeline)."""

    def setup_method(self):
        self.tokenizer = Tokenizer(TOKENIZER_PATH)

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_filters_to_databricks_thinking(self, mock_load):
        mock_load.return_value = make_fake_dataset(SAMPLE_ROWS)
        result = load_and_process_sft_data(self.tokenizer, max_seq_len=2048)
        # Only 2 of 3 rows have source == "databricks_thinking"
        assert len(result) == 2

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_returns_list_of_tuples(self, mock_load):
        mock_load.return_value = make_fake_dataset(SAMPLE_ROWS)
        result = load_and_process_sft_data(self.tokenizer, max_seq_len=2048)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2
            tokens, prompt_len = item
            assert isinstance(tokens, list)
            assert isinstance(prompt_len, int)

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_respects_max_seq_len(self, mock_load):
        mock_load.return_value = make_fake_dataset(SAMPLE_ROWS)
        # Use a very short max_seq_len so all samples get skipped
        result = load_and_process_sft_data(self.tokenizer, max_seq_len=10)
        assert len(result) == 0

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_token_ids_are_valid_integers(self, mock_load):
        mock_load.return_value = make_fake_dataset(SAMPLE_ROWS)
        result = load_and_process_sft_data(self.tokenizer, max_seq_len=2048)
        for tokens, prompt_len in result:
            assert all(isinstance(t, int) for t in tokens)
            assert all(t >= 0 for t in tokens)

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_prompt_len_within_bounds(self, mock_load):
        mock_load.return_value = make_fake_dataset(SAMPLE_ROWS)
        result = load_and_process_sft_data(self.tokenizer, max_seq_len=2048)
        for tokens, prompt_len in result:
            assert 0 < prompt_len < len(tokens)

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_empty_dataset(self, mock_load):
        mock_load.return_value = make_fake_dataset([])
        result = load_and_process_sft_data(self.tokenizer, max_seq_len=2048)
        assert result == []

    @patch("data_prep.prepare_cot_data.load_dataset")
    def test_no_matching_source(self, mock_load):
        rows = [{"input": "x", "think": "y", "output": "z", "source": "other"}]
        mock_load.return_value = make_fake_dataset(rows)
        result = load_and_process_sft_data(self.tokenizer, max_seq_len=2048)
        assert result == []


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
