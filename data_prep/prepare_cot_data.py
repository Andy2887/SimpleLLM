"""
Phase 0.2: Data Processing Pipeline

Loads the CoT dataset from HuggingFace, formats it into Llama 3.2 chat template,
and tokenizes with reasoning tokens mapped to reserved slots.

Token mapping:
    <think>   -> <|reserved_0|> (128002)
    </think>  -> <|reserved_1|> (128003)
    <answer>  -> <|reserved_2|> (128004)
    </answer> -> <|reserved_3|> (128005)
"""

import os
from datasets import load_dataset

# Reasoning token IDs (mapped to reserved slots in Llama 3 vocabulary)
THINK_START_ID = 128002   # <think>
THINK_END_ID = 128003     # </think>
ANSWER_START_ID = 128004  # <answer>
ANSWER_END_ID = 128005    # </answer>

SYSTEM_MESSAGE = "You are a helpful assistant that thinks step by step."


def format_chat_tokens(tokenizer, input_text, think_text, answer_text):
    """
    Format a single CoT example into token IDs using Llama 3.2 chat template.

    Template:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
        {system_message}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>\n\n
        {input}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>\n\n
        <think>\n{think}\n</think>\n<answer>\n{answer}\n</answer><|eot_id|>

    Returns:
        tokens: list[int] - full token sequence
        prompt_len: int - number of prompt tokens (for loss masking)
    """
    sp = tokenizer.special
    enc = tokenizer.model.encode

    BOS = sp["<|begin_of_text|>"]
    START_HEADER = sp["<|start_header_id|>"]
    END_HEADER = sp["<|end_header_id|>"]
    EOT = sp["<|eot_id|>"]

    # --- Prompt tokens (system + user + assistant header) ---
    prompt_tokens = [BOS]

    # System message
    prompt_tokens += [START_HEADER] + enc("system") + [END_HEADER]
    prompt_tokens += enc("\n\n" + SYSTEM_MESSAGE)
    prompt_tokens += [EOT]

    # User message
    prompt_tokens += [START_HEADER] + enc("user") + [END_HEADER]
    prompt_tokens += enc("\n\n" + input_text)
    prompt_tokens += [EOT]

    # Assistant header
    prompt_tokens += [START_HEADER] + enc("assistant") + [END_HEADER]
    prompt_tokens += enc("\n\n")

    # --- Response tokens (<think>...<answer>...) ---
    response_tokens = [THINK_START_ID]
    response_tokens += enc("\n" + think_text + "\n")
    response_tokens += [THINK_END_ID]
    response_tokens += enc("\n")
    response_tokens += [ANSWER_START_ID]
    response_tokens += enc("\n" + answer_text + "\n")
    response_tokens += [ANSWER_END_ID]
    response_tokens += [EOT]

    return prompt_tokens + response_tokens, len(prompt_tokens)


def load_and_process_sft_data(tokenizer, max_seq_len=2048):
    """
    Load the SFT split, filter to databricks_thinking source,
    tokenize into Llama 3.2 chat format.

    Args:
        tokenizer: Tokenizer instance (from llama.py)
        max_seq_len: maximum sequence length (samples exceeding this are skipped)

    Returns:
        list of (token_ids: list[int], prompt_len: int) tuples
    """
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("comoZ/reasoning-dataset", split="sft")

    # Filter to databricks_thinking source
    ds = ds.filter(lambda x: x["source"] == "databricks_thinking")
    print(f"Filtered to {len(ds)} databricks_thinking examples")

    processed = []
    skipped = 0

    for row in ds:
        tokens, prompt_len = format_chat_tokens(
            tokenizer, row["input"], row["think"], row["output"]
        )
        if len(tokens) > max_seq_len:
            skipped += 1
            continue
        processed.append((tokens, prompt_len))

    print(f"Processed {len(processed)} examples ({skipped} skipped, exceeded {max_seq_len} tokens)")
    return processed


def load_and_process_rl_data(tokenizer, max_seq_len=2048):
    """
    Load the RL split, filter to gsm8k source.
    Returns prompts and ground-truth answers for GRPO (Phase 2).

    Args:
        tokenizer: Tokenizer instance (from llama.py)
        max_seq_len: maximum prompt length

    Returns:
        list of dicts with 'prompt_tokens' (list[int]) and 'answer' (str) keys
    """
    print("Loading dataset from HuggingFace...")
    ds = load_dataset("comoZ/reasoning-dataset", split="rl")

    # Filter to gsm8k source
    ds = ds.filter(lambda x: x["source"] == "gsm8k")
    print(f"Filtered to {len(ds)} gsm8k examples")

    sp = tokenizer.special
    enc = tokenizer.model.encode

    processed = []
    for row in ds:
        # Build prompt tokens only (no response — the model generates its own)
        prompt_tokens = [sp["<|begin_of_text|>"]]
        prompt_tokens += [sp["<|start_header_id|>"]] + enc("system") + [sp["<|end_header_id|>"]]
        prompt_tokens += enc("\n\n" + SYSTEM_MESSAGE)
        prompt_tokens += [sp["<|eot_id|>"]]
        prompt_tokens += [sp["<|start_header_id|>"]] + enc("user") + [sp["<|end_header_id|>"]]
        prompt_tokens += enc("\n\n" + row["input"])
        prompt_tokens += [sp["<|eot_id|>"]]
        prompt_tokens += [sp["<|start_header_id|>"]] + enc("assistant") + [sp["<|end_header_id|>"]]
        prompt_tokens += enc("\n\n")

        if len(prompt_tokens) > max_seq_len:
            continue

        processed.append({
            "prompt_tokens": prompt_tokens,
            "answer": row["output"],
        })

    print(f"Processed {len(processed)} RL examples")
    return processed


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from llama import Tokenizer

    tokenizer_path = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/tokenizer.model")
    tokenizer = Tokenizer(tokenizer_path)

    print("=== SFT Data ===")
    sft_data = load_and_process_sft_data(tokenizer)
    if sft_data:
        tokens, prompt_len = sft_data[0]
        print(f"First example: {len(tokens)} tokens, prompt_len={prompt_len}")
        print(f"Decoded prompt: {tokenizer.decode(tokens[:prompt_len])}")
        print(f"Decoded response: {tokenizer.decode(tokens[prompt_len:])}")

    print("\n=== RL Data ===")
    rl_data = load_and_process_rl_data(tokenizer)
    if rl_data:
        print(f"First prompt: {len(rl_data[0]['prompt_tokens'])} tokens")
        print(f"First answer: {rl_data[0]['answer']}")
