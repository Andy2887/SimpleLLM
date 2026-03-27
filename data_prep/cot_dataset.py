"""
Phase 0.3: Dataset Class for Chain-of-Thought Fine-Tuning

CoTDataset returns (input_ids, target_ids) pairs with prompt tokens masked
in target_ids so the loss is only computed on the assistant's response
(the <think>...</think><answer>...</answer> portion).
"""

import torch
from torch.utils.data import Dataset


class CoTDataset(Dataset):
    """
    Dataset for CoT fine-tuning.

    Each sample is a (input_ids, target_ids) pair where:
    - input_ids = tokens[:-1]  (model input)
    - target_ids = tokens[1:]  (prediction target, with prompt positions masked to -100)
    """

    def __init__(self, data, max_seq_len=2048):
        """
        Args:
            data: list of (token_ids: list[int], prompt_len: int) tuples
                  from prepare_cot_data.load_and_process_sft_data
            max_seq_len: maximum sequence length (samples exceeding this are skipped)
        """
        self.samples = []
        for tokens, prompt_len in data:
            if len(tokens) <= max_seq_len:
                self.samples.append((tokens, prompt_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, prompt_len = self.samples[idx]

        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)

        # Mask prompt tokens so loss is only computed on the response.
        # tokens[0:prompt_len] are the prompt. In the shifted target_ids,
        # positions 0 through prompt_len-2 predict the next prompt token,
        # so we mask those. Position prompt_len-1 predicts the first
        # response token — that's the first position we train on.
        target_ids[:prompt_len - 1] = -100

        return input_ids, target_ids


def cot_collate_fn(batch, pad_token_id=128001, ignore_index=-100):
    """
    Dynamic padding collate function for CoT data.
    Pads each batch to the longest sequence length.

    Args:
        batch: list of (input_ids, target_ids) tuples
        pad_token_id: token used to pad inputs (128001 = <|end_of_text|>)
        ignore_index: value used to pad targets (ignored by cross-entropy loss)
    """
    input_ids_list, target_ids_list = zip(*batch)
    max_len = max(len(ids) for ids in input_ids_list)

    inputs_padded = []
    targets_padded = []

    for input_ids, target_ids in zip(input_ids_list, target_ids_list):
        pad_len = max_len - len(input_ids)
        padded_input = torch.cat([
            input_ids,
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ])
        padded_target = torch.cat([
            target_ids,
            torch.full((pad_len,), ignore_index, dtype=torch.long)
        ])
        inputs_padded.append(padded_input)
        targets_padded.append(padded_target)

    return torch.stack(inputs_padded), torch.stack(targets_padded)
