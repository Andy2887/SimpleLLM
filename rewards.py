"""
Phase 2.1: Reward Functions for GRPO

Correctness reward: +1 if response contains <answer>...</answer> with the ground truth, 0 otherwise.
Answers don't need to be exact — as long as the <answer> field contains the ground truth, it counts.
"""

import re


def correctness_reward(response_text, ground_truth):
    """
    +1 if <answer>...</answer> contains the ground truth (string containment
    or numerical equivalence), 0 otherwise.
    """
    match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
    if match is None:
        return 0.0

    answer_content = match.group(1).strip()
    gt = ground_truth.strip()

    # String containment (e.g. "Andy's age is 15" matches ground truth "15")
    if gt in answer_content:
        return 1.0

    # Numerical equivalence (e.g. "15.0" matches "15")
    try:
        gt_num = float(gt)
        numbers = re.findall(r'-?\d+\.?\d*', answer_content)
        for num_str in numbers:
            if float(num_str) == gt_num:
                return 1.0
    except ValueError:
        pass

    return 0.0
