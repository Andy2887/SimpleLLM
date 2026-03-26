# Plan: Enable Reasoning for Llama 3.2 1B via Full Fine-Tuning SFT + GRPO

## Goal

Transform the Llama 3.2 1B model into a reasoning model that can solve logic puzzles by:
1. **SFT (Supervised Fine-Tuning)** with **full fine-tuning** on chain-of-thought data with `<think>` and `<answer>` structured output
2. **RL (GRPO — Group Relative Policy Optimization)** to further refine reasoning quality through reward-based optimization

---

## Phase 0: Data Preparation

### 0.1 — Chain-of-Thought Dataset

Use a CoT dataset and create samples in this format:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant that thinks step by step.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{input}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
<think>
{think}
</think>
<answer>
{output}
</answer><|eot_id|>
```

**Data source:** [comoZ/reasoning-dataset](https://huggingface.co/datasets/comoZ/reasoning-dataset)

This dataset has two split: 'sft' and 'fl'. Here's an example of one row:

```
{
   "input":"From Monday to Friday, Elle practices piano for 30 minutes. On Saturday, she practices piano three times as much as on a weekday. There is no practice on Sunday.  How many hours does Elle spend practicing piano each week?",
   "output":"4",
   "think":"From Monday to Friday, Elle practices 0.50 x 5 = 2.5 hours.\nOn Saturday, she practices 0.50 x 3 = 1.5 hours.\nEach week, Elle practices piano for 2.5 + 1.5 = 4 hours.",
   "source":"gsm8k",
   "type":"math",
   "task_type":"Math",
   "rubrics":"1. Correctly identifies the practice times for weekdays (30 minutes each), Saturday (3 * 30 minutes), and Sunday (0 minutes). Explicitly states the week consists of 5 weekdays.\n2. Accurately calculates the total practice time in minutes: (5 days * 30 minutes/day) + (3 * 30 minutes) + 0 minutes. Demonstrates correct multiplication and addition to find the total minutes.\n3. Correctly converts the total practice time from minutes to hours by dividing by 60.  The final answer is a reasonable value given the practice schedule (less than 5 hours)."
}
```

You could look at `data_prep/check_data.py` for more information about how to access data from this dataset.

For SFT, we will use the 'sft' split. We will only train on the rows where "source: databricks_thinking".

For RL, we will use the 'rl' split. We will only train on the the rows where "source: gsm8k".

### 0.2 — Data Processing Pipeline

Create `data_prep/prepare_cot_data.py`:
- Load raw dataset (Hugging Face datasets)
- Format into the Llama 3.2 chat template using the existing `Tokenizer` class
- Map `<think>`, `</think>`, `<answer>`, `</answer>` to existing reserved token slots in the tokenizer (`<|reserved_0|>` → 128002, `<|reserved_1|>` → 128003, `<|reserved_2|>` → 128004, `<|reserved_3|>` → 128005). No vocab resize needed — these embedding rows already exist in the pretrained weights.
- 100% training set. No validation set.

### 0.3 — Dataset Class

Create `cot_dataset.py`:
- `CoTDataset(torch.utils.data.Dataset)` — returns `(input_ids, target_ids)` pairs
- Mask the prompt tokens in `target_ids` with `-100` (only compute loss on the assistant's response, i.e., the `<think>...</think><answer>...</answer>` portion)
- Dynamic padding collate function (adapt from existing `custom_collate_fn` in `gpt_instruction_finetuning.py`)
- Max sequence length: 2048 tokens (configurable; balance between context and memory)

---

## Phase 1: Supervised Fine-Tuning (SFT) — Full Fine-Tuning

### 1.1 — SFT Training Script

Create `sft_reasoning.py`:

**Training Config:**
```python
sft_config = {
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.1,
    "warmup_steps": 100,
    "max_seq_len": 1024,
    "grad_clip": 1.0,
}
```

**Training Loop:**
- Train all model parameters (full fine-tuning)
- Optimizer: AdamW (over all parameters)
- LR schedule: linear warmup → cosine decay
- Loss: cross-entropy with `ignore_index=-100` (masked prompt tokens)
- Gradient clipping: max norm 1.0
- Gradient accumulation to simulate larger batch sizes
- Validation loss every N steps
- Save full model checkpoint at end of each epoch

---

### 1.2 — GPU Memory Estimate (GRPO with Full Fine-Tuning)

| Component | Calculation | Memory |
|---|---|---|
| Model weights (BF16) | 1B × 2 bytes | ~2 GB |
| Gradients (BF16) | 1B × 2 bytes | ~2 GB |
| Optimizer states (AdamW FP32) | 1B × 12 bytes | ~12 GB |
| Activations (batch=16, seq~1024) | varies | ~5 GB |
| Stored log-probs (policy + ref) | negligible | ~0.01 GB |
| **Total** | | **~20 - 25 GB** |

---

## Phase 2: Reinforcement Learning (GRPO)

### 2.1 — Reward Functions

Create `rewards.py` with composable reward signals:

| Reward Component    | Weight | Description |
|---------------------|--------|-------------|
| **Format reward**   | 0.1    | +1 if output contains valid `<think>...</think><answer>...</answer>` structure, 0 otherwise |
| **Correctness reward** | 0.9 | +1 if extracted answer matches ground truth (exact match or numerical equivalence), 0 otherwise |

Primary signal is **correctness** — this is what drives the model to actually reason better.

### 2.2 — GPU Memory Estimate (GRPO with Full Fine-Tuning)

TODO

### 2.3 — GRPO Training Script

Create `grpo_reasoning.py`:

**GRPO Config:**
```python
grpo_config = {
    "epochs": 2,
    "group_size": 4,              # G completions per prompt
    "batch_size": 2,              # prompts per batch (generates G * batch_size completions)
    "learning_rate": 5e-7,        # much lower than SFT
    "kl_coeff": 0.05,             # KL penalty coefficient (beta)
    "clip_eps": 0.2,              # PPO-style clipping epsilon
    "max_gen_len": 1024,          # max tokens for generated completions
    "temperature": 0.7,           # sampling temperature for exploration
    "grad_clip": 1.0,
}
```

### 2.4 — GRPO Data

- Use the **same question set** as SFT (or a superset), but only the prompts (not the CoT traces)
- Ground-truth answers are used only by the reward function, never shown to the model
- The model must discover its own reasoning chains — this is the key benefit of RL over SFT

### 2.5 — GRPO Monitoring

- Track: mean reward, reward std, KL divergence, policy loss, advantage distribution
- Log sample completions every N steps to inspect reasoning quality
- Save checkpoints periodically → `checkpoints/grpo_reasoning_step_{N}.pth`

---

## Phase 3: Evaluation

### 3.1 — Benchmarks

Evaluate on held-out reasoning benchmarks:
- **LogiQA** (logical reasoning) — primary benchmark
- **ReClor** (reading comprehension requiring logical reasoning) — stretch goal
- Custom held-out set of logic puzzles from the training distribution

### 3.2 — Evaluation Script

Create `eval_reasoning.py`:
- Load model checkpoint
- For each problem: generate with `temperature=0`, extract `<answer>` content
- Compare to ground truth (exact match / numerical equivalence)
- Report accuracy, format compliance rate, average reasoning length
