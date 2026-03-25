# Plan: Enable Reasoning for Llama 3.2 via SFT + GRPO

## Goal

Transform the Llama 3.2 1B model into a reasoning model by:
1. **SFT (Supervised Fine-Tuning)** on chain-of-thought data with `<think>` and `<answer>` structured output
2. **RL (GRPO — Group Relative Policy Optimization)** to further refine reasoning quality through reward-based optimization

---

## Phase 0: Data Preparation

### 0.1 — Chain-of-Thought Dataset

Create or curate a CoT dataset where each sample follows this format:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant that thinks step by step.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
<think>
{step-by-step reasoning}
</think>
<answer>
{final answer}
</answer><|eot_id|>
```

**Data source:**
- **OpenR1-Math-220k** — high-quality math reasoning traces

### 0.2 — Data Processing Pipeline

Create `data/prepare_cot_data.py`:
- Load raw dataset (JSON/JSONL/Hugging Face datasets)
- Filter for quality: discard samples where `<think>` or `<answer>` tags are missing/malformed
- Format into the Llama 3.2 chat template using the existing `Tokenizer` class
- Register `<think>`, `</think>`, `<answer>`, `</answer>` as special tokens in the tokenizer (extend vocab from 128,256 → 128,260 and resize the embedding/output layers)
- Split into train (90%) / val (10%)
- Save as tokenized `.pt` files for fast loading

### 0.3 — Dataset Class

Create `cot_dataset.py`:
- `CoTDataset(torch.utils.data.Dataset)` — returns `(input_ids, target_ids)` pairs
- Mask the prompt tokens in `target_ids` with `-100` (only compute loss on the assistant's response, i.e., the `<think>...</think><answer>...</answer>` portion)
- Dynamic padding collate function (adapt from existing `custom_collate_fn` in `gpt_instruction_finetuning.py`)
- Max sequence length: 2048 tokens (configurable; balance between context and memory)

---

## Phase 1: Supervised Fine-Tuning (SFT)

### 1.1 — SFT Training Script

Create `sft_reasoning.py`:

**Training Config:**
```python
sft_config = {
    "epochs": 3,
    "batch_size": 2,              # small for 1B on consumer GPU
    "gradient_accumulation_steps": 8,  # effective batch size = 16
    "learning_rate": 2e-5,
    "weight_decay": 0.1,
    "warmup_steps": 100,
    "max_seq_len": 2048,
    "grad_clip": 1.0,
}
```

**Training Loop:**
- Optimizer: AdamW (reuse pattern from `gpt_instruction_finetuning.py`)
- LR schedule: linear warmup → cosine decay
- Loss: cross-entropy with `ignore_index=-100` (masked prompt tokens)
- Gradient clipping: max norm 1.0
- Gradient accumulation to simulate larger batch sizes
- Validation loss every N steps
- Save checkpoint at end of each epoch

**Memory Optimization (for consumer GPUs):**
- bfloat16 mixed precision via `torch.autocast`
- Gradient checkpointing (`torch.utils.checkpoint`) on transformer blocks to trade compute for memory
- Gradient accumulation (effective batch size 16, actual batch size 2)

### 1.2 — SFT Validation

- Track val loss per epoch
- Generate sample outputs on a held-out set of 20 questions after each epoch
- Verify model produces well-formed `<think>...</think><answer>...</answer>` structure
- Save the best checkpoint by val loss → `checkpoints/sft_reasoning.pth`

---

## Phase 2: Reinforcement Learning (GRPO)

### 2.1 — Reward Functions

Create `rewards.py` with composable reward signals:

| Reward Component    | Weight | Description |
|---------------------|--------|-------------|
| **Format reward**   | 0.1    | +1 if output contains valid `<think>...</think><answer>...</answer>` structure, 0 otherwise |
| **Correctness reward** | 0.7 | +1 if extracted answer matches ground truth (exact match or numerical equivalence), 0 otherwise |
| **Reasoning length reward** | 0.1 | Small bonus for non-trivial reasoning (penalize empty `<think>` blocks), capped to avoid reward hacking |
| **Cosine similarity reward** | 0.1 | Soft partial-credit via embedding similarity between predicted and ground-truth answer (optional, for non-exact-match tasks) |

Primary signal is **correctness** — this is what drives the model to actually reason better.

### 2.2 — GRPO Training Script

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

### 2.3 — GRPO Data

- Use the **same question set** as SFT (or a superset), but only the prompts (not the CoT traces)
- Ground-truth answers are used only by the reward function, never shown to the model
- The model must discover its own reasoning chains — this is the key benefit of RL over SFT

### 2.4 — GRPO Monitoring

- Track: mean reward, reward std, KL divergence, policy loss, advantage distribution
- Log sample completions every N steps to inspect reasoning quality
- Save checkpoints periodically → `checkpoints/grpo_reasoning_step_{N}.pth`

---

## Phase 3: Evaluation

### 3.1 — Benchmarks

Evaluate on held-out reasoning benchmarks:
- **GSM8K** (grade-school math) — primary benchmark
- **MATH** (competition math) — stretch goal
- Custom held-out set from the training distribution

### 3.2 — Evaluation Script

Create `eval_reasoning.py`:
- Load model checkpoint
- For each problem: generate with `temperature=0`, extract `<answer>` content
- Compare to ground truth (exact match / numerical equivalence)
- Report accuracy, format compliance rate, average reasoning length
