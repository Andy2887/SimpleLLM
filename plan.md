# Plan: Enable Reasoning for Llama 3.1 8B via LoRA SFT + GRPO

## Goal

Transform the Llama 3.1 8B model into a reasoning model by:
1. **SFT (Supervised Fine-Tuning)** with **LoRA (Low-Rank Adaptation)** on chain-of-thought data with `<think>` and `<answer>` structured output
2. **RL (GRPO — Group Relative Policy Optimization)** to further refine reasoning quality through reward-based optimization

## Fine-Tuning Method: LoRA

We use **LoRA** instead of full fine-tuning for parameter-efficient training.

**Why LoRA:**
- Gets 90-95% of full fine-tuning performance at a fraction of the cost
- Works well for domain adaptation and instruction tuning
- Regularizes better with limited data, reducing overfitting
- Enables fine-tuning on consumer GPUs

**LoRA Config:**
- Rank: 64, Alpha: 128
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Trainable params: ~1-2% of total model parameters

**GPU Memory Estimate (8B model with LoRA, BF16 base):**

| Component | Calculation | Memory |
|---|---|---|
| Base model weights (BF16) | 8B × 2 bytes | ~16 GB |
| LoRA adapter weights | ~160M × 2 bytes | ~0.3 GB |
| Gradients (LoRA params only) | ~160M × 2 bytes | ~0.3 GB |
| Optimizer states (LoRA params, AdamW FP32) | ~160M × 12 bytes | ~1.9 GB |
| Activations (batch=2, seq=2048) | varies | ~4-8 GB |
| **Total** | | **~20-24 GB** |

A single GPU with 24 GB VRAM (e.g., RTX 4090 or A5000) is sufficient.

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
- Format into the Llama 3.1 chat template using the existing `Tokenizer` class
- Map `<think>`, `</think>`, `<answer>`, `</answer>` to existing reserved token slots in the tokenizer (`<|reserved_0|>` → 128002, `<|reserved_1|>` → 128003, `<|reserved_2|>` → 128004, `<|reserved_3|>` → 128005). No vocab resize needed — these embedding rows already exist in the pretrained weights.
- Split into train (90%) / val (10%)

### 0.3 — Dataset Class

Create `cot_dataset.py`:
- `CoTDataset(torch.utils.data.Dataset)` — returns `(input_ids, target_ids)` pairs
- Mask the prompt tokens in `target_ids` with `-100` (only compute loss on the assistant's response, i.e., the `<think>...</think><answer>...</answer>` portion)
- Dynamic padding collate function (adapt from existing `custom_collate_fn` in `gpt_instruction_finetuning.py`)
- Max sequence length: 2048 tokens (configurable; balance between context and memory)

---

## Phase 1: Supervised Fine-Tuning (SFT) with LoRA

### 1.1 — SFT Training Script

Create `sft_reasoning.py`:

**Training Config:**
```python
sft_config = {
    "epochs": 3,
    "batch_size": 2,              # small for 8B on consumer GPU
    "gradient_accumulation_steps": 8,  # effective batch size = 16
    "learning_rate": 2e-5,
    "weight_decay": 0.1,
    "warmup_steps": 100,
    "max_seq_len": 2048,
    "grad_clip": 1.0,
    # LoRA config
    "lora_rank": 64,
    "lora_alpha": 128,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}
```

**Training Loop:**
- Freeze base model weights; only train LoRA adapter parameters
- Optimizer: AdamW (only over LoRA parameters)
- LR schedule: linear warmup → cosine decay
- Loss: cross-entropy with `ignore_index=-100` (masked prompt tokens)
- Gradient clipping: max norm 1.0
- Gradient accumulation to simulate larger batch sizes
- Validation loss every N steps
- Save LoRA adapter checkpoint at end of each epoch

**Memory Optimization (for consumer GPUs):**
- LoRA reduces trainable parameters to ~1-2% of total — estimated **~20-24 GB VRAM** for an 8B model
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

### 2.2 — GPU Memory Estimate (GRPO with LoRA)

GRPO is more memory-intensive than SFT due to multi-completion generation and reference log-prob computation. With LoRA, the reference model is free — it shares the base weights (just disables LoRA adapters).

**Generation phase** (forward only, batch=2 × group_size=4 = 8 sequences):

| Component | Memory |
|---|---|
| Base model + LoRA (BF16) | ~16.3 GB |
| KV cache (8 seqs × ~1536 tokens) | ~1.5 GB |
| **Peak** | **~18 GB** |

**Training phase** (forward + backward, the bottleneck):

| Component | Calculation | Memory |
|---|---|---|
| Base model weights (BF16) | 8B × 2 bytes | ~16 GB |
| LoRA adapters | ~160M × 2 bytes | ~0.3 GB |
| LoRA gradients | ~160M × 2 bytes | ~0.3 GB |
| Optimizer states (AdamW FP32) | ~160M × 12 bytes | ~1.9 GB |
| Activations (batch=2, seq~1536) | varies | ~6-10 GB |
| Stored log-probs (policy + ref) | negligible | ~0.01 GB |
| **Total** | | **~24-28 GB** |

A 24 GB GPU (RTX 4090) is tight but feasible with gradient checkpointing. A 48 GB GPU (A6000/A40) gives comfortable headroom.

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
- **GSM8K** (grade-school math) — primary benchmark
- **MATH** (competition math) — stretch goal
- Custom held-out set from the training distribution

### 3.2 — Evaluation Script

Create `eval_reasoning.py`:
- Load model checkpoint
- For each problem: generate with `temperature=0`, extract `<answer>` content
- Compare to ground truth (exact match / numerical equivalence)
- Report accuracy, format compliance rate, average reasoning length
