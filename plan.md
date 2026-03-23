# Fine-Tuning 1.5B Parameter Model on vast.ai

## Memory Requirements

For the 1558M (GPT-2 XL) model in fp32:

| Component                          | VRAM        |
|------------------------------------|-------------|
| Model parameters                   | ~6.2 GB     |
| Gradients                          | ~6.2 GB     |
| AdamW optimizer states             | ~12.4 GB    |
| Activations (batch=8, seq=512)     | ~8-12 GB    |
| **Total**                          | **~33-37 GB** |

**Recommended GPU:** A100 40GB or A6000 48GB. A 24GB GPU (RTX 3090/4090) can work if you add mixed precision training and reduce batch size.

---

## Step-by-Step vast.ai Guide

### 1. Create an Account & Add Credits
- Sign up at [vast.ai](https://vast.ai)
- Add credits via the billing page (start with ~$5-10, training will cost $1-3/hr depending on GPU)

### 2. Rent a GPU Instance
- Go to the **Search** page
- Filter for:
  - **GPU RAM:** >=40 GB (for safe fp32), or >=24 GB (if you add mixed precision)
  - **GPU type:** A100, A6000, or RTX 4090
  - **Disk space:** >=50 GB (model weights + data)
  - **Docker image:** `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel` (or similar recent PyTorch image)
- Sort by **price** and pick a cheap instance
- Click **Rent** and wait for it to start

### 3. Connect to the Instance
```bash
# Install vast.ai CLI
pip install vastai

# Set your API key (found in Account settings)
vastai set api-key YOUR_API_KEY

# SSH into your instance (or use the web terminal)
vastai ssh-url INSTANCE_ID
```

### 4. Upload Your Project
```bash
# Option A: From your local machine, use scp
scp -r /Users/yuanliheng/Desktop/Tech-Projects/projects/ai/build_your_llm_from_scratch user@INSTANCE_IP:/workspace/

# Option B: Push to GitHub, then clone on the instance
git clone https://github.com/YOUR_REPO.git /workspace/project
```

### 5. Install Dependencies on the Instance
```bash
cd /workspace/build_your_llm_from_scratch
pip install tiktoken numpy tqdm
# torch is already installed in the PyTorch Docker image
# tensorflow is only needed for weight download - install it too
pip install tensorflow
```

### 6. Download the 1558M Pretrained Weights
```bash
# This will download ~6GB of GPT-2 XL weights from OpenAI
python gpt_download.py 1558M
```

### 7. Run Fine-Tuning
```bash
python gpt_instruction_finetuning.py \
  --device cuda \
  --parameter 1558M \
  --epochs 2 \
  --batch_size 4 \
  --lr 5e-5 \
  --max_length 512
```
Use batch_size 4 (instead of 8) to be safe on memory. Increase if VRAM allows.

### 8. Download the Fine-Tuned Weights
```bash
# From your local machine
scp user@INSTANCE_IP:/workspace/build_your_llm_from_scratch/finetuned_weights/1558M_finetuned_weights.pth ./finetuned_weights/
```

### 9. Destroy the Instance
Don't forget -- you're billed by the hour:
```bash
vastai destroy instance INSTANCE_ID
```

---

## Code Modifications Needed

### 1. Mixed Precision Training (strongly recommended, saves ~40% VRAM)

In `gpt_instruction_finetuning.py`, in the `train_model_simple` function, wrap the forward pass and loss computation with `torch.cuda.amp`:

```python
# Add at the top of the file
from torch.cuda.amp import autocast, GradScaler

# Before the training loop, create a scaler
scaler = GradScaler()

# Inside the training loop, replace:
#   logits = model(input_batch)
#   loss = ...
#   loss.backward()
#   optimizer.step()
# With:
#   with autocast():
#       logits = model(input_batch)
#       loss = ...
#   scaler.scale(loss).backward()
#   scaler.step(optimizer)
#   scaler.update()
```

Do the same for `calc_loss_batch` when called during validation (wrap in `autocast` + `torch.no_grad`).

### 2. Reduce Batch Size / Add Gradient Accumulation (if OOM)

If you still get out-of-memory errors, reduce `--batch_size` to 2 or 1, and add gradient accumulation to maintain effective batch size:

```python
# Accumulate gradients over N mini-batches before stepping
accumulation_steps = 4  # effective batch = batch_size * accumulation_steps
optimizer.zero_grad()
for i, (input_batch, target_batch) in enumerate(train_loader):
    loss = ...
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. Add Checkpointing (recommended for long runs)

Save a checkpoint every epoch so you don't lose progress if the instance dies:

```python
# At the end of each epoch in train_model_simple:
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
}, f"checkpoint_epoch_{epoch}.pth")
```

### 4. No Other Changes Needed

Your code already supports `--device cuda` and `--parameter 1558M`, so the core logic works as-is. The tokenizer (`tiktoken`) works the same on Linux. The weight download script works on any platform.

---

## Cost Estimate

| GPU           | $/hr (approx) | Time for 2 epochs | Total      |
|---------------|----------------|--------------------|------------|
| A100 40GB     | $0.80-1.50     | ~1-2 hours         | $1-3       |
| A6000 48GB    | $0.40-0.80     | ~2-3 hours         | $1-2.50    |
| RTX 4090 24GB | $0.30-0.50     | ~2-4 hours (needs mixed precision) | $0.60-2 |

Prices vary by availability. The 1,100-example dataset is small, so training should be fast.

---

## Priority Summary

1. **Must do:** Use `--device cuda` and `--parameter 1558M` (already supported)
2. **Strongly recommended:** Add mixed precision training (biggest impact on fitting in VRAM)
3. **Recommended:** Add checkpointing (protects against instance interruption)
4. **If needed:** Gradient accumulation (only if you still hit OOM after mixed precision)
