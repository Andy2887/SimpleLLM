"""
Phase 1.1: Supervised Fine-Tuning (SFT) — Full Fine-Tuning

Trains all parameters of Llama 3.2 1B on chain-of-thought data to teach
the model to produce <think>...</think><answer>...</answer> structured output.

Usage:
    python sft_reasoning.py --device auto
    python sft_reasoning.py --device cuda --epochs 3 --batch_size 4
"""

import os
import math
import argparse
import torch
from torch.utils.data import DataLoader

from llama import Llama3Model, Tokenizer, LLAMA32_CONFIG_1B, load_weights_into_llama
from cot_dataset import CoTDataset, cot_collate_fn
from data_prep.prepare_cot_data import load_and_process_sft_data

SFT_CONFIG = {
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.1,
    "warmup_steps": 100,
    "max_seq_len": 2048,
    "grad_clip": 1.0,
}


def get_lr(step, warmup_steps, total_steps, max_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description="SFT for reasoning (full fine-tuning)")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--epochs", type=int, default=SFT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=SFT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=SFT_CONFIG["learning_rate"])
    parser.add_argument("--max_seq_len", type=int, default=SFT_CONFIG["max_seq_len"])
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N batches")
    parser.add_argument(
        "--weights",
        default=os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/consolidated.00.pth"),
    )
    parser.add_argument(
        "--tokenizer",
        default=os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/tokenizer.model"),
    )
    args = parser.parse_args()

    # ---- Device ----
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- Data ----
    print("Preparing data...")
    tokenizer = Tokenizer(args.tokenizer)
    data = load_and_process_sft_data(tokenizer, max_seq_len=args.max_seq_len)
    dataset = CoTDataset(data, max_seq_len=args.max_seq_len)
    print(f"Dataset size: {len(dataset)} samples")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=cot_collate_fn,
        drop_last=True,
    )

    # ---- Model ----
    print("Loading model...")
    config = LLAMA32_CONFIG_1B
    model = Llama3Model(config)

    print("Loading pretrained weights...")
    params = torch.load(args.weights, map_location="cpu", weights_only=True)
    load_weights_into_llama(model, config, params)
    del params

    model.to(device)
    model.train()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=SFT_CONFIG["weight_decay"],
        betas=(0.9, 0.999),
    )

    # ---- Training ----
    total_steps = len(train_loader) * args.epochs
    global_step = 0

    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total optimizer steps: {total_steps}")
    print("Starting training...\n")

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()

            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1),
                target_batch.flatten(),
                ignore_index=-100,
            )
            loss.backward()

            # Update learning rate
            lr = get_lr(global_step, SFT_CONFIG["warmup_steps"], total_steps, args.lr)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), SFT_CONFIG["grad_clip"])

            optimizer.step()
            global_step += 1

            batch_loss = loss.item()
            epoch_loss += batch_loss
            num_batches += 1

            if (batch_idx + 1) % args.log_interval == 0:
                print(
                    f"  Epoch {epoch+1}/{args.epochs} | "
                    f"Batch {batch_idx+1}/{len(train_loader)} | "
                    f"Loss: {batch_loss:.4f} | "
                    f"LR: {lr:.2e}"
                )

        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}\n")

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/sft_reasoning.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")
    print("Training complete!")


if __name__ == "__main__":
    main()
