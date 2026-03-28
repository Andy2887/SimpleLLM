import os
import csv
import argparse
import time
import torch
from torch.utils.data import DataLoader

from llama import Llama3Model, Tokenizer, LLAMA32_CONFIG_1B, load_weights_into_llama
from data_prep.cot_dataset import CoTDataset, cot_collate_fn
from data_prep.prepare_cot_data import load_and_process_sft_data

SFT_CONFIG = {
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "max_seq_len": 1024,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
}


def main():
    parser = argparse.ArgumentParser(description="SFT for reasoning (full fine-tuning)")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--epochs", type=int, default=SFT_CONFIG["epochs"])
    parser.add_argument("--batch_size", type=int, default=SFT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=SFT_CONFIG["learning_rate"])
    parser.add_argument("--max_seq_len", type=int, default=SFT_CONFIG["max_seq_len"])
    parser.add_argument("--weight_decay", type=float, default=SFT_CONFIG["weight_decay"])
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N batches")
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing (enabled by default)")
    parser.add_argument("--mid_epoch_checkpoints", action="store_true",
                        help="Save checkpoints every 1/10 epoch")
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
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing = True
        print("Gradient checkpointing enabled")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- Scheduler ----
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ---- Training ----
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    metrics_file = open("metrics/sft_metrics.csv", "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["Epoch", "Batch", "Loss", "LR"])

    steps_per_epoch = len(train_loader)
    checkpoint_interval = max(1, steps_per_epoch // 10)
    print(f"Batches per epoch: {steps_per_epoch}")
    print(f"Batch size: {args.batch_size}")
    print(f"Total optimizer steps: {total_steps}")
    print("Starting training...\n")

    global_step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            batch_start = time.time()

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

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), SFT_CONFIG["grad_clip"])

            optimizer.step()
            scheduler.step()

            batch_time = time.time() - batch_start
            batch_loss = loss.item()
            epoch_loss += batch_loss
            num_batches += 1
            global_step += 1

            remaining_steps = total_steps - global_step
            eta_seconds = remaining_steps * batch_time
            eta_min, eta_sec = divmod(int(eta_seconds), 60)
            eta_hr, eta_min = divmod(eta_min, 60)

            if (batch_idx + 1) % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  Epoch {epoch+1}/{args.epochs} | "
                    f"Batch {batch_idx+1}/{steps_per_epoch} | "
                    f"Loss: {batch_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Batch time: {batch_time:.2f}s | "
                    f"ETA: {eta_hr:02d}:{eta_min:02d}:{eta_sec:02d}"
                )
                metrics_writer.writerow([epoch + 1, batch_idx + 1, f"{batch_loss:.4f}", f"{lr:.2e}"])
                metrics_file.flush()

            # Save checkpoint every 1/10 of an epoch
            if args.mid_epoch_checkpoints and (batch_idx + 1) % checkpoint_interval == 0:
                ckpt_path = f"checkpoints/sft_reasoning_ep{epoch+1}_step{batch_idx+1}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"  Saved checkpoint: {ckpt_path}")

        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}\n")

    # Save final checkpoint
    ckpt_path = "checkpoints/sft_reasoning_final.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved final checkpoint: {ckpt_path}")
    metrics_file.close()
    print(f"Saved metrics: metrics/sft_metrics.csv")
    print("Training complete!")


if __name__ == "__main__":
    main()
