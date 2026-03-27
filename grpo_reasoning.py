import os
import csv
import argparse
import time
import torch

from llama import Llama3Model, Tokenizer, LLAMA32_CONFIG_1B
from data_prep.prepare_cot_data import load_and_process_rl_data
from utils import generate
from rewards import correctness_reward

GRPO_CONFIG = {
    "num_rollouts": 4,
    "learning_rate": 1e-6,
    "max_gen_len": 1024,
    "temperature": 0.7,
    "grad_clip": 1.0,
    "weight_decay": 0.1,
}


def main():
    parser = argparse.ArgumentParser(description="GRPO for reasoning (full fine-tuning)")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--num_rollouts", type=int, default=GRPO_CONFIG["num_rollouts"])
    parser.add_argument("--lr", type=float, default=GRPO_CONFIG["learning_rate"])
    parser.add_argument("--max_gen_len", type=int, default=GRPO_CONFIG["max_gen_len"])
    parser.add_argument("--temperature", type=float, default=GRPO_CONFIG["temperature"])
    parser.add_argument("--grad_clip", type=float, default=GRPO_CONFIG["grad_clip"])
    parser.add_argument("--weight_decay", type=float, default=GRPO_CONFIG["weight_decay"])
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                        help="Disable gradient checkpointing (enabled by default)")
    parser.add_argument(
        "--weights",
        default="checkpoints/sft_reasoning_final.pth",
        help="Path to SFT checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        default=os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/tokenizer.model"),
    )
    args = parser.parse_args()

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

    print("Preparing RL data...")
    tokenizer = Tokenizer(args.tokenizer)
    rl_data = load_and_process_rl_data(tokenizer, max_seq_len=args.max_gen_len)
    print(f"RL dataset size: {len(rl_data)} questions")

    print("Loading model...")
    config = LLAMA32_CONFIG_1B
    model = Llama3Model(config)

    print(f"Loading SFT weights from {args.weights}...")
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    del state_dict

    model.to(device)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing = True
        print("Gradient checkpointing enabled")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ---- Training ----
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    metrics_file = open("metrics/rl_metrics.csv", "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["Step", "Loss", "Reward_Avg", "Avg_Response_Len"])

    total_steps = len(rl_data)
    checkpoint_interval = max(1, total_steps // 10)
    eot_id = tokenizer.special["<|eot_id|>"]

    print(f"Total steps (questions): {total_steps}")
    print(f"Num rollouts per question: {args.num_rollouts}")
    print(f"Checkpoint every {checkpoint_interval} steps (1/10 of total)")
    print("Starting GRPO training...\n")

    timer_start = time.time()

    for step, sample in enumerate(rl_data):
        prompt_tokens = sample["prompt_tokens"]
        ground_truth = sample["answer"]
        prompt_tensor = torch.tensor([prompt_tokens], device=device)

        # ---- Generation phase (no grad) ----
        model.eval()
        completions = []
        response_texts = []
        for _ in range(args.num_rollouts):
            with torch.no_grad():
                full_seq = generate(
                    model=model,
                    idx=prompt_tensor.clone(),
                    max_new_tokens=args.max_gen_len,
                    context_size=config["context_length"],
                    temperature=args.temperature,
                    eos_id=eot_id,
                    use_cache=True,
                )
            response_ids = full_seq[0, len(prompt_tokens):]
            completions.append(response_ids)
            response_texts.append(tokenizer.decode(response_ids.tolist()))

        # ---- Reward phase ----
        rewards = torch.tensor(
            [correctness_reward(text, ground_truth) for text in response_texts],
            device=device,
            dtype=torch.float32,
        )
        reward_avg = rewards.mean().item()
        avg_response_len = sum(len(c) for c in completions) / len(completions)

        # ---- Advantage calculation ----
        if rewards.std() < 1e-8:
            # All rewards identical — skip this training step
            print(
                f"Step {step+1}/{total_steps} | "
                f"Loss: skipped (uniform rewards) | "
                f"Reward: {reward_avg:.4f} | "
                f"Avg len: {avg_response_len:.1f}"
            )
            metrics_writer.writerow([step + 1, "N/A", f"{reward_avg:.4f}", f"{avg_response_len:.1f}"])
            metrics_file.flush()

            if (step + 1) % 10 == 0:
                elapsed = time.time() - timer_start
                avg_step_time = elapsed / 10
                remaining = avg_step_time * (total_steps - step - 1)
                rem_min, rem_sec = divmod(int(remaining), 60)
                rem_hr, rem_min = divmod(rem_min, 60)
                print(
                    f"  [Timer] Last 10 steps: {elapsed:.1f}s | "
                    f"ETA: {rem_hr:02d}:{rem_min:02d}:{rem_sec:02d}"
                )
                prompt_text = tokenizer.decode(prompt_tokens)
                print(f"  [Sample] Q: ...{prompt_text[-150:]}")
                print(f"  [Sample] A: {response_texts[0][:300]}")
                print()
                timer_start = time.time()
            continue

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        # ---- Training phase ----
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        for i in range(args.num_rollouts):
            response_ids = completions[i]
            if len(response_ids) == 0:
                continue

            full_ids = torch.cat([prompt_tensor[0], response_ids]).unsqueeze(0)
            logits = model(full_ids)

            # Log probs for response tokens only
            # logits[t] predicts token[t+1], so logits[prompt_len-1:-1] predicts response tokens
            prompt_len = len(prompt_tokens)
            response_logits = logits[0, prompt_len - 1 : -1, :]
            log_probs = torch.log_softmax(response_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(1, response_ids.unsqueeze(1)).squeeze(1)

            # GRPO loss: -advantage * sequence log prob, averaged over rollouts
            # Using sum (not mean) so shorter correct answers get larger gradients
            rollout_loss = -advantages[i] * token_log_probs.sum()
            (rollout_loss / args.num_rollouts).backward()
            total_loss += rollout_loss.item() / args.num_rollouts

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        # ---- Logging (every step) ----
        print(
            f"Step {step+1}/{total_steps} | "
            f"Loss: {total_loss:.6f} | "
            f"Reward: {reward_avg:.4f} | "
            f"Avg len: {avg_response_len:.1f}"
        )
        metrics_writer.writerow([step + 1, f"{total_loss:.6f}", f"{reward_avg:.4f}", f"{avg_response_len:.1f}"])
        metrics_file.flush()

        # ---- Sample output + timer (every 10 steps) ----
        if (step + 1) % 10 == 0:
            elapsed = time.time() - timer_start
            avg_step_time = elapsed / 10
            remaining = avg_step_time * (total_steps - step - 1)
            rem_min, rem_sec = divmod(int(remaining), 60)
            rem_hr, rem_min = divmod(rem_min, 60)
            print(
                f"  [Timer] Last 10 steps: {elapsed:.1f}s | "
                f"ETA: {rem_hr:02d}:{rem_min:02d}:{rem_sec:02d}"
            )
            prompt_text = tokenizer.decode(prompt_tokens)
            print(f"  [Sample] Q: ...{prompt_text[-150:]}")
            print(f"  [Sample] A: {response_texts[0][:300]}")
            print()
            timer_start = time.time()

        # ---- Checkpoint (every 1/10 of total steps) ----
        if (step + 1) % checkpoint_interval == 0:
            ckpt_path = f"checkpoints/fl_reasoning_step{step+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    ckpt_path = "checkpoints/rl_reasoning_final.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved final checkpoint: {ckpt_path}")
    metrics_file.close()
    print("Saved metrics: metrics/rl_metrics.csv")
    print("GRPO training complete!")


if __name__ == "__main__":
    main()
