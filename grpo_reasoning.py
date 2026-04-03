import os
import csv
import re
import argparse
import time
import torch

from llama import Llama3Model, Tokenizer, LLAMA32_CONFIG_1B
from data_prep.prepare_cot_data import load_and_process_rl_data
from utils import generate


def correctness_reward(response_text, ground_truth):
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

GRPO_CONFIG = {
    "num_rollouts": 8,
    "learning_rate": 1e-5,
    "max_gen_len": 1024,
    "temperature": 1.0,
    "grad_clip": 1.0,
    "weight_decay": 0.1,
}

def sequence_logprob_and_entropy(model, token_ids, prompt_len):
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)

    targets = token_ids[1:]
    selected = logprobs[:-1].gather(1, targets.unsqueeze(-1)).squeeze(-1)

    # Log-prob of the generated answer tokens (sum over answer steps)
    selected_answer_logprobs = selected[prompt_len - 1:]
    logp_all_steps = torch.sum(selected_answer_logprobs)

    # Calculate entropy
    # As a rule of thumb:
    # Very low entropy (≈ 0–0.5) means one token dominates the distribution, where the model is highly confident and behaves almost deterministically
    # Moderate entropy (≈ 1–2) means that the probability mass is shared among a few more tokens, which is typical during stable training
    # High entropy (≫ 2, approaching log(vocabulary_size); here: log(128256) = 5.1), the probability mass is spread across many tokens, and the model is highly uncertain and behaves close to random
    all_answer_logprobs = logprobs[:-1][prompt_len - 1:]
    if all_answer_logprobs.numel() == 0:  # Safeguard if the model immediately returns EOS token
        entropy_all_steps = logp_all_steps.new_tensor(0.0)
    else:
        all_answer_probs = torch.exp(all_answer_logprobs)  # convert logprob to prob
        plogp = all_answer_probs * all_answer_logprobs     # elementwise p * log p
        step_entropy = -torch.sum(plogp, dim=-1)           # sum over vocab -> entropy per step
        entropy_all_steps = torch.mean(step_entropy)       # average over answer steps

    return logp_all_steps, entropy_all_steps

def compute_grpo_loss(
    model,
    tokenizer,
    prompt_tokens,
    ground_truth,
    device,
    config,
    num_rollouts=4,
    max_gen_len=1024,
    temperature=0.7,
):
    eot_id = tokenizer.special["<|eot_id|>"]
    prompt_tensor = torch.tensor([prompt_tokens], device=device)
    prompt_len = len(prompt_tokens)

    roll_logps, roll_rewards, roll_entropies, samples = [], [], [], []

    was_training = model.training
    model.eval()

    # generate rollouts
    for _ in range(num_rollouts):
        # Stage 1: generate rollout
        with torch.no_grad():
            full_seq = generate(
                model=model,
                idx=prompt_tensor.clone(),
                max_new_tokens=max_gen_len,
                context_size=config["context_length"],
                temperature=temperature,
                eos_id=eot_id,
                use_cache=True,
            )
        token_ids = full_seq[0]
        response_ids = token_ids[prompt_len:]
        text = tokenizer.decode(response_ids.tolist())

        # Stage 2: compute reward
        reward = correctness_reward(text, ground_truth)
        roll_rewards.append(reward)

        # Stage 3: compute logprob and entropy
        logp, entropy = sequence_logprob_and_entropy(model, token_ids, prompt_len)
        roll_logps.append(logp)
        roll_entropies.append(entropy.item())

        samples.append({
            "text": text,
            "reward": reward,
            "gen_len": len(response_ids),
        })
    
    if was_training:
        model.train()

    # collect all rewards & compute advantages
    rewards = torch.tensor(roll_rewards, device=device)
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

    # collect all logprobs & compute policy gradient loss
    logps = torch.stack(roll_logps)
    pg_loss = -(advantages.detach() * logps).mean()

    return {
        "pg_loss": pg_loss.item(),
        "rewards": roll_rewards,
        "advantages": advantages.detach().cpu().tolist(),
        "entropies": roll_entropies,
        "samples": samples,
        "loss_tensor": pg_loss,
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
    parser.add_argument("--mid_epoch_checkpoints", action="store_true",
                        help="Save checkpoints every 100 steps")
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Number of steps (samples) to run (e.g. 100 means indices 0..99)")
    parser.add_argument("--start_from", type=int, default=0,
                        help="Sample index to start from (e.g. 20 means process indices 20..N-1)")
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
    if len(rl_data) > 2000:
        rl_data = rl_data[:2000]
    if args.start_from > 0:
        rl_data = rl_data[args.start_from:]
        print(f"Starting from sample index {args.start_from}")
    if args.num_steps is not None:
        rl_data = rl_data[:args.num_steps]
        print(f"Limiting to {args.num_steps} steps")
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
    metrics_writer.writerow(["Step", "Loss", "Reward_Avg", "Avg_Response_Len", "Adv_Avg", "Adv_Std", "Entropy"])

    total_steps = len(rl_data)
    checkpoint_interval = 100

    print(f"Total steps (questions): {total_steps}")
    print(f"Num rollouts per question: {args.num_rollouts}")
    print("Starting GRPO training...\n")

    timer_start = time.time()
    prev_ckpt_path = None

    for step, sample in enumerate(rl_data):
        prompt_tokens = sample["prompt_tokens"]
        ground_truth = sample["answer"]

        # Compute GRPO loss
        optimizer.zero_grad()
        stats = compute_grpo_loss(
            model=model,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            ground_truth=ground_truth,
            device=device,
            config=config,
            num_rollouts=args.num_rollouts,
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
        )

        reward_avg = sum(stats["rewards"]) / len(stats["rewards"])
        avg_response_len = sum(
            s["gen_len"] for s in stats["samples"]
        ) / len(stats["samples"])
        adv = stats["advantages"]
        adv_avg = sum(adv) / len(adv)
        adv_std = (sum((a - adv_avg) ** 2 for a in adv) / len(adv)) ** 0.5
        entropy_avg = sum(stats["entropies"]) / len(stats["entropies"])

        stats["loss_tensor"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        print(
            f"Step {step+1}/{total_steps} | "
            f"Loss: {stats['pg_loss']:.6f} | "
            f"Reward: {reward_avg:.4f} | "
            f"Avg len: {avg_response_len:.1f} | "
            f"Adv avg: {adv_avg:.4f} | "
            f"Adv std: {adv_std:.4f} | "
            f"Entropy: {entropy_avg:.4f}"
        )
        metrics_writer.writerow([step + 1, f"{stats['pg_loss']:.6f}", f"{reward_avg:.4f}", f"{avg_response_len:.1f}", f"{adv_avg:.4f}", f"{adv_std:.4f}", f"{entropy_avg:.4f}"])
        metrics_file.flush()

        # Sample output + timer (every 10 steps)
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
            print(f"  [Sample] A: {stats['samples'][0]['text']}")
            print()
            timer_start = time.time()

        # Checkpoint (every 1/10 of total steps) if enabled
        if args.mid_epoch_checkpoints and (step + 1) % checkpoint_interval == 0:
            ckpt_path = f"checkpoints/rl_reasoning_step{step+1}.pth"
            if prev_ckpt_path and os.path.exists(prev_ckpt_path):
                os.remove(prev_ckpt_path)
                print(f"  Deleted previous checkpoint: {prev_ckpt_path}")
            torch.save(model.state_dict(), ckpt_path)
            prev_ckpt_path = ckpt_path
            print(f"  Saved checkpoint: {ckpt_path}")

    # Delete last mid-epoch checkpoint before saving final
    if prev_ckpt_path and os.path.exists(prev_ckpt_path):
        os.remove(prev_ckpt_path)
        print(f"Deleted previous checkpoint: {prev_ckpt_path}")
    ckpt_path = "checkpoints/rl_reasoning_final.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved final checkpoint: {ckpt_path}")
    metrics_file.close()
    print("Saved metrics: metrics/rl_metrics.csv")
    print("GRPO training complete!")


if __name__ == "__main__":
    main()
