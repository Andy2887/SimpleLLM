import json
import argparse
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from utils import GPTModel
from gpt_download import download_and_load_gpt2, load_weights_into_gpt


def format_instruction(entry):
    instruction = entry["instruction"]
    input_text = entry.get("input", "")
    output_text = entry["output"]

    if input_text:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output_text}<|endoftext|>"
        )
    else:
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output_text}<|endoftext|>"
        )
    return prompt


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for entry in data:
            formatted = format_instruction(entry)
            token_ids = tokenizer.encode(formatted, allowed_special={"<|endoftext|>"})
            # Truncate if too long
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            self.data.append(token_ids)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        token_ids = self.data[idx]
        input_ids = torch.tensor(token_ids[:-1])
        target_ids = torch.tensor(token_ids[1:])
        return input_ids, target_ids


def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100):
    input_ids_list, target_ids_list = zip(*batch)

    max_len = max(len(ids) for ids in input_ids_list)

    inputs_padded = []
    targets_padded = []

    for input_ids, target_ids in zip(input_ids_list, target_ids_list):
        pad_len = max_len - len(input_ids)
        # Pad input with pad_token_id
        padded_input = torch.cat([
            input_ids,
            torch.full((pad_len,), pad_token_id, dtype=torch.long)
        ])
        # Pad target with ignore_index so padding doesn't contribute to loss
        padded_target = torch.cat([
            target_ids,
            torch.full((pad_len,), ignore_index, dtype=torch.long)
        ])
        inputs_padded.append(padded_input)
        targets_padded.append(padded_target)

    return torch.stack(inputs_padded), torch.stack(targets_padded)


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten(), ignore_index=-100
    )
    return loss


def evaluate_loss(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
            num_batches += 1
    model.train()
    return total_loss / num_batches if num_batches > 0 else 0.0


def train(model, train_loader, val_loader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

            print(f"--- Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f} ---")

        avg_train_loss = total_loss / num_batches
        avg_val_loss = evaluate_loss(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train loss: {avg_train_loss:.4f} | "
              f"Val loss: {avg_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on instruction data.")
    parser.add_argument("--device", default="cpu", help="Device: cpu, cuda, or mps.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    with open("instruction-data.json", "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} instruction examples")

    # Split into train/val (90/10)
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = InstructionDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = InstructionDataset(val_data, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=custom_collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=custom_collate_fn, drop_last=False
    )

    gpt_config = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
    }

    _, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    model = GPTModel(gpt_config)
    load_weights_into_gpt(model, params)
    model.to(device)
    print("Loaded pretrained GPT-2 weights")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    print(f"\nTraining for {args.epochs} epochs...")
    train(model, train_loader, val_loader, optimizer, device, args.epochs)

    # Save finetuned weights
    torch.save(model.state_dict(), "parameters_after_finetuning.pth")
    print("\nSaved finetuned weights to parameters_after_finetuning.pth")


if __name__ == "__main__":
    main()
