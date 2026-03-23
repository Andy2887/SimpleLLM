from utils import *
from gpt_download import *
import argparse
import os

MODEL_CONFIGS = {
    "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "774M": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    for _ in range(max_new_tokens):

        # If the number of input tokens is greater than context length, we cut the input token
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        # We only look at the last token (the one we want)
        logits = logits[:, -1, :]

        # Filter logits with top_k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            logits = logits - logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Stop if EOS token is generated
        if eos_id is not None and idx_next.item() == eos_id:
            break

        # Append token to the sequence as the input of next round
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

def format_instruction_prompt(prompt):
    """Wrap user prompt in Alpaca-style instruction template for finetuned model."""
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt}\n\n"
        f"### Response:\n"
    )


def main(gpt_config, input_prompt, model_size, device):

    gpt = GPTModel(gpt_config)

    finetuned_path = f"finetuned_weights/{model_size}_finetuned_weights.pth"

    # Check if there are finetuned weights in the repository.
    # If yes, we will use the finetuned parameters
    # Otherwise, we will use the GPT-2 pretrained weights from OpenAI
    if os.path.exists(finetuned_path):
        print(f"Loading finetuned weights from {finetuned_path}")
        gpt.load_state_dict(torch.load(finetuned_path, map_location=device, weights_only=True))
        input_prompt = format_instruction_prompt(input_prompt)
    else:
        print(f"No finetuned weights found for {model_size}, loading OpenAI GPT-2 weights")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        load_weights_into_gpt(gpt, params)
        del settings, params

    gpt.to(device)
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    # torch.manual_seed(123)

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer).to(device),
        max_new_tokens=100,
        context_size=gpt_config["context_length"],
        top_k=10,
        temperature=1.0,
        eos_id=50256
    )

    output_text = token_ids_to_text(token_ids, tokenizer)

    # Extract just the response when using finetuned model
    if "### Response:" in output_text:
        output_text = output_text.split("### Response:")[-1].strip()

    print("Output text:\n", output_text)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Generate text with a pretrained GPT-2 model.")
    parser.add_argument(
        "--prompt",
        default="Hello! I am your personal assistant.",
        help="Prompt for text generation."
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for running inference, e.g., cpu, cuda, mps, or auto."
    )
    parser.add_argument(
        "--parameter",
        default="774M",
        choices=["124M", "355M", "774M", "1558M"],
        help="Model size: 124M (small), 355M (medium), 774M (large), 1558M (xl)."
    )

    args = parser.parse_args()

    INPUT_PROMPT = args.prompt
    DEVICE = torch.device(args.device)
    MODEL_SIZE = args.parameter

    print("PyTorch:", torch.__version__)
    print("Device:", DEVICE)
    print("Model size:", MODEL_SIZE)

    CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    CONFIG.update(MODEL_CONFIGS[MODEL_SIZE])

    main(CONFIG, INPUT_PROMPT, MODEL_SIZE, DEVICE)