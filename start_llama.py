from llama import *
from utils import *
import argparse

MODEL_CONFIGS = {
    "1B": {
        "config": LLAMA32_CONFIG_1B,
        "weights": os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/consolidated.00.pth"),
        "tokenizer": os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/tokenizer.model"),
    },
    "3B": {
        "config": LLAMA32_CONFIG_3B,
        "weights": os.path.expanduser("~/.llama/checkpoints/Llama3.2-3B/consolidated.00.pth"),
        "tokenizer": os.path.expanduser("~/.llama/checkpoints/Llama3.2-3B/tokenizer.model"),
    },
    "8B": {
        "config": LLAMA31_CONFIG_8B,
        "weights": os.path.expanduser("~/.llama/checkpoints/Llama3.1-8B/consolidated.00.pth"),
        "tokenizer": os.path.expanduser("~/.llama/checkpoints/Llama3.1-8B/tokenizer.model"),
    },
}


def format_prompt(input_text):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant that thinks step by step.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{input_text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def main(config, input_prompt, device, weights_path, tokenizer_path):
    print("Loading model...")
    model = Llama3Model(config)

    print("Loading pretrained weights...")
    params = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "tok_embeddings.weight" in params:
        load_weights_into_llama(model, config, params)
    else:
        model.load_state_dict(params)
    del params

    print("Moving model to corresponding device...")
    model.to(device)
    model.eval()

    tokenizer = Tokenizer(tokenizer_path)

    if input_prompt:
        prompts = [input_prompt]
    else:
        prompts = None

    while True:
        if prompts:
            user_input = prompts.pop(0)
        else:
            try:
                user_input = input("\nEnter prompt (or 'exit'/'quit' to stop): ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            if user_input.strip().lower() in ("exit", "quit"):
                print("Exiting.")
                break

        prompt = format_prompt(user_input)

        print("Start generating output...")

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(prompt, tokenizer).to(device),
            max_new_tokens=1024,
            context_size=config["context_length"],
            top_k=10,
            temperature=0.3,
            eos_id=tokenizer.special["<|eot_id|>"]
        )

        prompt_len = text_to_token_ids(prompt, tokenizer).shape[1]
        output_text = token_ids_to_text(token_ids[:, prompt_len:], tokenizer)

        print(f"Output text:\n{output_text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate text with a pretrained Llama model."
    )
    parser.add_argument(
        "--model",
        default="1B",
        choices=["1B", "3B", "8B"],
        help="Model size: 1B (Llama 3.2), 3B (Llama 3.2), or 8B (Llama 3.1)."
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Initial prompt for text generation. If omitted, starts in interactive mode."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for running inference, e.g., cpu, cuda, mps, or auto."
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to the pretrained weights file. Defaults to the standard path for the chosen model."
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to the tokenizer model file. Defaults to the standard path for the chosen model."
    )

    args = parser.parse_args()

    model_info = MODEL_CONFIGS[args.model]
    config = model_info["config"]
    weights_path = args.weights or model_info["weights"]
    tokenizer_path = args.tokenizer or model_info["tokenizer"]

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("PyTorch:", torch.__version__)
    print("Device:", device)
    print("Model:", args.model)

    main(config, args.prompt, device, weights_path, tokenizer_path)
