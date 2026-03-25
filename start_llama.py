from llama import *
from utils import *
import argparse

DEFAULT_WEIGHTS_PATH = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/consolidated.00.pth")
DEFAULT_TOKENIZER_PATH = os.path.expanduser("~/.llama/checkpoints/Llama3.2-1B/tokenizer.model")


def main(config, input_prompt, device, weights_path, tokenizer_path):
    print("Loading model...")
    model = Llama3Model(config)

    print("Loading pretrained weights...")
    params = torch.load(weights_path, map_location="cpu", weights_only=True)
    load_weights_into_llama(model, config, params)
    del params

    print("Moving model to corresponding device...")
    model.to(device)
    model.eval()

    tokenizer = Tokenizer(tokenizer_path)

    print("Start generating output...")

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_prompt, tokenizer).to(device),
        max_new_tokens=50,
        context_size=config["context_length"],
        top_k=10,
        temperature=0.3,
        eos_id=tokenizer.special["<|end_of_text|>"]
    )

    output_text = token_ids_to_text(token_ids, tokenizer)

    # Strip the BOS token from output if present
    if output_text.startswith("<|begin_of_text|>"):
        output_text = output_text[len("<|begin_of_text|>"):]

    print("Output text:\n", output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate text with a pretrained Llama 3.2 model."
    )
    parser.add_argument(
        "--prompt",
        default="Every effort",
        help="Prompt for text generation."
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for running inference, e.g., cpu, cuda, mps, or auto."
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS_PATH,
        help="Path to the pretrained weights file."
    )
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER_PATH,
        help="Path to the tokenizer model file."
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

    print("PyTorch:", torch.__version__)
    print("Device:", device)

    main(LLAMA32_CONFIG_1B, args.prompt, device, args.weights, args.tokenizer)
