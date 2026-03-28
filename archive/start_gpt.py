# from gpt import *
# from utils import *
# from gpt_download import *
# import argparse
# import os

# FINETUNED_WEIGHTS_PATH = "parameters_after_finetuning.pth"

# def main(gpt_config, input_prompt, model_size, device):

#     gpt = GPTModel(gpt_config)

#     # Check if there are finetuned weights in the repository.
#     # If yes, we will use the finetuned parameters
#     # Otherwise, we will use the GPT-2 pretrained weights from OpenAI
#     if os.path.exists(FINETUNED_WEIGHTS_PATH):
#         print(f"Loading finetuned weights from {FINETUNED_WEIGHTS_PATH}")
#         gpt.load_state_dict(torch.load(FINETUNED_WEIGHTS_PATH, map_location=device, weights_only=True))
#         input_prompt = format_instruction_prompt(input_prompt)
#     else:
#         print("No finetuned weights found, loading OpenAI GPT-2 weights")
#         settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
#         load_weights_into_gpt(gpt, params)

#     gpt.to(device)
#     gpt.eval()

#     tokenizer = tiktoken.get_encoding("gpt2")
#     # torch.manual_seed(123)

#     token_ids = generate(
#         model=gpt,
#         idx=text_to_token_ids(input_prompt, tokenizer).to(device),
#         max_new_tokens=100,
#         context_size=gpt_config["context_length"],
#         top_k=10,
#         temperature=1.0,
#         eos_id=50256
#     )

#     output_text = token_ids_to_text(token_ids, tokenizer)

#     # Extract just the response when using finetuned model
#     if "### Response:" in output_text:
#         output_text = output_text.split("### Response:")[-1].strip()

#     print("Output text:\n", output_text)


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Generate text with a pretrained GPT-2 model.")
#     parser.add_argument(
#         "--prompt",
#         default="Hello! I am your personal assistant.",
#         help="Prompt for text generation."
#     )
#     parser.add_argument(
#         "--device",
#         default="cpu",
#         help="Device for running inference, e.g., cpu, cuda, mps, or auto."
#     )

#     args = parser.parse_args()

#     # torch.manual_seed(123)

#     # CHOOSE_MODEL = "gpt2-small (124M)"
#     INPUT_PROMPT = args.prompt
#     DEVICE = torch.device(args.device)

#     print("PyTorch:", torch.__version__)
#     print("Device:", DEVICE)


#     CONFIG = {
#         "vocab_size": 50257,
#         "context_length": 1024,
#         "drop_rate": 0.0,
#         "qkv_bias": True,
#         "emb_dim": 768,
#         "n_layers": 12,
#         "n_heads": 12,
#     }

#     # model_configs = {
#     #     "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
#     #     "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
#     #     "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
#     #     "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
#     # }

#     # model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

#     # CONFIG.update(model_configs[CHOOSE_MODEL])

#     main(CONFIG, INPUT_PROMPT, "124M", DEVICE)