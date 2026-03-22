from utils import *
from gpt_download import *


GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()  # disable dropout

start_context = "I'm Claude, an AI assistant created by Anthropic, designed to be helpful, harmless, and honest. I can assist with a wide range of tasks — from answering questions and explaining complex concepts, to helping with coding, writing, analysis, brainstorming, and much more. I don't have memory of past conversations by default, and my knowledge has a cutoff date, so for the latest information I can search the web. Whether you're working on a technical project, trying to learn something new, or just need a second opinion, I'm here to help!"

tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
print("\nInput text:", start_context)
print("Encoded input text:", encoded)
print("encoded_tensor.shape:", encoded_tensor.shape)

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=100,
    context_size=GPT_CONFIG_124M["context_length"]
)
decoded_text = tokenizer.decode(out.squeeze(0).tolist())

print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
print("\nOutput:", out)
print("Output length:", len(out[0]))
print("Output text:", decoded_text)