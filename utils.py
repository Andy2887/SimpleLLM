import torch

def text_to_token_ids(text, tokenizer):
    # encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None, use_cache=True):

    if use_cache:
        cache = KVCache(model.cfg["n_layers"])
        model.reset_kv_cache()
    else:
        cache = None

    for _ in range(max_new_tokens):

        if use_cache:
            if model.current_pos == 0:
                # Prefill: process entire prompt
                input_ids = idx
            else:
                # Decode: only the last token (previous K/V are cached)
                input_ids = idx[:, -1:]
        else:
            input_ids = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(input_ids, cache=cache)

        if use_cache:
            model.current_pos += input_ids.shape[1]

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
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def format_instruction_prompt(prompt):
    """Wrap user prompt in Alpaca-style instruction template for finetuned model."""
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt}\n\n"
        f"### Response:\n"
    )

class KVCache:
    def __init__(self, n_layers):
        self.cache = [None] * n_layers

    def get(self, layer_idx):
        return self.cache[layer_idx]

    def update(self, layer_idx, value):
        self.cache[layer_idx] = value

    def get_all(self):
        return self.cache

    def reset(self):
        for i in range(len(self.cache)):
            self.cache[i] = None
