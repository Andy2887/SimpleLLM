import torch

def text_to_token_ids(text, tokenizer):
    # encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

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
