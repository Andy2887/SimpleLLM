# SimpleLLM

A GPT-2-style language model built from scratch in PyTorch.

## Project Structure

| File | Description |
|------|-------------|
| `llm.py` | Entry point — generates text with a pretrained GPT-2 model |
| `utils.py` | Core model components: `GPTModel`, `TransformerBlock`, `MultiHeadAttention`, tokenizer utilities |
| `gpt_download.py` | Downloads GPT-2 pretrained weights from OpenAI and loads them into my model |

![Alt text](architecture.svg)

## Getting Started

### Prerequisites

- Python 3.10+
- Dependencies: `torch`, `tiktoken`, `tensorflow`, `numpy`, `requests`, `tqdm`

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Usage

```bash
# Generate text with the default prompt
python llm.py

# Custom prompt
python llm.py --prompt "The meaning of life is"

# Use GPU (CUDA or Apple Silicon)
python llm.py --prompt "Once upon a time" --device cuda
python llm.py --prompt "Once upon a time" --device mps
```

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Vocabulary size | 50,257 |
| Context length | 1,024 tokens |
| Embedding dimension | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Total parameters | ~124M |

## TODO

- [ ] Fine-tune the model

## Acknowledgement

My project follows Sebastian Raschka's tutorial: [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch).