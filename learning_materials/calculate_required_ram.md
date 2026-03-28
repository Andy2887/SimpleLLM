# Calculate Required RAM

How to know how much RAM you need on your GPU?

## Full Fine-Tuning

```
Total RAM = Model Weights + Optimizer States + Gradients + Activations
```

### 1. Model Weights

fp32 -> 4 Bytes
bf16 -> 2 Bytes

```
# 1GB is 1,000,000,000 bytes
Memory (GB) = (num_parameters × bytes_per_param) / 1e9
```

### 2. Optimizer States

Mixed Precision Training (standard practice):

```
Optimizer total = Master weights (fp32) + Adam moments (fp32)
                = num_params × 4 bytes + 2 × num_params × 4 bytes
                = 3 × num_params × 4 bytes
```

### 3. Gradients

One gradient per parameter

```
Gradients = num_params × bytes_per_param
```

### 4. Activation

This is the hardest to estimate.

```
Activations ≈ batch_size × seq_len × hidden_dim × num_layers × bytes_per_param
```

## Inference

```
Total RAM = Model Weights + KV cache + Activations
```

### 5. KV Cache

```
KV cache bytes = 2 × num_layers × hidden_dim × seq_len × bytes_per_element (fp32 or bf16)
```

