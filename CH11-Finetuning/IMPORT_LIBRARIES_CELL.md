# Import Libraries Cell - Detailed Breakdown

## Overview

This document explains every library imported in Step 1 of `fine_tune_llm.ipynb`, what each does, and where to find official documentation.

---

## Complete Import Statement

```python
from unsloth import FastLanguageModel
import torch
#import os
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
#from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
```

---

## 1. Unsloth: `FastLanguageModel` and `is_bfloat16_supported`

### What is Unsloth?

Unsloth is a **high-performance optimization library** that makes LLM fine-tuning 2-3x faster and uses 30-50% less GPU memory through custom CUDA kernels and optimized algorithms.

### FastLanguageModel

**Purpose:** Optimized model loading and inference wrapper

**Key Features:**
- **4-bit Quantization**: Automatically loads models in 4-bit (NF4 format), reducing 16GB models to ~4GB
- **LoRA Optimization**: Applies Low-Rank Adaptation with custom kernels (2-3x speedup)
- **Gradient Checkpointing**: Saves memory by recomputing activations instead of storing them
- **Inference Mode**: Provides `.for_inference()` for 2x faster token generation
- **Automatic GPU Placement**: Distributes model across available GPUs

**Usage in Notebook:**

```python
# Model loading with automatic optimization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./local-DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,  # Reduces model size 4x
)

# Inference mode (2x faster)
FastLanguageModel.for_inference(model)

# LoRA adapter application
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", ...],
)
```

### is_bfloat16_supported()

**Purpose:** GPU precision detection

**What it does:**
- Checks if your GPU supports **BFloat16** (Brain Float 16-bit)
- BFloat16 is faster but less precise than FP16
- Automatically chooses best precision for your hardware

**Return Values:**
- `True`: GPU supports BF16 → Use for faster training
- `False`: GPU doesn't support BF16 → Fall back to FP16

**Usage in Notebook:**

```python
# Automatic precision selection based on GPU
fp16=not is_bfloat16_supported(),  # Use FP16 if no BF16
bf16=is_bfloat16_supported(),      # Use BF16 if available
```

**Precision Comparison:**
| Precision | Speed | Memory | Quality | Use Case |
|-----------|-------|--------|---------|----------|
| FP32 | Slowest | Most | Best | Inference |
| FP16 | Medium | Medium | Good | Training (older GPUs) |
| BF16 | Fastest | Least | Good | Training (newer GPUs) |
| INT8/INT4 | Fast | Least | Fair | Inference only |

### Official Documentation

- 🔗 **GitHub**: https://github.com/unslothai/unsloth
- 🔗 **Website**: https://unsloth.ai/
- 🔗 **Installation**: https://github.com/unslothai/unsloth#-installation-instructions
- 🔗 **API Reference**: https://github.com/unslothai/unsloth/wiki

### Performance Impact

```
Without Unsloth:
  Training time: 2-4 hours for 3 epochs
  VRAM used: 8-9 GB
  Training speed: 0.4 samples/sec

With Unsloth:
  Training time: 45 min - 1.5 hours for 3 epochs  ← 2-3x faster
  VRAM used: 5-6 GB                               ← 33% less
  Training speed: 1.2-1.5 samples/sec             ← 3x throughput
```

---

## 2. PyTorch: `torch`

### What is PyTorch?

PyTorch is the **fundamental deep learning framework** powering everything in this notebook. Without it, GPU acceleration, automatic differentiation, and neural network operations wouldn't work.

### Key Responsibilities

**GPU Acceleration:**
```python
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

print(torch.cuda.get_device_name(0))  # "NVIDIA GB10"
print(torch.cuda.get_device_properties(0))  # Memory, compute capability
```

**Automatic Differentiation (Backpropagation):**
- Computes gradients automatically during training
- Used internally by `trainer.train()`
- No manual gradient computation needed

**Tensor Operations:**
```python
# All model computations use PyTorch tensors
inputs = tokenizer([prompt], return_tensors="pt")  # PyTorch format
outputs = model.generate(**inputs)  # PyTorch operations
```

### Common PyTorch Concepts in Fine-tuning

| Concept | What it does | Used in notebook |
|---------|-------------|------------------|
| **CUDA** | GPU acceleration backend | Model loading, training |
| **Tensors** | Multi-dimensional arrays | Input/output data |
| **Autograd** | Automatic differentiation | Gradient computation |
| **Device Management** | CPU/GPU placement | `.to("cuda")` calls |
| **Mixed Precision** | FP16/BF16 training | Precision selection |

### Official Documentation

- 🔗 **Official Site**: https://pytorch.org/
- 🔗 **Documentation**: https://pytorch.org/docs/stable/index.html
- 🔗 **CUDA & GPU Setup**: https://pytorch.org/get-started/locally/
- 🔗 **Tensors Tutorial**: https://pytorch.org/tutorials/beginner/basics/tensors_tutorial.html
- 🔗 **Autograd (Gradients)**: https://pytorch.org/docs/stable/autograd.html
- 🔗 **CUDA Semantics**: https://pytorch.org/docs/stable/notes/cuda.html

### PyTorch in Training Pipeline

```
Input Data
    ↓
PyTorch Tensors (.to("cuda"))
    ↓
Forward Pass (Model computation)
    ↓
Loss Computation
    ↓
Backward Pass (Autograd computes gradients)
    ↓
Optimizer Step (Update weights)
    ↓
Next Iteration
```

---

## 3. Hugging Face Datasets: `load_from_disk`

### What is Hugging Face Datasets?

A library for **efficient dataset loading, caching, and processing**. Handles datasets in optimized binary formats (Arrow/Parquet) instead of JSON/CSV.

### load_from_disk() Function

**Purpose:** Load pre-processed datasets from disk in milliseconds

**What it does:**
- Reads Arrow/Parquet format (columnar, compressed)
- Memory-mapped access (doesn't load entire dataset into RAM)
- Automatic batching support
- Integrates seamlessly with TRL trainers

**Usage in Notebook:**

```python
# Load 16,000 medical Q&A examples
dataset_on_disk = load_from_disk("./medical-o1-reasoning-SFT/hf_format")

# Datasets automatically handles:
# - Batching for training
# - Shuffling
# - Caching
# - Format conversions
```

### Dataset Format Used

```
./medical-o1-reasoning-SFT/hf_format/
├── dataset_info.json          # Metadata
├── state.json                 # State tracking
└── data/
    ├── train-00000-of-00001/  # Train split
    │   ├── dataset.arrow      # Actual data (binary)
    │   └── indices.arrow      # Row indices
    └── ...
```

This format is:
- **Fast**: O(1) random access to rows
- **Compact**: 50-70% smaller than JSON
- **Memory-efficient**: Loads only needed chunks

### Comparison: load_from_disk vs Manual Loading

```python
# With Hugging Face Datasets (fast, memory-efficient)
dataset = load_from_disk("./data/hf_format")  # ~100ms for 16K rows
# Memory: Only loaded data in batches

# vs Manual JSON loading (slow, memory-heavy)
import json
with open("data.json") as f:
    data = json.load(f)  # ~2-3 seconds for 16K rows
# Memory: Entire 500MB JSON in RAM at once
```

### Official Documentation

- 🔗 **Main Docs**: https://huggingface.co/docs/datasets
- 🔗 **Loading Datasets**: https://huggingface.co/docs/datasets/loading
- 🔗 **Dataset Processing**: https://huggingface.co/docs/datasets/process
- 🔗 **Save & Load**: https://huggingface.co/docs/datasets/save_load
- 🔗 **API Reference**: https://huggingface.co/docs/datasets/package_reference/main_classes

### Data Workflow in Notebook

```
data_download.py
    ↓
Downloads: huggingface_hub.snapshot_download()
    ↓
./medical-o1-reasoning-SFT/hf_format/ (Arrow format)
    ↓
load_from_disk() in notebook
    ↓
Dataset object with 16,000 examples
    ↓
Formatting function (train_prompt_style)
    ↓
SFTTrainer (handles batching)
```

---

## 4. TRL: `SFTTrainer` and `SFTConfig`

### What is TRL?

**Transformer Reinforcement Learning** is Hugging Face's library for LLM training, specifically optimized for:
- Supervised Fine-Tuning (SFT)
- Reinforcement Learning from Human Feedback (RLHF)
- Direct Preference Optimization (DPO)

In this notebook, we use **SFT** (Supervised Fine-Tuning).

### SFTTrainer

**Purpose:** High-level training loop for fine-tuning LLMs

**What it Handles Automatically:**
- Dataset batching and shuffling
- Gradient accumulation
- Mixed precision training
- Learning rate scheduling
- Checkpoint saving
- Metric logging
- Distributed training (multi-GPU)
- Token packing for efficiency

**Usage in Notebook:**

```python
trainer = SFTTrainer(
    model=model,                    # Model with LoRA adapters
    tokenizer=tokenizer,            # Tokenizer for encoding
    train_dataset=dataset,          # Training data
    dataset_text_field="text",      # Column name with formatted text
    max_seq_length=2048,            # Max input length
    packing=True,                   # Combine examples for efficiency
    args=SFTConfig(...),            # Training configuration
)

# Train for 10 steps (about 5 minutes)
trainer.train()
```

### SFTConfig

**Purpose:** Configuration object for training hyperparameters (replaces Transformers' TrainingArguments)

**Key Parameters:**

| Parameter | Purpose | Value in Notebook |
|-----------|---------|------------------|
| `per_device_train_batch_size` | Examples per GPU | 2 (test) / 4 (prod) |
| `gradient_accumulation_steps` | Gradient accumulation | 4 (test) / 8 (prod) |
| `max_steps` | Total training steps | 10 (test only) |
| `num_train_epochs` | Epochs to train | 3 (production) |
| `learning_rate` | LR for LoRA | 2e-4 (test) / 1e-4 (prod) |
| `warmup_steps` | Warmup iterations | 5 (test) / ratio 0.03 (prod) |
| `lr_scheduler_type` | LR decay strategy | linear (test) / cosine (prod) |
| `optim` | Optimizer type | adamw_8bit (Unsloth optimized) |
| `logging_steps` | Log every N steps | 10 (test) / 25 (prod) |
| `save_strategy` | Checkpoint saving | "no" (disable, use memory) |
| `packing` | Combine examples | True |
| `packing_strategy` | Packing algorithm | "bfd" (Block-Fit Dot) |
| `bf16` | Use BFloat16 | True (if GPU supports) |
| `fp16` | Use Float16 | True (if no BF16) |

**SFTConfig vs TrainingArguments:**
```python
# Old way (Transformers)
from transformers import TrainingArguments
args = TrainingArguments(output_dir="outputs", ...)

# New way (TRL) - LLM-specific, more features
from trl import SFTConfig
args = SFTConfig(packing=True, packing_strategy="bfd", ...)
```

### Official Documentation

- 🔗 **TRL GitHub**: https://github.com/huggingface/trl
- 🔗 **SFTTrainer Docs**: https://huggingface.co/docs/trl/en/sft_trainer
- 🔗 **SFTConfig Reference**: https://huggingface.co/docs/trl/en/sft_trainer#trl.SFTConfig
- 🔗 **Examples**: https://github.com/huggingface/trl/tree/main/examples
- 🔗 **Full API**: https://huggingface.co/docs/trl/en/trainer_base

### Training Workflow

```
SFTTrainer.train()
    ↓
For each step:
    ├── Load batch from dataset
    ├── Tokenize and pack examples
    ├── Forward pass (model inference)
    ├── Compute loss
    ├── Backward pass (gradients)
    ├── Accumulate gradients (4 steps)
    ├── Optimizer step (weight update)
    └── Log metrics
    ↓
Save fine-tuned model
```

---

## 5. Commented-Out Libraries (Why Removed)

### `#import os`

**Original Purpose:**
```python
import os
os.path.exists("./outputs")
os.makedirs("./outputs", exist_ok=True)
```

**Why Removed:**
- Unsloth automatically handles path creation
- SFTConfig creates `output_dir` automatically
- No longer needed with modern libraries

### `#from transformers import TrainingArguments`

**Original Purpose:**
```python
from transformers import TrainingArguments
args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,
    ...
)
```

**Why Removed:**
- **Version Conflict**: TrainingArguments was creating `push_to_hub_token` field internally
- **Incompatibility**: Older SFTConfig versions don't accept `push_to_hub_token`
- **Better Alternative**: SFTConfig is purpose-built for LLM training
- **Feature Gap**: SFTConfig has LLM-specific features like `packing` and `packing_strategy`

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│   Your Fine-tuning Code (Notebook)      │
└────────────────┬────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌──────────┐ ┌────────┐ ┌─────────────┐
│ Unsloth  │ │ PyTorch│ │ TRL Trainer │
│  (Opt)   │ │ (Core) │ │  (Training) │
└────┬─────┘ └───┬────┘ └──────┬──────┘
     │           │              │
     └───────────┼──────────────┘
                 │
              ┌──▼──────────────────────┐
              │  Hugging Face Datasets   │
              │  (Data Loading)          │
              └──────────────┬───────────┘
                             │
                    ┌────────▼─────────┐
                    │  GPU (NVIDIA)    │
                    │  VRAM: 121.7 GB  │
                    └──────────────────┘
```

### Data Flow

```
Dataset (16K examples)
    ↓
[Datasets] load_from_disk()
    ↓
[Formatting] train_prompt_style format
    ↓
[SFTTrainer] Batch creation & packing
    ↓
[PyTorch] Convert to tensors, move to GPU
    ↓
[Unsloth] Optimized model forward pass
    ↓
[Loss computation & backprop]
    ↓
[Weight updates via optimizer]
    ↓
[Fine-tuned model saved]
```

---

## Version Compatibility

The notebook was tested with:

| Library | Version | Purpose |
|---------|---------|---------|
| **Unsloth** | 2026.7.6+ | 2-3x speedup, memory optimization |
| **PyTorch** | 2.11.0+ | Deep learning framework |
| **CUDA** | 12.1 | GPU acceleration |
| **Transformers** | 5.14.1+ | Model & tokenizer loading |
| **TRL** | Latest | SFT trainer & config |
| **Datasets** | Latest | Data loading |
| **PEFT** | Latest | LoRA adapters |

---

## Performance Metrics by Library

### Unsloth Contribution
- **Speed**: 2-3x faster (custom CUDA kernels)
- **Memory**: 30-50% less (gradient offloading)
- **Compatibility**: Works with TRL, Transformers, PEFT

### PyTorch Contribution
- **GPU Utilization**: ~70-80% (parallel operations)
- **Precision**: Configurable (FP16, BF16, FP32)
- **Stability**: Production-ready, heavily tested

### TRL Contribution
- **Ease of Use**: 200+ lines of code → 10 lines
- **Features**: Automatic batching, logging, checkpointing
- **Flexibility**: Supports SFT, RLHF, DPO training paradigms

### Datasets Contribution
- **Speed**: 100x faster than JSON/CSV loading
- **Memory**: 50-70% more compact than JSON
- **Integration**: Native support for batching & shuffling

---

## Quick Reference

### Import Everything Needed
```python
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
```

### Key Functions per Library
```python
# Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(model, ...)
FastLanguageModel.for_inference(model)
is_bf16_supported = is_bfloat16_supported()

# PyTorch
torch.cuda.is_available()
torch.cuda.get_device_name(0)
tensor = torch.tensor([...]).to("cuda")

# Datasets
dataset = load_from_disk("./data/hf_format")
dataset.map(formatting_function, batched=True)

# TRL
trainer = SFTTrainer(model=model, args=SFTConfig(...))
trainer.train()
```

---

## Further Learning

1. **Understanding Fine-tuning**: [HF Course - Fine-tuning](https://huggingface.co/course/chapter3/3)
2. **LoRA Explained**: [LoRA Paper](https://arxiv.org/abs/2106.09685)
3. **GPU Memory Optimization**: [Hugging Face Memory](https://huggingface.co/docs/transformers/performance/training)
4. **Unsloth Deep Dive**: [Unsloth Blog](https://unsloth.ai/)
5. **CUDA Basics**: [NVIDIA CUDA Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

**Document Version**: 1.0  
**Last Updated**: 2026-08-08  
**Notebook**: fine_tune_llm.ipynb (Step 1)
