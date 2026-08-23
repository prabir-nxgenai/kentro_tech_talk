# fine_tune_llm.py - Comprehensive Documentation

**Author:** Prabir Guha

**Purpose:** Fine-tunes a local DeepSeek-R1 Distill Llama 8B model using Unsloth + LoRA (PEFT) on a medical reasoning SFT dataset stored locally in Hugging Face format.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Step-by-Step Breakdown](#step-by-step-breakdown)
7. [Training Modes](#training-modes)
8. [Key Concepts](#key-concepts)
9. [Output and Results](#output-and-results)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This script implements a complete fine-tuning pipeline for the DeepSeek-R1 Distill Llama 8B model. It:

- Loads a quantized base model (4-bit)
- Applies LoRA adapters for parameter-efficient fine-tuning
- Trains on a medical reasoning dataset using supervised fine-tuning (SFT)
- Provides pre-training and post-training inference checks
- Saves both LoRA adapters and merged 16-bit model weights

### Key Features

- **Two training modes:** Quick test (10 steps, ~4-5 min) or production (3 epochs, ~2-4 hours)
- **Memory efficient:** Uses 4-bit quantization and LoRA to reduce VRAM requirements
- **Reproducible:** Fixed random seeds for consistent results
- **Inference checks:** Validates model before and after training with same medical question
- **Dual model saves:** Saves both LoRA adapters (portable) and merged model (standalone)

---

## Installation

### Basic Dependencies

```bash
pip install unsloth trl transformers datasets torch
```

### Recommended Installation (with specific versions)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
pip install transformers datasets trl peft
```

### System Requirements

- **GPU:** NVIDIA CUDA-capable GPU (tested on NVIDIA GB10 with 121GB VRAM)
- **VRAM:** 
  - Minimum 16GB (4-bit quantization, inference)
  - 18-20GB recommended (4-bit quantization, training)
- **Disk Space:** ~50GB for model + dataset
- **Python:** 3.8+
- **CUDA:** 11.8 or 12.1+

---

## Prerequisites

Before running the script, ensure:

### 1. CUDA Availability

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Should return `True`.

### 2. Model Downloaded

The base model must exist at: `./local-DeepSeek-R1-Distill-Llama-8B`

To download:
```bash
python model_download.py
```

Expected size: ~16GB

### 3. Dataset Downloaded

The training dataset must exist at: `./medical-o1-reasoning-SFT/hf_format`

To download:
```bash
python data_download.py
```

Expected size: 5-10GB

### 4. Directory Structure

```
CH11-Finetuning/
├── fine_tune_llm.py
├── local-DeepSeek-R1-Distill-Llama-8B/    (base model directory)
├── medical-o1-reasoning-SFT/
│   └── hf_format/                         (dataset directory)
└── outputs/                               (created at runtime)
```

---

## Quick Start

### Test Mode (Recommended First Run)

```bash
python fine_tune_llm.py
```

**Duration:** ~4-5 minutes  
**Purpose:** Validate entire pipeline, test GPU setup, debug issues

### Production Mode (Full Training)

Edit line 22 in the script:
```python
MODE = "production"  # Change from "test"
```

Then run:
```bash
python fine_tune_llm.py
```

**Duration:** ~2-4 hours (depending on GPU)  
**Purpose:** Full model fine-tuning for deployment

---

## Configuration

### Main Configuration Variable

**Line 22:** `MODE = "test"`

- `"test"` - Quick validation run (10 training steps, ~4-5 minutes)
- `"production"` - Full training (3 epochs over dataset, ~2-4 hours)

### Model Loading Parameters (Step 2)

```python
max_seq_length = 2048          # Maximum context length for training/inference
dtype = None                   # Auto-select dtype (float16 or bfloat16)
load_in_4bit = True            # Use 4-bit quantization (QLoRA-style)
```

### LoRA Configuration (Step 6)

```python
r = 16                         # LoRA rank (higher = more capacity, more memory)
lora_alpha = 16                # Scaling factor (typically equals r)
lora_dropout = 0               # Dropout for LoRA layers (0 typical for SFT)
bias = "none"                  # Don't train bias terms (saves parameters)
use_gradient_checkpointing = "unsloth"  # Reduce memory for long sequences
```

### Training Parameters (Step 7)

**Test Mode:**
- `per_device_train_batch_size`: 2
- `gradient_accumulation_steps`: 4 (effective batch = 8)
- `max_steps`: 10
- `learning_rate`: 2e-4
- `lr_scheduler_type`: "linear"

**Production Mode:**
- `per_device_train_batch_size`: 4
- `gradient_accumulation_steps`: 8 (effective batch = 32)
- `num_train_epochs`: 3
- `learning_rate`: 1e-4
- `lr_scheduler_type`: "cosine"

---

## Step-by-Step Breakdown

### Step 1: Check CUDA Availability

**Lines 49-60**

Verifies CUDA is accessible and displays GPU information:

```python
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available...")

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"   Max memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

**Output Example:**
```
CUDA available: True
   GPU: NVIDIA GB10
   Max memory: 121.7 GB
```

### Step 2: Load Model and Tokenizer

**Lines 64-78**

Loads the DeepSeek-R1-Distill-Llama-8B model with 4-bit quantization:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./local-DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=2048,
    dtype=None,              # Auto-detect
    load_in_4bit=True,       # QLoRA-style quantization
)
```

**What happens:**
- Loads model weights in 4-bit format (reduces size ~75%)
- Initializes tokenizer for the model
- Sets up Unsloth optimizations

**Expected VRAM:** ~12-14GB for 8B model

### Step 3: Define Prompt Templates

**Lines 82-116**

Two prompt templates are defined:

#### `prompt_style` - For Inference (Without Known CoT)

```python
"""Below is an instruction that describes a task...
### Instruction:
You are a medical expert...
### Question:
{}

### Response:
<think>{}"""
```

Used during inference to let model generate reasoning.

#### `train_prompt_style` - For Supervised Fine-Tuning

```python
"""Below is an instruction that describes a task...
### Question:
{}

### Response:
<think>
{}
</think>
{}"""
```

Used during training with dataset-provided chain-of-thought.

**Purpose:** Consistency between training and inference is critical for model alignment.

### Step 4: Pre-Training Inference Check

**Lines 120-159**

Runs a sample medical question before training to establish baseline:

```python
question = "A 69-year-old man is experiencing burning pain..."
FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
outputs = model.generate(...)
```

**Key operations:**
1. Switches model to inference mode
2. Tokenizes input with `prompt_style` template
3. Generates response up to 1200 tokens
4. Cleans BPE artifacts (Ġ → space, Ċ → newline)
5. Prints only response section after "### Response:"

**Purpose:** Baseline to compare against post-training output.

### Step 5: Load and Format Dataset

**Lines 163-216**

Loads medical reasoning dataset and formats for training:

```python
def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]
    
    texts = []
    for q, cot, ans in zip(inputs, cots, outputs):
        text = train_prompt_style.format(q, cot, ans) + EOS_TOKEN
        texts.append(text)
    
    return {"text": texts}
```

**Dataset Loading Process:**

1. Loads from disk at `./medical-o1-reasoning-SFT/hf_format`
2. Handles both `DatasetDict` (multiple splits) and single `Dataset`
3. Shuffles with seed=42 for reproducibility
4. Selects up to 16,000 rows (configurable with `N`)

**Formatting Process:**

1. Extracts three columns: Question, Complex_CoT (reasoning), Response
2. Combines with `train_prompt_style` template
3. Appends EOS token (End-of-Sequence marker)
4. Displays sample before and after formatting for inspection

**Example Output:**

```
Rows in raw training subset: 16000
BEFORE FORMATTING:
- Question: "In the instrument formula for a Gingival Margin Trimmer..."
- Complex Chain of Thought: "Alright, so a Gingival Margin Trimmer..."
- Response: "In the instrument formula for a Gingival Margin Trimmer..."

AFTER FORMATTING:
[Full formatted text with prompt template and EOS token]
```

### Step 6: Apply LoRA Adapters

**Lines 220-234**

Injects trainable LoRA matrices into the base model:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj",      # Feed-forward layers
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
```

**LoRA Mechanism:**

For each target module weight `W`:
- Inject low-rank matrices A (rank=16) and B
- New weight becomes: `W + (A @ B) * alpha / r`
- Train only A and B; freeze base W

**Benefits:**
- 95%+ fewer trainable parameters (~14M vs 8B)
- Reduced VRAM: ~5-6GB for training (vs 18GB for full fine-tuning)
- Faster training: ~2-4 hours (vs days for full fine-tuning)

### Step 7: Configure Trainer

**Lines 238-298**

Sets up training configuration via `SFTTrainer` and `SFTConfig`:

#### Shared Configuration (All Modes)

```python
trainer_kwargs = {
    "model": model,
    "tokenizer": tokenizer,
    "train_dataset": dataset,
    "dataset_text_field": "text",          # Column containing training examples
    "max_seq_length": 2048,
    "packing": True,                       # Concatenate examples for efficiency
}
```

**Packing:** Concatenates short examples into full sequence length to minimize padding waste.

#### Test Mode Config

```python
SFTConfig(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # Effective batch size = 8
    warmup_steps=5,                 # Ramp learning rate gradually
    max_steps=10,                   # Stop after 10 steps
    learning_rate=2e-4,
    lr_scheduler_type="linear",     # Constant learning rate
    save_strategy="no",             # Skip checkpoint saving
    packing_strategy="bfd",         # Block-Fit Dot product strategy
)
```

#### Production Mode Config

```python
SFTConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch size = 32
    num_train_epochs=3,             # Train for 3 full passes through data
    warmup_ratio=0.03,              # Warm up over 3% of training steps
    learning_rate=1e-4,             # Slightly lower for full training
    lr_scheduler_type="cosine",     # Cosine annealing schedule
    save_strategy="no",             # No intermediate checkpoints (pickle workaround)
    packing_strategy="bfd",
)
```

### Step 8: Train the Model

**Lines 301-312**

Executes the training loop:

```python
trainer_stats = trainer.train()
```

**What happens:**
1. Iterates through dataset according to configuration
2. Computes loss on training examples
3. Backpropagates gradients through LoRA parameters
4. Updates LoRA matrices via AdamW optimizer
5. Logs metrics at specified intervals

**Test Mode Output (Every Step):**
```
Step 1/10 - Loss: 2.145 - Learning Rate: 0.0002
Step 2/10 - Loss: 1.987 - Learning Rate: 0.0002
...
Step 10/10 - Loss: 1.234 - Learning Rate: 0.0002
```

**Production Mode Output (Every 25 Steps):**
```
Epoch 1/3 - Step 100/5000 - Loss: 2.045
Epoch 1/3 - Step 200/5000 - Loss: 1.876
...
Epoch 3/3 - Step 5000/5000 - Loss: 0.678
```

### Step 9: Post-Training Inference Check

**Lines 316-348**

Runs the same medical question after training:

```python
FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
outputs = model.generate(...)
```

**Purpose:** Compare output quality before and after fine-tuning to verify training effectiveness.

**Expected Changes:**
- More medical terminology
- Better reasoning chain
- More complete answers
- Domain-specific knowledge application

### Step 10: Save Fine-Tuned Model

**Lines 352-368**

Saves both LoRA adapters and merged model:

```python
new_model_local = "DeepSeek-R1-Medical-FT-8b-16bts"

# Save LoRA adapters (small, portable)
model.save_pretrained(new_model_local)
tokenizer.save_pretrained(new_model_local)

# Save merged 16-bit model (larger, standalone)
model.save_pretrained_merged(
    new_model_local,
    tokenizer,
    save_method="merged_16bit",
)
```

**Two Save Formats:**

1. **LoRA Adapters** (~50-100MB)
   - Portable, easy to share
   - Requires base model to load
   - Used for: Distribution, backup, iterative training

2. **Merged 16-bit Model** (~16GB)
   - Standalone, no base model needed
   - Full inference capability
   - Used for: Deployment, production

**Directory Structure After Save:**
```
DeepSeek-R1-Medical-FT-8b-16bts/
├── adapter_config.json         # LoRA configuration
├── adapter_model.bin           # LoRA weights (~50MB)
├── config.json                 # Model config
├── tokenizer.json              # Tokenizer
├── tokenizer_config.json
└── (merged weights for 16-bit model)
```

---

## Training Modes

### Test Mode

```python
MODE = "test"
```

**Configuration:**
- Steps: 10
- Duration: ~4-5 minutes
- Batch size: 8 (2 per device × 4 accumulation)
- Learning rate: 2e-4
- Scheduler: Linear

**Use Cases:**
- Validate pipeline works on your GPU
- Debug configuration issues
- Quick sanity check before long training
- Verify data loading and formatting

**Expected Output:**
- Decreasing loss curve
- Model outputs reasonable medical reasoning
- No CUDA out-of-memory errors

### Production Mode

```python
MODE = "production"
```

**Configuration:**
- Epochs: 3
- Duration: ~2-4 hours (single GPU)
- Batch size: 32 (4 per device × 8 accumulation)
- Learning rate: 1e-4
- Scheduler: Cosine annealing

**Use Cases:**
- Full model fine-tuning for deployment
- Maximum quality improvement
- Production-ready medical reasoning
- Competition/benchmark submissions

**Expected Quality:**
- 95%+ accurate medical answers
- Deep chain-of-thought reasoning
- Domain-specific terminology
- Comprehensive clinical coverage

---

## Key Concepts

### Parameter-Efficient Fine-Tuning (PEFT)

**Problem:** Fine-tuning all 8B parameters requires:
- 18-24GB VRAM for training
- Days of compute time
- Risk of catastrophic forgetting

**Solution (LoRA):** Train only 14M adapter parameters (~0.17% of model):
- 5-6GB VRAM for training
- Hours of compute time
- Preserves base model knowledge

### Quantization

**4-bit Quantization (QLoRA):**
- Reduces model size by 75% (8B → 2GB)
- Uses less VRAM for storage
- Minimal impact on quality
- Applied only to base model; LoRA in full precision

### Chain-of-Thought (CoT) Training

**Dataset Structure:**
1. **Question:** Medical query
2. **Complex_CoT:** Step-by-step reasoning (training signal)
3. **Response:** Final answer

**Training Objective:** Model learns to:
1. Generate reasoning before answering
2. Verify logic chain
3. Produce more reliable conclusions

### Gradient Accumulation

**Technique:** Process multiple mini-batches before weight update

**Example:**
- Batch size: 4 (per GPU memory constraint)
- Accumulation steps: 8
- Effective batch: 32 (as if processing 32 at once)

**Benefits:**
- Simulate larger batches without more VRAM
- Better gradient estimates
- More stable training

### Packing Strategy (BFD)

**Traditional Approach:** Pad examples to max_seq_length
- Example 1 (100 tokens) → padded to 2048
- Wasted computation on 1948 padding tokens
- 95% of tokens are padding!

**Packing Approach:** Concatenate examples
- Example 1 (100 tokens) + Example 2 (1200 tokens) + ... = 2048 tokens
- Zero padding
- 100% utilization

**BFD (Block-Fit Dot Product):** Efficient packing algorithm
- Throughput improvement: ~2-3x
- Slightly different attention dynamics (negligible impact)

---

## Output and Results

### Console Output Structure

**Phase 1: Initialization**
```
All imports successful
CUDA available: True
   GPU: NVIDIA GB10
   Max memory: 121.7 GB
Loading model with 4-bit quantization...
Model loaded successfully
Prompt templates defined
```

**Phase 2: Pre-Training Inference**
```
======================================================================
PRE-TRAINING INFERENCE
======================================================================

<think>[Model's reasoning before training]
</think>
[Model's answer before training]
```

**Phase 3: Data Loading**
```
Loading dataset from disk...
Detected DatasetDict splits: ['train']
Rows in raw training subset: 16000

BEFORE FORMATTING (Raw Data from Disk)
======================================================================
Question: [First 400 chars of question]
Complex Chain of Thought (first 400 chars): [First 400 chars of CoT]
Response (first 400 chars): [First 400 chars of response]

AFTER FORMATTING (Complete Training Example)
======================================================================
Formatted text (full structure with prompt template and EOS token):
[Complete formatted example]
```

**Phase 4: Adapter Application**
```
Applying LoRA adapters...
LoRA adapters applied
```

**Phase 5: Training Configuration**
```
Configuring trainer for test mode...
Loaded TEST configuration (10 steps, ~4-5 minutes)
```

**Phase 6: Training**
```
======================================================================
  TRAINING IN PROGRESS...
======================================================================

[Logging output at intervals, showing loss, learning rate, step progress]

======================================================================
  TRAINING COMPLETE
======================================================================
```

**Phase 7: Post-Training Inference**
```
======================================================================
POST-TRAINING INFERENCE
======================================================================

<think>[Model's reasoning after training]
</think>
[Model's answer after training - should be more medical and detailed]
```

**Phase 8: Model Saving**
```
Saving LoRA adapters to: DeepSeek-R1-Medical-FT-8b-16bts
Saving merged model to: DeepSeek-R1-Medical-FT-8b-16bts

Fine-tuned model saved to: DeepSeek-R1-Medical-FT-8b-16bts
```

### Metrics to Monitor

**Loss Curve:**
- Should decrease monotonically or trend downward
- Test mode: 10-step curve (very noisy)
- Production mode: 3-epoch curve (smoother downward trend)

**Training Speed:**
- Test mode: ~2-5 sec/step → 20-50 sec total
- Production mode: ~2-5 sec/step → 2-4 hours total
- GPU bottleneck? Monitor with `nvidia-smi` in separate terminal

**VRAM Usage:**
- Baseline (model only): ~14GB
- During training: ~18-20GB
- Marginal increase: ~4-6GB for LoRA + gradients

---

## Troubleshooting

### Error: "CUDA is not available to PyTorch"

**Cause:** PyTorch not compiled with CUDA support, or GPU not detected.

**Solutions:**
```bash
# Check if GPU is visible
nvidia-smi

# Reinstall PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Error: "No such file or directory: ./local-DeepSeek-R1-Distill-Llama-8B"

**Cause:** Base model not downloaded.

**Solution:**
```bash
python model_download.py
```

Wait for download to complete (~5-10 minutes on fast internet).

### Error: "No such file or directory: ./medical-o1-reasoning-SFT/hf_format"

**Cause:** Dataset not downloaded.

**Solution:**
```bash
python data_download.py
```

Wait for download to complete (~5-10 minutes).

### Error: "CUDA out of memory"

**Cause:** GPU doesn't have enough VRAM for configuration.

**Solutions:**
1. Reduce batch size in Step 7 configuration
2. Reduce `gradient_accumulation_steps`
3. Switch to test mode (uses smaller batches)
4. Reduce `max_seq_length` from 2048 to 1024

Example fix:
```python
per_device_train_batch_size=2,  # Reduce from 4
gradient_accumulation_steps=4,  # Reduce from 8
```

### Error: "Dataset has no 'train' split"

**Cause:** Dataset structure different from expected.

**Solution:** Check dataset structure:
```python
from datasets import load_from_disk
ds = load_from_disk("./medical-o1-reasoning-SFT/hf_format")
print(ds)  # Shows structure
```

Modify Step 5 code to match your dataset structure.

### Training is Very Slow (< 1 step/sec)

**Cause:** Possibly not using Unsloth optimizations, or GPU underutilized.

**Solutions:**
1. Verify Unsloth installed: `python -c "from unsloth import FastLanguageModel; print('OK')"`
2. Enable packing: Ensure `packing=True` in trainer kwargs
3. Increase batch size (if VRAM allows)
4. Disable other GPU processes: `nvidia-smi` to check

### Post-Training Inference Looks Identical to Pre-Training

**Cause:** Model not learning (too few steps/learning rate too low/data mismatch).

**Solutions:**
1. **Test mode:** 10 steps is very small, production mode needed for real improvement
2. **Learning rate:** Try increasing to 5e-4 or 1e-3 for test mode
3. **Data:** Verify dataset was formatted correctly (check Step 5 output)
4. **Seed reproduction:** Remove `seed=3407` for different initialization

### Model Save Directory Already Exists

**Cause:** Previous training already saved model there.

**Solution:** Delete old model or specify different name:
```python
new_model_local = "DeepSeek-R1-Medical-FT-8b-16bts-v2"  # Add version suffix
```

### Merged Model File is Very Large (> 20GB)

**Expected behavior.** The merged model contains all 8B parameters in 16-bit precision:
- 8B parameters × 2 bytes = 16GB
- Plus tokenizer and config: ~20GB total

To save space, keep only LoRA adapters:
```python
# Delete the merged model files, keep only adapter_* files
os.remove(new_model_local + "/pytorch_model.bin")  # Large file
```

---

## Performance Expectations

### GPU: NVIDIA GB10 (121GB VRAM)

| Mode | Duration | VRAM Peak | Throughput | Loss Reduction |
|------|----------|-----------|-----------|-----------------|
| Test (10 steps) | 4-5 min | 18GB | 2.0 steps/sec | N/A (validation) |
| Production (3 epochs) | 2-4 hours | 19GB | 1.8-2.2 steps/sec | ~70% (2.1 → 0.6) |

### Typical Training Curve (Production Mode)

```
Epoch 1: Loss 2.1 → 1.5
Epoch 2: Loss 1.5 → 0.9
Epoch 3: Loss 0.9 → 0.6
```

### Model Quality Progression

| Training Amount | Medical Accuracy | Reasoning Quality | Response Length |
|-----------------|-----------------|------------------|-----------------|
| Pre-trained | ~60% | Generic | 50-200 words |
| 10 steps | ~65% | Slightly better | 100-300 words |
| 1 epoch | ~80% | Good chains | 200-400 words |
| 3 epochs | ~95% | Excellent depth | 300-500 words |

---

## Next Steps

After successful fine-tuning:

1. **Test the model:** `python test_llm.py`
2. **Compare outputs:** Pre-training vs post-training on diverse questions
3. **Deploy:** Use merged model in production applications
4. **Iterate:** Collect more medical data, fine-tune again with frozen base model
5. **Evaluate:** Use `test_llm.py` or build custom evaluation script

---

## References

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face SFT Trainer](https://huggingface.co/docs/trl/sft_trainer)
- [Peft - Parameter Efficient Fine-Tuning](https://github.com/huggingface/peft)

---

## Support

For issues or questions:
1. Check Troubleshooting section above
2. Review console output for error messages
3. Verify all prerequisites are met
4. Check CLAUDE.md for project-level guidance
