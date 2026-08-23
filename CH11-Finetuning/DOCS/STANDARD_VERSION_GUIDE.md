# Standard Version Guide (`finetune-v2-standard.ipynb`)

## Overview

This guide covers the **Standard Version** of fine-tuning using only core libraries:
- **Transformers** - Model loading and generation
- **PEFT** - LoRA (Parameter-Efficient Fine-Tuning)
- **TRL** - SFTTrainer for supervised fine-tuning
- **BitsAndBytes** - 4-bit quantization
- **PyTorch** - GPU acceleration

No Unsloth required - **maximum portability and clarity**.

---

## Step-by-Step Walkthrough

### Step 0: Auto-Download Dataset & Model

```python
# Downloads:
# 1. Dataset: python_code_instructions_18k_alpaca (18,612 examples)
# 2. Model: meta-llama/Llama-2-7b-hf (~13 GB)
```

**Output:**
```
✅ Dataset already exists at ./python_code_instructions_18k_alpaca/hf_format
✅ Model already exists at ./llama-2-7b-hf
✅ ALL DOWNLOADS COMPLETE
```

**Note:** Uses Llama 2 instead of Llama 3.1 (Llama 3.1 requires special access token)

---

### Step 1: Import Libraries

All standard libraries - nothing special needed.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer
```

---

### Step 2: Configuration

Three key configurations:

#### A. 4-bit Quantization (BitsAndBytes)
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable 4-bit
    bnb_4bit_quant_type="nf4",            # NormalFloat 4-bit (best quality)
    bnb_4bit_use_double_quant=True,       # Double quantization
    bnb_4bit_compute_dtype=torch.bfloat16, # Computation precision
)
```

What each does:
- `load_in_4bit=True`: Reduces model size 4x (7B → ~2GB)
- `bnb_4bit_quant_type="nf4"`: Best quality for 4-bit
- `bnb_4bit_use_double_quant=True`: Further compression
- `bnb_4bit_compute_dtype=torch.bfloat16`: Use BF16 for math (if available)

#### B. LoRA Configuration (PEFT)
```python
lora_config = LoraConfig(
    r=16,                      # Rank (8-32 typical, higher = more params)
    lora_alpha=32,             # Scaling factor
    lora_dropout=0.05,         # Dropout for regularization
    bias="none",               # No bias in LoRA
    task_type=TaskType.CAUSAL_LM,  # Task type
    target_modules=[           # Which layers to apply LoRA to
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)
```

LoRA Hyperparameter Guide:
- `r=16`: Good balance (8 = smaller, 32 = larger)
- `lora_alpha=32`: Scale factor (2x r is typical)
- `lora_dropout=0.05`: Regularization (prevent overfitting)

#### C. Prompt Template
```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input...
### Instruction:
{}
### Input:
{}
### Response:
{}"""
```

---

### Step 3: Load Model & Test Before Training

```python
# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,  # Apply 4-bit
    device_map="auto",               # Auto GPU placement
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

**Output:** Shows inference before training (baseline behavior)

---

### Step 4: Load & Format Dataset

```python
def formatting_func(examples):
    # Format each example with Alpaca template
    texts = []
    for instruction, input_text, output_text in zip(...):
        text = alpaca_prompt.format(instruction, input_text, output_text)
        texts.append(text)
    return {"text": texts}

# Load and format
dataset = load_from_disk(dataset_path)
dataset = dataset.map(formatting_func, batched=True)
```

Dataset becomes: list of formatted prompt strings

---

### Step 5: Setup LoRA

```python
from peft import get_peft_model

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

**Output:**
```
trainable params: 41,943,040 || all params: 8,072,204,288 || trainable%: 0.52%
```

Only 0.52% of parameters need training!

---

### Step 6: Configure Trainer

```python
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=2,      # Batch size per GPU
    gradient_accumulation_steps=4,      # Accumulate 4 steps before update
    warmup_steps=5,                     # Linear warmup
    max_steps=100,                      # 100 steps for testing
    learning_rate=2e-4,                 # LoRA learning rate
    fp16=not torch.cuda.is_bf16_supported(),  # FP16 if no BF16
    bf16=torch.cuda.is_bf16_supported(),      # BF16 if available
    logging_steps=1,                    # Log every step
    optim="paged_adamw_32bit",          # Memory-efficient optimizer
    weight_decay=0.01,                  # L2 regularization
    lr_scheduler_type="linear",         # Linear warmup/decay
    save_strategy="no",                 # Don't save checkpoints (avoid pickle issues)
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",          # Column with formatted text
    max_seq_length=2048,                # Truncate longer sequences
    packing=False,                      # Don't pack multiple examples
    args=training_args,
)
```

---

### Step 7: Train

```python
trainer_stats = trainer.train()
```

**Expected Output:**
```
GPU = NVIDIA GB10. Max memory = 119.699 GB.
7.137 GB of memory reserved.

Starting training...
[100/100 Step]
```

**Training metrics shown:**
- Loss
- Learning rate
- GPU memory
- Training speed (samples/sec)

---

### Step 8: Training Statistics

After training completes:

```
TRAINING COMPLETE - Statistics
⏱️  Training time: 500 seconds (8.3 minutes)
💾 Peak memory used: 8.2 GB
📊 Memory usage: 6.8% of max
```

---

### Step 9: Test After Training

Run inference on fine-tuned model - should show improvement over baseline.

**Before training:** Generic response  
**After training:** Domain-specific Python code

---

### Step 10: Save Model

Two versions saved:

#### A. LoRA Adapters (50-100 MB)
```python
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```

Use when you want to keep base model separate:
```python
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("model_path")
model = PeftModel.from_pretrained(base_model, "lora_model")
```

#### B. Merged Model (Full 7B)
```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("model_merged")
tokenizer.save_pretrained("model_merged")
```

Use for standalone inference:
```python
model = AutoModelForCausalLM.from_pretrained("model_merged")
```

---

## Customization Options

### Change Model Size
```python
# Use 13B instead of 7B
model_id = "meta-llama/Llama-2-13b-hf"  # Larger, slower, higher quality
model_id = "meta-llama/Llama-2-7b-hf"   # Smaller, faster, current
```

### Change LoRA Rank
```python
lora_config = LoraConfig(
    r=8,   # Smaller = fewer params, faster, lower quality
    r=16,  # Current (good default)
    r=32,  # Larger = more params, slower, potentially better
    ...
)
```

### Change Training Length
```python
# Quick test (default - 100 steps)
max_steps = 100

# Full training (1 epoch)
num_train_epochs = 1
# Remove max_steps when using num_train_epochs

# Production (3 epochs)
num_train_epochs = 3
```

### Change Learning Rate
```python
learning_rate = 1e-4   # Conservative
learning_rate = 2e-4   # Current (good default)
learning_rate = 5e-4   # Aggressive
```

### Change Batch Size
```python
per_device_train_batch_size = 1  # Smaller, slower, less memory
per_device_train_batch_size = 2  # Current (balanced)
per_device_train_batch_size = 4  # Faster, needs more memory
```

### Enable Checkpointing
```python
# To save intermediate checkpoints (use with caution):
save_strategy = "steps",
save_steps = 50,
save_total_limit = 3,  # Keep only 3 most recent
```

---

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
per_device_train_batch_size = 1

# Or reduce sequence length
max_seq_length = 1024

# Or increase gradient accumulation
gradient_accumulation_steps = 8
```

### Model Loading Fails
```python
# Make sure model is downloaded
ls -lh llama-2-7b-hf/

# Or re-download
from huggingface_hub import snapshot_download
snapshot_download("meta-llama/Llama-2-7b-hf", local_dir="./llama-2-7b-hf")
```

### Dataset Not Found
```python
# Run Step 0 first to auto-download
# Or manually load:
python -c "from datasets import load_dataset; \
  dataset = load_dataset('iamtarun/python_code_instructions_18k_alpaca'); \
  dataset.save_to_disk('./python_code_instructions_18k_alpaca/hf_format')"
```

### Slow Training
```python
# Enable packing (faster but riskier)
packing = True

# Increase batch size
per_device_train_batch_size = 4

# Increase num_proc for data loading
dataset_num_proc = 4
```

---

## Performance Expectations

### System Requirements
- **GPU**: 8GB minimum, 16GB+ recommended
- **Disk**: 30GB free (model + dataset + outputs)
- **RAM**: 16GB+ system RAM

### Training Speed
```
Throughput: 0.4-0.5 samples/second
100 steps: ~3-4 minutes
1 epoch (9,300 steps): ~8-10 hours
3 epochs: ~24-30 hours
```

### Memory Usage
```
Base model (4-bit): 4-5 GB
Training overhead:  3-4 GB
─────────────────────────
Total:              8-9 GB
```

---

## Comparison: Standard vs Unsloth

| Aspect | Standard | Unsloth |
|--------|----------|---------|
| Speed | 1x baseline | 2-3x faster |
| Memory | 8-9 GB | 5-6 GB |
| Setup | Simple | Requires install |
| Code clarity | Very clear | Optimized (less clear) |
| Compatibility | Everywhere | CUDA/GPU required |

---

## Advanced: Custom Dataset Format

If you have your own data:

```python
def format_custom_dataset(examples):
    formatted = []
    for item in examples:
        prompt = f"Q: {item['question']}\nA: {item['answer']}"
        formatted.append({"text": prompt})
    return formatted

# Use your data
from datasets import Dataset
custom_data = Dataset.from_dict({...})
custom_data = custom_data.map(format_custom_dataset)
```

---

## Key Differences from Unsloth

| Component | Standard | Unsloth |
|-----------|----------|---------|
| Model loading | AutoModelForCausalLM + BitsAndBytes | FastLanguageModel |
| LoRA setup | PEFT LoraConfig + get_peft_model | FastLanguageModel.get_peft_model |
| Inference | model.eval() + generate | FastLanguageModel.for_inference |
| Optimizer | paged_adamw_32bit | adamw_8bit |
| Speed | Baseline | 2-3x faster |

Both are fully compatible - can mix and match!

---

**Version**: 1.0  
**Created**: 2026-08-08  
**Notebook**: finetune-v2-standard.ipynb
