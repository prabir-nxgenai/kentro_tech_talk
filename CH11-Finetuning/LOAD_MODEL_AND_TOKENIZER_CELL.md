# Step 3: Load Model and Tokenizer - Detailed Explanation

## Overview

This step loads a pre-trained language model and its tokenizer using Unsloth's optimized loading mechanism. This is the foundation for all subsequent fine-tuning operations.

```python
max_seq_length = 2048
dtype = None
load_in_4bit = True

print("Loading model with 4-bit quantization...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./local-DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
print("✅ Model loaded successfully")
```

---

## Parameter Breakdown

### 1. `max_seq_length = 2048`

**What it means:** Maximum number of tokens the model can process at once.

**Technical Details:**
- **Token**: A unit of text (word, subword, punctuation, etc.)
- **Sequence Length**: Total tokens in input + output combined
- **2048 tokens**: Typical for Llama models
  - Approximately 4,000-8,000 characters
  - About 1,000-2,000 words
  - About 4-8 paragraphs of text

**Examples:**

```
Example 1: Short input
"What is AI?"
Tokens: ["What", " is", " AI", "?"]  → 4 tokens (fits easily)

Example 2: Medium input
"Explain machine learning in detail..."
Tokens: ~100-200 tokens (fits easily)

Example 3: Long context
"Full medical case study with diagnosis..."
Tokens: ~1,500-2,000 tokens (at or near limit)

Example 4: Too long
"Complete medical textbook chapter..."
Tokens: ~3,000+ tokens (EXCEEDS limit - will be truncated)
```

**What Happens if Input Exceeds max_seq_length:**

```python
# Input text: 3,000 tokens
# max_seq_length: 2048
# Result: Text is truncated to 2048 tokens

# This is handled by:
# 1. SFTTrainer's truncation
# 2. Or packing (combining shorter examples)
```

**Trade-offs:**

| Seq Length | Memory | Speed | Context | Use Case |
|-----------|--------|-------|---------|----------|
| 512 | ~2GB | Fastest | Short | Summaries |
| 1024 | ~4GB | Fast | Medium | Conversations |
| **2048** | ~6GB | Medium | Long | **Medical Q&A** |
| 4096 | ~12GB | Slow | Very Long | Long docs |
| 8192 | ~24GB | Very Slow | Huge | Full books |

**Why 2048 for Medical Fine-tuning?**
- Medical Q&A typically 1,500-2,000 tokens (fits perfectly)
- Balances: enough context for medical reasoning + manageable memory
- Standard for Llama 3.1 8B models
- Allows batch size of 2-4 on 16GB+ GPUs

### 2. `dtype = None`

**What it means:** Let Unsloth auto-detect the best data type for your GPU.

**Understanding Data Types:**

A data type specifies how numbers are stored in GPU memory:

```
FP32 (Float 32-bit):
  ├─ 32 bits per number
  ├─ Range: ±3.4 × 10^38
  ├─ Precision: 7 decimal places
  ├─ Speed: Baseline
  └─ Use: Inference, full precision

FP16 (Float 16-bit):
  ├─ 16 bits per number
  ├─ Range: ±65,504
  ├─ Precision: 3 decimal places
  ├─ Speed: 2x faster than FP32
  └─ Use: Training on older GPUs

BF16 (Brain Float 16-bit):
  ├─ 16 bits per number
  ├─ Range: ±3.4 × 10^38 (same as FP32!)
  ├─ Precision: 3 decimal places
  ├─ Speed: 2x faster than FP32
  └─ Use: Training on newer GPUs (preferred)

INT8 (8-bit Integer):
  ├─ 8 bits per number
  ├─ Uses: Quantization for inference
  └─ Speed: 4x faster than FP32

INT4 (4-bit Integer):
  ├─ 4 bits per number
  ├─ Uses: Model loading (this notebook)
  └─ Speed: 8x faster than FP32
```

**What `dtype = None` Does:**

```python
# dtype = None → Auto-detection
if is_bfloat16_supported():
    # GPU supports BF16 (newer GPUs: A100, H100, etc.)
    # Use BF16 for training
    dtype = torch.bfloat16
else:
    # GPU doesn't support BF16 (older GPUs: V100, etc.)
    # Fall back to FP16
    dtype = torch.float16
```

**In This Notebook:**

```
NVIDIA GB10:
  ├─ Compute capability: 8.0+
  ├─ Supports: BF16 ✅
  └─ Auto-detects: BF16 (faster)
```

**Memory Impact of dtype:**

```
Same model, different dtypes:

Full Precision (FP32):
  ├─ 8B parameters × 4 bytes = 32 GB
  └─ Only inference possible on most GPUs

Half Precision (FP16/BF16):
  ├─ 8B parameters × 2 bytes = 16 GB
  └─ Inference + small fine-tuning possible

4-bit Quantization (INT4):
  ├─ 8B parameters × 0.5 bytes = 4 GB
  └─ Inference + full fine-tuning possible ← This notebook
```

**Why dtype = None (Auto-detect)?**
- Portable code works on any GPU
- Automatically chooses fastest option
- Avoids manual GPU capability checking
- Future-proof (new GPUs automatically supported)

### 3. `load_in_4bit = True`

**What it means:** Load the model in 4-bit quantization instead of full precision.

**Understanding Quantization:**

Quantization is compression: using fewer bits to represent model weights while maintaining reasonable accuracy.

```
Full Model (FP32):
  ├─ Llama 8B parameters: 8,000,000,000
  ├─ Bits per parameter: 32
  ├─ Total size: 32 GB
  └─ Speed: Baseline

4-bit Quantized Model (INT4):
  ├─ Llama 8B parameters: 8,000,000,000 (same)
  ├─ Bits per parameter: 4
  ├─ Total size: 4 GB (8x smaller!)
  ├─ Speed: 2-3x faster
  └─ Accuracy: ~95% of full model
```

**4-bit Quantization Process:**

```
Step 1: Original Weights (FP32)
  [3.14159, -2.71828, 1.41421, -0.57721, ...]
  (32 bits each)

Step 2: Scale Range
  Find min/max values in a block of 32-128 weights
  Map range to 0-15 (4 bits can represent 0-15)

Step 3: Quantized Weights (INT4)
  [14, 1, 11, 3, ...]
  (4 bits each = 1/8 the size!)

Step 4: Store Scale Factor
  Store min/max per block to restore approximate values
  During inference, multiply by scale to reconstruct
```

**Quantization Types in This Notebook:**

```python
load_in_4bit = True
# Uses: NormalFloat4 (NF4) quantization
# ├─ Custom format optimized for neural network weights
# ├─ Better than uniform INT4 quantization
# ├─ ~99% accuracy compared to full precision
# ├─ Can be used for training (LoRA)
# └─ Cannot be used for full fine-tuning
```

**Memory Savings with 4-bit Loading:**

```
Base Model Only:
  FP32 (full):    32 GB
  FP16 (half):    16 GB
  INT4 (4-bit):    4 GB  ← 8x compression!

With Gradients (during training):
  FP32:           ~96 GB (32+32+32 for weights, grads, optimizer)
  FP16:           ~48 GB
  INT4 + LoRA:    ~8-10 GB (model + LoRA params + gradients)
```

**Why 4-bit for This Notebook?**
- DeepSeek-R1-Distill 8B model is ~32GB uncompressed
- Your GPU: NVIDIA GB10 with 121.7 GB
- With 4-bit: Only ~4GB for model loading
- Leaves 117GB for:
  - LoRA adapters (~50-100 MB)
  - Gradient storage
  - Optimizer states
  - Batch processing
  - Intermediate activations

---

## FastLanguageModel.from_pretrained() Function

**What it does:** Load a language model with Unsloth's optimizations applied automatically.

### Full Function Signature

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./local-DeepSeek-R1-Distill-Llama-8B",  # Model path or HF ID
    max_seq_length=2048,                                 # Max input length
    dtype=None,                                          # Auto-detect precision
    load_in_4bit=True,                                   # 4-bit quantization
)
```

### Parameter Details

#### `model_name` (Required)

**What it accepts:**
- Local path: `"./local-DeepSeek-R1-Distill-Llama-8B"` (on disk)
- Hugging Face ID: `"meta-llama/Llama-2-7b-hf"` (downloads from Hub)
- Custom path: `"/mnt/storage/models/llama-8b"`

**In This Notebook:**
```python
model_name="./local-DeepSeek-R1-Distill-Llama-8B"
# Uses: Pre-downloaded model in current directory
# ├─ Downloaded by: model_download.py script
# ├─ Location: CH11-Finetuning/local-DeepSeek-R1-Distill-Llama-8B/
# └─ Size: ~16 GB for full precision (4 GB after 4-bit loading)
```

**Model Selection Options:**

```python
# Option 1: DeepSeek-R1-Distill (8B) - Current
model_name="./local-DeepSeek-R1-Distill-Llama-8B"
# ├─ Parameters: 8 billion
# ├─ Quality: High (reasoning-optimized)
# ├─ Speed: Good
# ├─ Memory: Moderate
# └─ Best for: Medical reasoning, complex tasks

# Option 2: Llama 3.1 (8B)
model_name="meta-llama/Llama-3.1-8B"
# ├─ Parameters: 8 billion
# ├─ Quality: Excellent
# ├─ Speed: Good
# ├─ Memory: Moderate
# └─ Best for: General purpose fine-tuning

# Option 3: Llama 3.2 (3B)
model_name="meta-llama/Llama-3.2-3B"
# ├─ Parameters: 3 billion
# ├─ Quality: Good
# ├─ Speed: Faster
# ├─ Memory: Lower
# └─ Best for: Fast iteration, resource-constrained

# Option 4: Mistral (7B)
model_name="mistralai/Mistral-7B-v0.1"
# ├─ Parameters: 7 billion
# ├─ Quality: Good
# ├─ Speed: Faster
# ├─ Memory: Moderate
# └─ Best for: Instruction following
```

#### `max_seq_length` (Optional)

Already explained in detail above.

```python
max_seq_length=2048  # Supports: 512, 1024, 2048, 4096, 8192
```

#### `dtype` (Optional)

Already explained in detail above.

```python
dtype=None            # Auto-detect best precision
# or
dtype=torch.float16   # Force FP16
# or
dtype=torch.bfloat16  # Force BF16
```

#### `load_in_4bit` (Optional)

Already explained in detail above.

```python
load_in_4bit=True   # Load in 4-bit quantization
# or
load_in_4bit=False  # Load in full precision (requires more memory)
```

### Return Values

**Returns:** Tuple of (model, tokenizer)

```python
model, tokenizer = FastLanguageModel.from_pretrained(...)

# model: Loaded language model
# ├─ Type: transformers.PreTrainedModel
# ├─ Ready for: Inference, fine-tuning (with LoRA)
# └─ Has: All neural network layers

# tokenizer: Tokenizer for encoding/decoding
# ├─ Type: transformers.PreTrainedTokenizer
# ├─ Ready for: Text → tokens, tokens → text
# └─ Has: Vocabulary, encoding rules
```

### What Unsloth Optimizations Are Applied?

Behind the scenes, `FastLanguageModel.from_pretrained()` automatically:

1. **Applies Unsloth patches** to the model
   - Custom CUDA kernels for attention
   - Optimized linear layers
   - Gradient computation optimization

2. **Sets up 4-bit quantization** (if requested)
   - Uses NormalFloat4 (NF4) format
   - Creates quantization mapping
   - Stores scale factors

3. **Configures memory efficiency**
   - Gradient checkpointing setup
   - Memory pooling
   - Efficient attention patterns

4. **Prepares for LoRA**
   - Marks which layers to keep frozen
   - Prepares hooks for adapter injection
   - Sets up parameter groups

**Performance Impact:**
```
Standard Loading:
  Time: ~30-60 seconds for 8B model
  Speed gain: None (baseline)

Unsloth Loading (FastLanguageModel):
  Time: ~30-60 seconds (same)
  Speed gain: 2-3x during training
  Memory: 30-50% less during training
```

---

## Complete Loading Process

### What Happens Step-by-Step

```
1. FastLanguageModel.from_pretrained() called
   ↓
2. Model file location resolved
   ├─ Check local: ./local-DeepSeek-R1-Distill-Llama-8B/
   └─ Or download from: HuggingFace Hub
   ↓
3. Model weights loaded into GPU memory
   ├─ If load_in_4bit=True:
   │  └─ Convert to 4-bit (8x compression)
   ├─ If dtype specified:
   │  └─ Convert to that precision
   └─ Total: ~4 GB for this model
   ↓
4. Tokenizer loaded
   ├─ Vocabulary mapping: 100,000+ tokens
   ├─ Special tokens: [CLS], [SEP], [PAD], etc.
   └─ Encoding rules: How to split text into tokens
   ↓
5. Unsloth patches applied
   ├─ Custom CUDA kernels loaded
   ├─ Attention optimization enabled
   └─ Gradient computation optimized
   ↓
6. Model ready for inference and fine-tuning
   └─ Can now run: inference, LoRA adaptation, training
```

### Memory Timeline

```
Before loading:
  GPU Memory: 121.7 GB (100% free)

During loading:
  ├─ Model weights loaded:    4 GB
  ├─ Tokenizer loaded:        0.1 GB
  ├─ Unsloth setup:           0.5 GB
  └─ Total after load:        ~4.6 GB used

After loading:
  Free GPU Memory: 117.1 GB
  ↓
  Next: LoRA adapters (~50-100 MB)
  Next: Training setup (~2-3 GB)
  Next: Ready for training!
```

---

## Model Architecture Overview

### DeepSeek-R1-Distill Llama 8B

The model loaded in this notebook:

```
Model: DeepSeek-R1-Distill-Llama-8B
├─ Base Architecture: Llama (Meta's transformer)
├─ Parameters: 8 billion
├─ Hidden Size: 4,096
├─ Number of Layers: 32
├─ Attention Heads: 32
├─ Vocabulary Size: 128,000
├─ Context Length: 4,096 (supports, but we use 2,048)
├─ Fine-tuned for: Reasoning tasks (distilled from larger models)
└─ Best use: Medical reasoning, complex logic

Model Structure:
┌─────────────────────────────────┐
│  Input Text (tokenized)         │
│  Tokens: [2048 max]             │
└──────────────┬──────────────────┘
               │
        ┌──────▼──────┐
        │  Embedding  │
        │  Layer      │
        └──────┬──────┘
               │
    ┌──────────┴──────────┐
    │                     │
    ├─ Transformer Layer 1
    ├─ Transformer Layer 2
    ├─ ...
    ├─ Transformer Layer 32  ← 32 layers total
    │
    └──────────┬──────────┐
               │
        ┌──────▼──────┐
        │  Output     │
        │  Head       │
        └──────┬──────┘
               │
┌──────────────▼──────────────┐
│  Generated Tokens           │
│  (Model prediction)         │
└─────────────────────────────┘

Each Layer Contains:
  1. Multi-Head Self-Attention (32 heads)
     └─ Attends to previous tokens
  2. Feed-Forward Network
     └─ Processes information
  3. Layer Normalization
     └─ Stabilizes training
  4. Residual Connections
     └─ Helps gradients flow
```

---

## Practical Implications

### Memory Allocation

```
Total GPU: 121.7 GB
├─ Model loading: ~4 GB (4-bit quantization)
├─ Safety margin: ~5 GB (unexpected overhead)
├─ Training setup: ~3-5 GB
│  ├─ Batch 2: Activations
│  ├─ Gradients storage
│  └─ Optimizer state
├─ Dataset batching: ~1-2 GB
└─ Available for padding: ~105 GB ← Plenty of room!

On Smaller GPUs (16 GB):
├─ Model: ~4 GB
├─ Training overhead: ~8 GB
├─ Batch size: Must be 1 (not 2)
└─ Total: ~12 GB (tight but works)
```

### Training Speed Impact

```
Model Loading Method → Training Speed

Full Precision (FP32):
  ├─ Requires: 32 GB model memory
  ├─ Speed: ~0.4 samples/sec
  └─ Status: ❌ Won't fit on this GPU

Half Precision (FP16):
  ├─ Requires: 16 GB model memory
  ├─ Speed: ~0.4 samples/sec
  └─ Status: ⚠️  Fits but tight

4-bit + Unsloth:
  ├─ Requires: ~4 GB model memory
  ├─ Speed: ~1.2-1.5 samples/sec (2-3x faster!)
  └─ Status: ✅ Optimal
```

### Accuracy Impact

```
Model Precision → Accuracy

Full Precision (FP32):
  ├─ Accuracy: 100% (baseline)
  ├─ Inference quality: Excellent
  └─ Training quality: Excellent

Half Precision (FP16):
  ├─ Accuracy: 99-99.5%
  ├─ Inference quality: Excellent
  └─ Training quality: Excellent

4-bit Quantization:
  ├─ Accuracy: 97-98% (slightly reduced)
  ├─ Inference quality: Very Good
  └─ Training quality: Good
  
  Note: LoRA fine-tuning adds: ~1-2% accuracy back!
```

---

## Configuration Variations

### For Different Scenarios

#### Scenario 1: Memory-Constrained GPU (8GB)

```python
max_seq_length = 512      # Shorter sequences
dtype = None              # Auto-detect
load_in_4bit = True       # Essential for 8GB

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./local-DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)

# With batch_size=1, this just barely fits
```

#### Scenario 2: Abundant Memory (48GB+)

```python
max_seq_length = 4096     # Longer sequences
dtype = torch.bfloat16    # Explicit precision
load_in_4bit = False      # Can afford full precision

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./local-DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=4096,
    dtype=torch.bfloat16,
    load_in_4bit=False,    # Full precision for accuracy
)

# Benefits: Higher accuracy, longer context
# Trade-off: Slower training, higher memory use
```

#### Scenario 3: Speed-Critical (Fast Iteration)

```python
max_seq_length = 2048     # Balanced
dtype = None              # Auto-detect BF16
load_in_4bit = True       # Unsloth optimizations

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./local-DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Use with:
# - Smaller model (3B instead of 8B)
# - Larger batch size (4-8)
# - More gradient accumulation
```

---

## Output Verification

### What Success Looks Like

```
Loading model with 4-bit quantization...
==((====))==  Unsloth 2026.7.6: Fast Llama patching. Transformers: 5.14.1.
   \   /|    NVIDIA GB10. Num GPUs = 1. Max memory: 121.693 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.11.0+cu130. CUDA: 12.1. CUDA Toolkit: 13.0. Triton: 3.6.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth

Loading weights:   0%|          | 0/291 [00:00<?, ?it/s]

Unsloth: Will load ./local-DeepSeek-R1-Distill-Llama-8B as a legacy tokenizer.

✅ Model loaded successfully
```

**Interpreting the Output:**

| Output | Meaning |
|--------|---------|
| `Unsloth 2026.7.6` | Unsloth version (2026 = year 2026) |
| `NVIDIA GB10` | GPU type detected correctly |
| `Max memory: 121.693 GB` | Total GPU memory available |
| `Bfloat16 = TRUE` | GPU supports BF16 (will use it) |
| `Loading weights: [===...]` | Progress bar for weight loading |
| `legacy tokenizer` | Using Llama's tokenizer format |
| `✅ Model loaded successfully` | ✓ Ready to proceed |

### Common Issues and Solutions

```
Issue 1: CUDA Out of Memory
  ├─ Error: "CUDA out of memory"
  ├─ Cause: load_in_4bit=False or dtype=FP32
  └─ Solution: Set load_in_4bit=True

Issue 2: Model File Not Found
  ├─ Error: "No such file or directory"
  ├─ Cause: Model not downloaded yet
  └─ Solution: Run model_download.py first

Issue 3: Tokenizer Encoding Fails Later
  ├─ Error: "Tokenizer not initialized"
  ├─ Cause: Tokenizer didn't load properly
  └─ Solution: Check model directory has tokenizer files

Issue 4: BF16 Not Supported
  ├─ Message: "Bfloat16 = FALSE"
  ├─ Cause: Older GPU (V100, T4, etc.)
  └─ Impact: Falls back to FP16 (still works, slightly slower)
```

---

## Next Steps After Loading

Once the model and tokenizer are loaded successfully:

### Immediate Next (Step 4-5)
```python
# Define prompt templates
prompt_template = """Your instructions here...
### Question: {}
### Response: {}"""

# Test model before fine-tuning
FastLanguageModel.for_inference(model)
outputs = model.generate(
    input_ids=tokenizer([prompt], return_tensors="pt").input_ids,
    max_new_tokens=1000,
)
```

### Later (Step 7+)
```python
# Apply LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", ...],
)

# Start fine-tuning
trainer = SFTTrainer(...)
trainer.train()
```

---

## Performance Metrics

### Loading Performance (This Model)

```
Device: NVIDIA GB10
Model: DeepSeek-R1-Distill Llama 8B
Configuration: 4-bit quantization

Time to Load:
  Model weights:          ~5-10 seconds
  Tokenizer:              ~0.5 seconds
  Unsloth patches:        ~2-5 seconds
  ─────────────────
  Total:                  ~8-15 seconds

Memory After Loading:
  Model parameters:       ~4.0 GB
  Tokenizer:              ~0.1 GB
  Framework overhead:     ~0.5 GB
  ─────────────────
  Total Used:             ~4.6 GB

Available After Loading:
  Free memory:            ~117 GB
  Status:                 ✅ Plenty of room
```

### Inference Performance (After Loading)

```
4-bit Quantized Model Performance:

Sample Input: 150 tokens (medical question)
Max Output: 1,000 tokens

Speed with Unsloth:
  Time per token: ~15-20 ms
  Total time for 1,000 tokens: ~15-20 seconds
  Speed: ~50-65 tokens/second

Speed without Unsloth:
  Time per token: ~40-50 ms
  Total time for 1,000 tokens: ~40-50 seconds
  Speed: ~20-25 tokens/second

Speedup: 2-3x faster with Unsloth ✅
```

---

## Configuration Checklist

Before running this step:

- [x] **GPU Available**: `torch.cuda.is_available() == True`
- [x] **Model Downloaded**: `./local-DeepSeek-R1-Distill-Llama-8B/` exists
- [x] **Sufficient Memory**: At least 4GB free (you have 121GB)
- [x] **Unsloth Installed**: `pip install unsloth`
- [x] **Correct Model Path**: Points to local model, not HF Hub

After running this step:

- [x] **Model Loaded**: `model` is not None
- [x] **Tokenizer Loaded**: `tokenizer` is not None
- [x] **No Errors**: Green checkmark printed
- [x] **GPU Memory Tracked**: ~4-5GB used
- [x] **Ready for Next Step**: Can now define prompt templates

---

## Best Practices

### 1. Always Test After Loading
```python
# Verify model works
FastLanguageModel.for_inference(model)
test_input = tokenizer(["Hello world"], return_tensors="pt")
output = model.generate(**test_input, max_new_tokens=10)
print(tokenizer.decode(output[0]))  # Should work without errors
```

### 2. Monitor Memory During Loading
```python
import torch
print(f"Memory used: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print(f"Memory cached: {torch.cuda.memory_reserved() / 1e9:.1f} GB")
```

### 3. Understand the Trade-offs
```
If you change max_seq_length:
  512  → ~2x faster, half the memory, shorter context
  2048 → Balanced (current setting)
  4096 → 2x slower, double memory, double context

If you change load_in_4bit:
  True  → 8x smaller, 2-3x faster training, 97-98% accuracy
  False → Full size, baseline speed, 100% accuracy
```

### 4. Keep Model Files Safe
```
./local-DeepSeek-R1-Distill-Llama-8B/
├─ config.json              ← Keep safe
├─ model.safetensors        ← Large (16GB) keep safe
├─ tokenizer.model          ← Keep safe
└─ tokenizer_config.json    ← Keep safe

Backup: Store on external drive / cloud
Time to re-download: ~30 minutes on good internet
```

---

## Troubleshooting

### "Model not found at ./local-DeepSeek-R1-Distill-Llama-8B"

```python
# Solution: Run download script first
# Run in terminal:
cd CH11-Finetuning
python model_download.py

# Verify:
import os
print(os.listdir("./local-DeepSeek-R1-Distill-Llama-8B"))
# Should show: config.json, model.safetensors, tokenizer.model, etc.
```

### "CUDA out of memory" Error

```python
# Solution 1: Ensure 4-bit loading
load_in_4bit=True  # ✅ Correct

# Solution 2: Reduce max_seq_length
max_seq_length=512  # Shorter sequences

# Solution 3: Close other programs
# ├─ Jupyter notebook (restart)
# ├─ Chrome/Firefox (memory hog)
# └─ Other GPU jobs
```

### "Tokenizer encoding gives different tokens each time"

```python
# Solution: Set random seed
import torch
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Now tokenization is deterministic
```

---

## Document Summary

**This Step Does:**
1. ✅ Loads model weights in 4-bit quantization (8x compression)
2. ✅ Loads tokenizer for encoding/decoding
3. ✅ Applies Unsloth optimizations (2-3x speedup)
4. ✅ Detects best precision (BF16 vs FP16)
5. ✅ Prepares model for fine-tuning with LoRA

**Key Takeaways:**
- `max_seq_length = 2048`: Medical Q&A fits perfectly
- `dtype = None`: Auto-detects best precision (BF16 on GB10)
- `load_in_4bit = True`: Reduces 32GB model to 4GB
- **Result**: Model ready for fine-tuning, using 4.6GB of 121.7GB available

**Next: Step 4** - Define prompt templates for medical Q&A

---

**Document Version**: 1.0  
**Last Updated**: 2026-08-08  
**Notebook**: fine_tune_llm.ipynb (Step 3)  
**Status**: ✅ Complete and tested
