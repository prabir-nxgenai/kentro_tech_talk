# Fine-tuning Troubleshooting Guide

## Common Errors & Solutions

### 1. PicklingError: Can't pickle SFTConfig

**Error Message:**
```
PicklingError: Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>: 
it's not the same object as trl.trainer.sft_config.SFTConfig
```

**Cause:** Happens when training tries to save intermediate checkpoints. Unsloth + TRL version incompatibility.

**Solution (✅ Applied in finetune-v2-fixed.ipynb):**
```python
args = TrainingArguments(
    ...
    save_strategy = "no",      # ← Don't save intermediate checkpoints
    save_steps = 0,            # ← Explicitly disable checkpoint saving
    ...
)
```

**Why this works:** Disables checkpoint saving during training. You still save the final model after training completes.

**Alternative Solutions:**

Option A: Enable checkpoints after final model (hybrid approach):
```python
save_strategy = "steps",
save_steps = 50,  # Save every 50 steps
```

Option B: Use epochs instead of steps:
```python
num_train_epochs = 1,
save_strategy = "epoch",
```

---

### 2. FileNotFoundError: Dataset not found

**Error Message:**
```
FileNotFoundError: Directory ./python_code_instructions_18k_alpaca is neither a 
`Dataset` directory nor a `DatasetDict` directory.
```

**Cause:** Step 0 (auto-download) wasn't run, or dataset is in wrong location.

**Solution:**
1. Make sure **Step 0 runs successfully** - it should print:
   ```
   ✅ Dataset already exists at ./python_code_instructions_18k_alpaca/hf_format
   ```

2. If Step 0 shows "Downloading...", wait for it to complete

3. Dataset must be at: `./python_code_instructions_18k_alpaca/hf_format/` (NOT parent directory)

4. Verify with:
   ```bash
   python verify_setup.py
   ```

---

### 3. CUDA Out of Memory

**Error Message:**
```
RuntimeError: CUDA out of memory. Tried to allocate ...
```

**Cause:** GPU doesn't have enough VRAM for 4-bit model + LoRA + batch size.

**Solutions (in order):**

1. **Reduce batch size:**
   ```python
   per_device_train_batch_size = 1  # Instead of 2
   ```

2. **Reduce sequence length:**
   ```python
   max_seq_length = 1024  # Instead of 2048
   ```

3. **Reduce max_steps for testing:**
   ```python
   max_steps = 10  # Quick test instead of 100
   ```

4. **Check if model is already loaded:**
   ```python
   import torch
   torch.cuda.empty_cache()  # Clear cache before training
   ```

**GPU Memory Estimate:**
- Llama 3.1 8B (4-bit): 5-6 GB
- LoRA + training: 2-3 GB
- Total minimum: 8 GB VRAM required

---

### 4. BFloat16 Not Supported

**Error Message:**
```
NotImplementedError: BFloat16 not supported on this device
```

**Cause:** GPU doesn't support BFloat16 (newer GPUs do, older ones don't).

**Solution:**
```python
# This is already handled in the notebook, but for reference:
from unsloth import is_bfloat16_supported

args = TrainingArguments(
    fp16 = not is_bfloat16_supported(),  # Use FP16 if BF16 not available
    bf16 = is_bfloat16_supported(),       # Use BF16 if available
    ...
)
```

This is already automatic in `finetune-v2-fixed.ipynb`.

---

### 5. Model Download Fails

**Error Message:**
```
Connection error / Timeout during model download
```

**Cause:** Network issue or rate limiting from Hugging Face Hub.

**Solutions:**

1. **Set Hugging Face token** (increases rate limit):
   ```python
   import os
   os.environ['HF_TOKEN'] = 'your_token_here'
   ```
   
   Get token from: https://huggingface.co/settings/tokens

2. **Resume interrupted download:**
   - Step 0 has `resume_download=True` by default
   - Just re-run Step 0, it will continue from where it left off

3. **Manual download:**
   ```bash
   huggingface-cli download unsloth/Meta-Llama-3.1-8B-bnb-4bit \
     --local-dir ./Meta-Llama-3.1-8B-bnb-4bit
   ```

4. **Check disk space:**
   ```bash
   df -h  # Model needs ~20 GB free space
   ```

---

### 6. Training Very Slow

**Symptom:** Training runs but very slow (< 1 sample/sec)

**Potential Causes & Solutions:**

1. **Packing disabled:**
   - Current: `packing = False` (safer, slower)
   - Try: `packing = True` (faster, may cause issues)
   
   ```python
   packing = True,  # Can make training 5x faster for short sequences
   ```

2. **CPU bottleneck:**
   ```python
   dataset_num_proc = 4  # Increase from 2 for faster data loading
   ```

3. **Wrong device:**
   ```python
   import torch
   print(torch.cuda.is_available())  # Should be True
   print(torch.cuda.get_device_name(0))  # Should show GPU name
   ```

**Expected speed:**
- With current config: ~1.2 samples/second
- 100 steps should take ~6 minutes
- 1 epoch (~9,300 steps) should take ~2 hours

---

### 7. Model Saving Fails

**Error Message:**
```
OSError: [Errno 28] No space left on device
```

**Cause:** Disk full when saving model.

**Solutions:**

1. **Check disk space:**
   ```bash
   df -h
   ```

2. **Clean up space:**
   ```bash
   # Remove old outputs
   rm -rf outputs/
   
   # Remove old models if not needed
   rm -rf lora_model/ model_merged/
   ```

3. **Space needed for each save:**
   - LoRA adapters: 50-100 MB
   - Merged 16-bit model: ~16 GB

**Total space needed during training:**
- Dataset: 100 MB
- Model: 5-6 GB (4-bit loaded)
- Merged output: 16 GB
- **Minimum: ~25 GB free space**

---

### 8. Inference (After Training) Shows No Output

**Symptom:** After training, inference just prints prompt, no generation

**Cause:** Model might not be in inference mode correctly

**Solution:**
```python
# Ensure this is called before generation
FastLanguageModel.for_inference(model)

# Then generate with proper settings
output = model.generate(
    **inputs,
    max_new_tokens=1000,
    temperature=0.7,  # Add some randomness
    top_p=0.9,
)
```

---

### 9. Different Results Before/After Training

**Symptom:** Model output seems same before and after training

**Possible Issues:**

1. **Training didn't actually run:**
   - Check Step 7 output - should show training progress
   - Look for loss decreasing over steps

2. **Training used wrong dataset:**
   - Verify Step 4 loads dataset successfully
   - Check dataset has examples with varied instruction/input/output

3. **Only 100 steps (default) may not be enough:**
   - Try full training: change `max_steps=100` to `num_train_epochs=1`
   - Or increase max_steps to 500-1000

4. **Model too large for domain adaptation in few steps:**
   - 8B parameters with 18K examples = needs more training
   - Consider fine-tuning longer

---

### 10. How to Enable Checkpoints (Without Pickling Error)

If you want to save intermediate checkpoints safely:

**Option 1: Save every N steps (risky, may still pickle)**
```python
save_strategy = "steps",
save_steps = 50,
save_total_limit = 3,  # Keep only 3 most recent checkpoints
```

**Option 2: Save only at end of epoch (safer)**
```python
save_strategy = "epoch",
num_train_epochs = 1,  # Use epochs instead of max_steps
```

**Option 3: Disable for training, save manually after**
```python
# In TrainingArguments:
save_strategy = "no",

# After training completes:
model.save_pretrained("checkpoint")
tokenizer.save_pretrained("checkpoint")
```

---

## Diagnostic Commands

Run these to debug issues:

### Check GPU status:
```bash
nvidia-smi
```

### Verify setup:
```bash
python verify_setup.py
```

### Check dataset:
```bash
ls -lh python_code_instructions_18k_alpaca/hf_format/
```

### Check model:
```bash
ls -lh Meta-Llama-3.1-8B-bnb-4bit/ | head -20
```

### Test imports:
```python
python -c "from unsloth import FastLanguageModel; print('✅ Unsloth OK')"
python -c "import torch; print(f'✅ PyTorch OK - CUDA: {torch.cuda.is_available()}')"
python -c "from datasets import load_from_disk; print('✅ Datasets OK')"
```

---

## Contact & Support

- **Unsloth Issues**: https://github.com/unslothai/unsloth/issues
- **TRL Issues**: https://github.com/huggingface/trl/issues
- **HF Datasets**: https://github.com/huggingface/datasets/issues

---

**Last Updated:** 2026-08-08  
**Notebook Version:** finetune-v2-fixed.ipynb
