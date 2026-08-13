#!/usr/bin/env python3
# fine_tune_llm.py
"""
Fine-tunes a local DeepSeek-R1 Distill Llama 8B model using Unsloth + LoRA (PEFT)
on a medical reasoning SFT dataset stored locally in Hugging Face format.

Author: Prabir Guha

Typical install:
  pip install unsloth trl transformers datasets torch

Assumptions:
  - You have a CUDA-capable GPU visible to PyTorch.
  - The model exists locally at: ./local-DeepSeek-R1-Distill-Llama-8B
  - The dataset exists locally at: ./medical-o1-reasoning-SFT/hf_format
"""

# =============================================================================
# Configuration: Select Training Mode
# =============================================================================
# Change MODE to "production" for full training (~2-4 hours)
MODE = "test"  # "test" or "production"
#MODE =  "production"

print("\n" + "="*70)
print(f"  FINE-TUNING MODE: {MODE.upper()}")
print("="*70)
if MODE == "test":
    print("  Quick Test Run: 10 steps, ~4-5 minutes")
    print("  Use for: Pipeline validation and debugging")
else:
    print("  Production Training: 3 epochs, ~2-4 hours")
    print("  Use for: Full model fine-tuning for deployment")
print("="*70 + "\n")

# =============================================================================
# Imports
# =============================================================================

from unsloth import FastLanguageModel
import torch
from datasets import load_from_disk
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
import time

print("All imports successful")


# =============================================================================
# Step 1: Check CUDA Availability
# =============================================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available to PyTorch. "
        "Fix your PyTorch + CUDA install before running fine-tuning."
    )

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"   Max memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


# =============================================================================
# Step 2: Load Model and Tokenizer
# =============================================================================

max_seq_length = 2048
dtype = None
load_in_4bit = True

print("\nLoading model with 4-bit quantization...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./local-DeepSeek-R1-Distill-Llama-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
print("Model loaded successfully (Unsloth 2026.8.15 auto-optimizations enabled)")


# =============================================================================
# Step 3: Define Prompt Templates
# =============================================================================

prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>{}"""

train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""

print("Prompt templates defined")


# =============================================================================
# Step 4: Pre-Training Inference Check
# =============================================================================

question = (
    "A 69-year-old man is experiencing burning pain, tingling, numbness, itching, "
    "and pins-and-needles sensations over the outer right thigh after 15-20 minutes "
    "of standing. The symptoms go away after sitting down. The man has diabetes, "
    "but controlled. What could be the possible cause(s) of his symptoms?"
)

FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)

response = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
response_text = response[0]

# Fix BPE space token artifacts FIRST (Ġ → space, Ċ → newline)
response_text = response_text.replace("Ġ", " ").replace("Ċ", "\n").replace("  ", " ")

print("\n" + "="*70)
print("PRE-TRAINING INFERENCE")
print("="*70 + "\n")

# Extract from ### Response: onward
if "### Response:" in response_text:
    response_only = response_text.split("### Response:")[1].strip()
    print(response_only)
elif "<think>" in response_text:
    # Fallback: extract from <think> onward
    response_only = response_text.split("<think>")[1].strip()
    print("<think>" + response_only)
else:
    print(response_text.strip())


# =============================================================================
# Step 5: Load and Format Dataset
# =============================================================================

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    inputs = examples["Question"]
    cots = examples["Complex_CoT"]
    outputs = examples["Response"]

    texts = []
    for q, cot, ans in zip(inputs, cots, outputs):
        text = train_prompt_style.format(q, cot, ans) + EOS_TOKEN
        texts.append(text)

    return {"text": texts}

print("Loading dataset from disk...")
dataset_path = "./medical-o1-reasoning-SFT/hf_format"
dataset_on_disk = load_from_disk(dataset_path, "en")

if hasattr(dataset_on_disk, "keys"):
    print(f"Detected DatasetDict splits: {list(dataset_on_disk.keys())}")
    base_train = dataset_on_disk["train"]
else:
    base_train = dataset_on_disk

base_train = base_train.shuffle(seed=42)
N = min(16000, len(base_train))
train_dataset = base_train.select(range(N))

print(f"Rows in raw training subset: {len(train_dataset)}")

# Display sample BEFORE formatting
print("\n" + "="*70)
print("BEFORE FORMATTING (Raw Data from Disk)")
print("="*70)
sample_question = train_dataset["Question"][0]
sample_cot = train_dataset["Complex_CoT"][0]
sample_response = train_dataset["Response"][0]

print(f"\nQuestion:\n{sample_question[:400]}{'...' if len(sample_question) > 400 else ''}")
print(f"\nComplex Chain of Thought (first 400 chars):\n{sample_cot[:400]}{'...' if len(sample_cot) > 400 else ''}")
print(f"\nResponse (first 400 chars):\n{sample_response[:400]}{'...' if len(sample_response) > 400 else ''}")

dataset = train_dataset.map(formatting_prompts_func, batched=True)
print(f"\nRows after formatting: {len(dataset)}")

# Display same sample AFTER formatting
print("\n" + "="*70)
print("AFTER FORMATTING (Complete Training Example)")
print("="*70)
print(f"\nFormatted text (full structure with prompt template and EOS token):\n")
print(dataset["text"][0])


# =============================================================================
# Step 6: Apply LoRA Adapters
# =============================================================================

print("Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("LoRA adapters applied")


# =============================================================================
# Step 7: Configure Trainer
# =============================================================================

print(f"Configuring trainer for {MODE} mode...")

# Build trainer kwargs
trainer_kwargs = {
    "model": model,
    "tokenizer": tokenizer,
    "train_dataset": dataset,
    "dataset_text_field": "text",
    "max_seq_length": max_seq_length,
    "packing": True,  # Enable packing for Unsloth's padding_free optimization
}

if MODE == "test":
    trainer = SFTTrainer(
        **trainer_kwargs,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=10,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            save_strategy="no",
            packing=True,
            packing_strategy="bfd",
            use_cpu=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to=[],
        ),
    )
    print("Loaded TEST configuration (10 steps, ~4-5 minutes)\n")
else:
    trainer = SFTTrainer(
        **trainer_kwargs,
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            num_train_epochs=3,
            warmup_ratio=0.03,
            learning_rate=1e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=25,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir="outputs",
            save_strategy="no",
            packing=True,
            packing_strategy="bfd",
            use_cpu=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to=[],
        ),
    )
    print("Loaded PRODUCTION configuration (3 epochs, ~2-4 hours)\n")

# =============================================================================
# Step 8: Train the Model
# =============================================================================

print("\n" + "="*70)
print("  TRAINING IN PROGRESS...")
print("="*70)
print(f"  Library versions: Unsloth 2026.8.15, TRL 0.24.0, Torch 2.4.0")
print(f"  Optimizations: Unsloth auto-optimizations, padding_free packing\n")

start_time = time.time()
trainer_stats = trainer.train()
elapsed = time.time() - start_time

print(f"\n  Total training time: {elapsed/3600:.1f} hours ({elapsed/60:.1f} minutes)")

print("\n" + "="*70)
print("  TRAINING COMPLETE")
print("="*70 + "\n")


# =============================================================================
# Step 9: Post-Training Inference Check
# =============================================================================

FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,
    use_cache=True,
)

response = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
response_text = response[0]

# Fix BPE space token artifacts FIRST (Ġ → space, Ċ → newline)
response_text = response_text.replace("Ġ", " ").replace("Ċ", "\n").replace("  ", " ")

print("\n" + "="*70)
print("POST-TRAINING INFERENCE")
print("="*70 + "\n")

# Extract from ### Response: onward
if "### Response:" in response_text:
    response_only = response_text.split("### Response:")[1].strip()
    print(response_only)
elif "<think>" in response_text:
    # Fallback: extract from <think> onward
    response_only = response_text.split("<think>")[1].strip()
    print("<think>" + response_only)
else:
    print(response_text.strip())


# =============================================================================
# Step 10: Save Fine-Tuned Model
# =============================================================================

new_model_local = "DeepSeek-R1-Medical-FT-8b-16bts"

print(f"Saving LoRA adapters to: {new_model_local}")
model.save_pretrained(new_model_local)
tokenizer.save_pretrained(new_model_local)

print(f"Saving merged model to: {new_model_local}")
model.save_pretrained_merged(
    new_model_local,
    tokenizer,
    save_method="merged_16bit",
)

print(f"\nFine-tuned model saved to: {new_model_local}")

