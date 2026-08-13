#!/usr/bin/env python3
# test_llm.py
"""
This script loads a *fine-tuned* medical reasoning model from disk and runs
a single inference test to verify that:

  1) The model files load correctly
  2) The tokenizer matches the model
  3) CUDA inference works
  4) The model produces a medically-reasoned answer in the expected format

Author: Prabir Guha

The model being tested here is:
  DeepSeek-R1-Medical-FT-8b-16bts

This is the merged 16-bit version of a model that was fine-tuned using:
  - Unsloth
  - LoRA (PEFT)
  - A medical reasoning SFT dataset

Typical install:
  pip install unsloth trl transformers datasets torch

Run:
  python test_llm.py

Expected outcome:
  - The script prints a detailed, chain-of-thought style answer
  - If CUDA or model paths are wrong, it will fail early
"""

# =============================================================================
# Imports
# =============================================================================

# Unsloth provides fast loading, inference optimizations, and PEFT integration.
from unsloth import FastLanguageModel

# torch is used to move tensors to GPU and check CUDA availability.
import torch

# os is not strictly required here, but is commonly imported for filesystem checks.
import os


# =============================================================================
# 1) Runtime configuration
# =============================================================================

# Maximum sequence length (context window) for the model.
# Unsloth supports RoPE scaling internally, so longer contexts can work if needed.
max_seq_length = 2048

# dtype controls the numerical precision:
#   None        -> let Unsloth auto-detect the best choice for your GPU
#   torch.float16 -> good for older GPUs (e.g., T4, V100)
#   torch.bfloat16 -> preferred on Ampere+ GPUs (more stable)
dtype = None

# load_in_4bit controls whether the model is loaded in 4-bit quantized form.
# For testing a *merged 16-bit* fine-tuned model, we usually set this to False
# so we evaluate the full-precision merged weights.
#
# If you are memory-constrained, you *could* set this to True, but the results
# might differ slightly due to quantization.
load_in_4bit = True


# =============================================================================
# 2) Load the fine-tuned model and tokenizer
# =============================================================================

# This path must point to the directory created by:
#   model.save_pretrained(...)
#   tokenizer.save_pretrained(...)
# or:
#   model.save_pretrained_merged(...)
#
# In this case, we are loading the merged 16-bit fine-tuned model.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./DeepSeek-R1-Medical-FT-8b-16bts",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

print(model.type)  # should print the model class (e.g., LlamaForCausalLM)


# =============================================================================
# 3) Prompt template (same style used during training)
# =============================================================================

# It is important that the *inference prompt format* matches the format
# the model saw during training. Otherwise, performance will degrade.
#
# Here we:
#   - Describe the task
#   - Set the role ("medical expert")
#   - Insert the question
#   - Open a <think> tag so the model can produce reasoning
#
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


# =============================================================================
# 4) Define a test medical question
# =============================================================================

# This is a classic clinical vignette suggestive of meralgia paresthetica
# (lateral femoral cutaneous nerve entrapment), commonly used to test
# reasoning over symptoms and triggers.
question = """A 69-year-old man is experiencing burning pain, tingling, numbness, itching, and pins-and-needles sensations over the outer right thigh 
after 15-20 minutes of standing. The symptoms seem to go away after sitting down. The man has diabetes, but controlled. 
What could be the possible cause(s) of his symptoms?"""


# =============================================================================
# 5) Prepare the model for fast inference
# =============================================================================

# This switches the model into an optimized inference mode:
#   - Disables training-only features
#   - Enables faster kernels and caching
#   - Reduces overhead during generation
FastLanguageModel.for_inference(model)


# =============================================================================
# 6) Tokenize the prompt and move tensors to GPU
# =============================================================================

# Format the prompt by inserting the question.
# The second {} is left empty because we want the model to *generate* the reasoning.
prompt = prompt_style.format(question, "")

# Tokenize the prompt:
#   return_tensors="pt" -> return PyTorch tensors
#   .to("cuda")         -> move tensors to GPU for fast generation
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")


# =============================================================================
# 7) Generate the model's response
# =============================================================================

outputs = model.generate(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=1200,  # allow long, detailed reasoning + answer
    use_cache=True,       # speeds up autoregressive decoding
)

# Decode token IDs back into human-readable text.
response = tokenizer.batch_decode(outputs, skip_special_tokens=False)
# Fix tokenizer artifacts: Ġ represents space, Ċ represents newline
response_text = response[0].replace("Ġ", " ").replace("Ċ", "\n")


# =============================================================================
# 8) Print only the model's answer section
# =============================================================================

# The model output contains the entire prompt + completion.
# We split on "### Response:" to show only what the model generated.
print("\n================ MODEL OUTPUT ================\n")
# Safely extract response after "### Response:" marker
if "### Response:" in response_text:
    print(response_text.split("### Response:")[1].strip())
else:
    print(response_text.strip())


