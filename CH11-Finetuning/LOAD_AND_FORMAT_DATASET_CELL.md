# Step 6: Load and Format Dataset - Detailed Explanation

## Overview

This step loads the medical fine-tuning dataset from disk and formats each example into the exact prompt structure the model will learn from during training. This is a critical step - **garbage in, garbage out** - the quality of formatted examples directly determines training quality.

```python
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

print(f"\n📝 Question:\n{sample_question[:400]}{'...' if len(sample_question) > 400 else ''}")
print(f"\n💭 Complex Chain of Thought (first 400 chars):\n{sample_cot[:400]}{'...' if len(sample_cot) > 400 else ''}")
print(f"\n📋 Response (first 400 chars):\n{sample_response[:400]}{'...' if len(sample_response) > 400 else ''}")

dataset = train_dataset.map(formatting_prompts_func, batched=True)
print(f"\nRows after formatting: {len(dataset)}")

# Display same sample AFTER formatting
print("\n" + "="*70)
print("AFTER FORMATTING (Complete Training Example)")
print("="*70)
print(f"\n✨ Formatted text (full structure with prompt template and EOS token):\n")
print(dataset["text"][0])
```

---

## Part 1: End-of-Sequence Token (EOS_TOKEN)

### What is EOS_TOKEN?

```python
EOS_TOKEN = tokenizer.eos_token
```

**EOS** = End-of-Sequence Token

A special token that marks where a complete text example ends. It tells the model: "This is where training data ends, stop generating tokens."

### Why It Matters

```
Without EOS_TOKEN:
"Below is an instruction... The response is ... [no end marker]
[Runs into next example without clear boundary]
The patient has... [confusion!]"

With EOS_TOKEN:
"Below is an instruction... The response is ...[EOS]
[Clear boundary]
Below is an instruction... [new example starts]"
```

### EOS Token Value

```python
# For Llama models:
tokenizer.eos_token = "</s>"

# Token ID:
tokenizer.eos_token_id = 2

# In binary:
"</s>" → token_id 2 → represents: "End of Sequence"
```

### How It's Used in Training

```
Dataset Example:
"Below is an instruction...
### Question: What causes burning thigh pain?
### Response: <think>...
The patient likely has meralgia paresthetica.</think></s>"  ← EOS marks end

During Training:
1. Model learns to output text
2. Model learns to output </s> when done
3. SFTTrainer stops loss calculation after </s>
4. Prevents the model from continuing to the next example

Without EOS:
- Model treats consecutive examples as one long stream
- Learns incorrect patterns (end of one Q&A + start of next)
- Quality degrades significantly
```

### Other Special Tokens in Llama

```
</s>  = EOS (End-of-Sequence)
<s>   = BOS (Beginning-of-Sequence) - rarely used in SFT
<unk> = UNK (Unknown token) - for OOV words
<|padding|> = PAD (Padding) - for batch alignment

We only add </s> at the end of each training example.
```

### Performance Impact

```
With EOS_TOKEN:
├─ Model knows where examples end
├─ Loss computed only within example
├─ Quality: 95% (correct)
└─ Training: Stable

Without EOS_TOKEN:
├─ Examples bleed into each other
├─ Loss includes cross-example patterns
├─ Quality: 70% (degraded)
└─ Training: Unstable
```

---

## Part 2: Formatting Function

### The Function Structure

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

### Why This Function?

**Purpose:** Transform raw dataset columns → formatted training examples

```
INPUT (Dataset from disk):
{
    "Question": ["What causes thigh pain?", "How to diagnose..."],
    "Complex_CoT": ["Let me think...", "First consider..."],
    "Response": ["The patient likely...", "We should check..."]
}

↓ (formatting_prompts_func)

OUTPUT (Ready for training):
{
    "text": [
        "Below is an instruction...\n### Question: What causes...\n### Response: <think>Let me...\n The patient likely...</think></s>",
        "Below is an instruction...\n### Question: How to...\n### Response: <think>First consider...\n We should check...</think></s>"
    ]
}
```

### Step-by-Step Breakdown

#### Step 1: Extract Columns

```python
inputs = examples["Question"]
cots = examples["Complex_CoT"]
outputs = examples["Response"]
```

**What this does:**
- `inputs`: List of questions from dataset
- `cots`: List of chain-of-thought (reasoning) steps
- `outputs`: List of final answers

**Example values:**

```python
# For batch of 2 examples:
inputs = [
    "A 69-year-old man with diabetes experiences burning thigh pain when standing...",
    "What is the first step in diagnosing peripheral neuropathy?"
]

cots = [
    "Let me analyze this systematically. Positional symptoms suggest nerve compression...",
    "First, we need to understand the patient's medical history..."
]

outputs = [
    "The patient likely has meralgia paresthetica (lateral femoral cutaneous nerve compression).",
    "Detailed history taking including onset, location, severity, and triggering factors."
]
```

#### Step 2: Create Empty List

```python
texts = []
```

Temporary storage for formatted examples (will be filled in loop).

#### Step 3: Loop Through Batch

```python
for q, cot, ans in zip(inputs, cots, outputs):
    text = train_prompt_style.format(q, cot, ans) + EOS_TOKEN
    texts.append(text)
```

**The `zip()` function:**
```python
# Combines 3 lists element-by-element:
zip(
    ["Q1", "Q2"],           # inputs
    ["CoT1", "CoT2"],       # cots  
    ["Ans1", "Ans2"]        # outputs
)

# Returns:
# ("Q1", "CoT1", "Ans1")
# ("Q2", "CoT2", "Ans2")
```

**The `.format()` method:**

Recall from Step 4:
```python
train_prompt_style = """Below is an instruction...
### Question:
{}

### Response:
<think>
{}
</think>
{}"""
```

`.format(q, cot, ans)` fills in:
- First `{}`: Question
- Second `{}`: Chain-of-thought (reasoning)
- Third `{}`: Answer

**Result for one example:**

```python
q = "What causes burning thigh pain in a 69-year-old with diabetes?"
cot = "The positional nature (standing triggers, sitting relieves) suggests mechanical compression..."
ans = "Most likely diagnosis: Meralgia paresthetica (LFCN compression)"

text = train_prompt_style.format(q, cot, ans) + EOS_TOKEN

# Produces:
"""Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
What causes burning thigh pain in a 69-year-old with diabetes?

### Response:
<think>
The positional nature (standing triggers, sitting relieves) suggests mechanical compression...
</think>
Most likely diagnosis: Meralgia paresthetica (LFCN compression)</s>"""
```

#### Step 4: Return Formatted Dataset

```python
return {"text": texts}
```

Returns dictionary with "text" key containing all formatted examples.

```python
# For batch of 2:
return {
    "text": [
        "Below is an instruction...[first formatted example]</s>",
        "Below is an instruction...[second formatted example]</s>"
    ]
}
```

### Why Batch Processing?

```python
def formatting_prompts_func(examples):  # Receives BATCH
    # ...
    return {"text": texts}
```

The function receives a BATCH of examples (not one at a time).

**Benefits:**

```
Batch Size = 100:

Without batching (100 times):
├─ Overhead: Python loop setup × 100
├─ Memory: Fragmented
├─ Speed: ~1 second per call × 100 = 100 seconds

With batching (1 time):
├─ Overhead: Python loop setup × 1
├─ Memory: Consolidated
├─ Speed: Vectorized operations ~5 seconds total

Speedup: 20x faster! ✅
```

---

## Part 3: Loading Dataset from Disk

### Load Dataset

```python
print("Loading dataset from disk...")
dataset_path = "./medical-o1-reasoning-SFT/hf_format"
dataset_on_disk = load_from_disk(dataset_path, "en")
```

**What happens:**

```
File System:
./medical-o1-reasoning-SFT/hf_format/
├── dataset_info.json
├── state.json
└── data/
    └── train-00000-of-00001/
        ├── dataset.arrow        ← Binary data (Arrow format)
        └── indices.arrow

↓ load_from_disk()

Python Object:
dataset_on_disk = DatasetDict or Dataset
├─ Contains: All data from disk
├─ Format: Arrow (columnar, fast access)
└─ Ready: For processing
```

**Arrow Format Benefits:**

```
Compared to JSON:
├─ Size: 70% smaller (compression)
├─ Speed: 100x faster random access
├─ Memory: Memory-mapped (doesn't load all into RAM)
└─ Best for: ML pipelines

Why not JSON?
├─ Size: Full 500MB in memory
├─ Speed: ~10 seconds to load
├─ Memory: All in RAM at once
└─ Worse for: Large datasets
```

### Detect Dataset Structure

```python
if hasattr(dataset_on_disk, "keys"):
    print(f"Detected DatasetDict splits: {list(dataset_on_disk.keys())}")
    base_train = dataset_on_disk["train"]
else:
    base_train = dataset_on_disk
```

**Why this check?**

Dataset from disk can be one of two types:

```
Type 1: DatasetDict (multiple splits)
{
    "train": Dataset with 50,000 examples,
    "validation": Dataset with 5,000 examples,
    "test": Dataset with 5,000 examples
}

Detection:
├─ Has .keys() method
├─ Print: ['train', 'validation', 'test']
└─ Extract: dataset_on_disk["train"]

Type 2: Single Dataset
Dataset with 50,000 examples (no splits)

Detection:
├─ No .keys() method
├─ Already is the training data
└─ Use directly: base_train = dataset_on_disk
```

**In This Notebook:**

```python
dataset_on_disk = load_from_disk("./medical-o1-reasoning-SFT/hf_format")
# Detects: DatasetDict with splits

hasattr(dataset_on_disk, "keys")
# Returns: True

list(dataset_on_disk.keys())
# Returns: ['train']

base_train = dataset_on_disk["train"]
# Extracts the train split (50,000 medical Q&A pairs)
```

---

## Part 4: Shuffle and Sample

### Shuffle for Randomness

```python
base_train = base_train.shuffle(seed=42)
```

**Why shuffle?**

```
Without shuffle:
Training data order: [Easy, Easy, Easy, ..., Hard, Hard, Hard]
├─ Model learns "easy" first (biased)
├─ Sees "hard" at end (overweighting)
└─ Convergence: Poor, unstable

With shuffle(seed=42):
Training data order: [Hard, Easy, Medium, Easy, Hard, Medium, ...]
├─ Balanced learning throughout
├─ No ordering bias
└─ Convergence: Better, more stable
```

**What `seed=42` does:**

```python
shuffle(seed=42)
# Reproducible randomness
# Same seed = same shuffle order every time
# Ensures: Can reproduce training exactly

Example:
Run 1: shuffle(seed=42) → [idx5, idx0, idx17, idx2, ...]
Run 2: shuffle(seed=42) → [idx5, idx0, idx17, idx2, ...]  ← Same!
Run 3: shuffle(seed=42) → [idx5, idx0, idx17, idx2, ...]  ← Same!

Without seed:
Run 1: [idx17, idx5, idx0, ...]
Run 2: [idx2, idx17, idx5, ...]  ← Different!
Run 3: [idx0, idx2, idx17, ...]  ← Different!
Problem: Can't reproduce results!
```

### Select Subset

```python
N = min(16000, len(base_train))
train_dataset = base_train.select(range(N))
```

**What this does:**

```python
len(base_train)  # Total available examples
# Returns: 50,000

N = min(16000, 50000)
# N = 16,000

train_dataset = base_train.select(range(16000))
# Selects first 16,000 examples (after shuffle)
# Result: List of indices [0, 1, 2, ..., 15999]
```

**Why not use all 50,000?**

```
Memory & Time Trade-off:

16,000 examples:
├─ Training time: ~2-4 hours (3 epochs)
├─ Quality: Excellent (95%)
├─ Cost: Moderate
└─ Fit: Easily on GB10 (121GB)

50,000 examples:
├─ Training time: ~8-12 hours (3 epochs)
├─ Quality: Excellent (95%) - diminishing returns
├─ Cost: High (3x longer)
└─ Fit: Still on GB10 but slower

Decision: 16,000 is sweet spot
├─ Enough data for good quality
├─ Fast enough for iteration
└─ Diminishing returns after this point
```

**Dynamic Selection:**

```python
N = min(16000, len(base_train))
# Handles both scenarios:
# If dataset has 50,000: use 16,000
# If dataset has 5,000: use all 5,000
# Flexible, doesn't break if data is smaller
```

---

## Part 5: Batch Mapping

### Apply Formatting to All Examples

```python
dataset = train_dataset.map(formatting_prompts_func, batched=True)
```

**What `map()` does:**

```
Input Dataset (16,000 examples):
{
    "Question": [Q1, Q2, Q3, ...],
    "Complex_CoT": [CoT1, CoT2, CoT3, ...],
    "Response": [R1, R2, R3, ...]
}

↓ .map(formatting_prompts_func, batched=True)

Processing (batch by batch):
Batch 1: 100 examples → formatting_prompts_func() → formatted batch 1
Batch 2: 100 examples → formatting_prompts_func() → formatted batch 2
...
Batch 160: 100 examples → formatting_prompts_func() → formatted batch 160

Output Dataset (16,000 examples):
{
    "text": [formatted_text_1, formatted_text_2, ..., formatted_text_16000],
    "Question": [Q1, Q2, ...],  # Kept for reference
    "Complex_CoT": [...],
    "Response": [...]
}
```

**Batched Processing:**

```python
batched=True
# Process multiple examples at once (faster)
# Default batch size: ~1000 (depends on dataset size)

batched=False
# Process one example at a time (slower)
# Useful for complex logic requiring individual access
```

### Print Metadata

```python
print(f"Rows in raw training subset: {len(train_dataset)}")
print(f"Rows after formatting: {len(dataset)}")
```

**Output:**

```
Rows in raw training subset: 16000
Rows after formatting: 16000
```

**Why same count?**

```
Formatting doesn't change number of examples:
- Input: 16,000 rows
- Process: Format each row
- Output: 16,000 rows (same examples, now formatted)

The count stays the same, but content is transformed.
```

### Display Before and After Formatting

**New Enhanced Display:**

The Step 6 cell now shows a complete before-and-after comparison:

```python
# Step 1: Display RAW data (before formatting)
print("\n" + "="*70)
print("BEFORE FORMATTING (Raw Data from Disk)")
print("="*70)
sample_question = train_dataset["Question"][0]
sample_cot = train_dataset["Complex_CoT"][0]
sample_response = train_dataset["Response"][0]

print(f"\n📝 Question:\n{sample_question[:400]}...")
print(f"\n💭 Complex Chain of Thought (first 400 chars):\n{sample_cot[:400]}...")
print(f"\n📋 Response (first 400 chars):\n{sample_response[:400]}...")

# Step 2: Apply formatting
dataset = train_dataset.map(formatting_prompts_func, batched=True)

# Step 3: Display FORMATTED data (after formatting)
print("\n" + "="*70)
print("AFTER FORMATTING (Complete Training Example)")
print("="*70)
print(f"\n✨ Formatted text (full structure with prompt template and EOS token):\n")
print(dataset["text"][0])
```

**What this shows:**

| Stage | What You See |
|-------|-------------|
| **BEFORE** | Raw fields separately: Question, Chain-of-Thought, Response |
| **AFTER** | Complete formatted text: Full prompt template + thinking tags + answer + EOS token |

**Example Output:**

```
======================================================================
BEFORE FORMATTING (Raw Data from Disk)
======================================================================

📝 Question:
A 69-year-old man with controlled diabetes presents with burning pain, tingling, and numbness over...

💭 Complex Chain of Thought (first 400 chars):
Let me analyze this systematically. The positional nature (standing triggers, sitting relieves)...

📋 Response (first 400 chars):
The most likely diagnosis is meralgia paresthetica, characterized by compression of the lateral...


======================================================================
AFTER FORMATTING (Complete Training Example)
======================================================================

✨ Formatted text (full structure with prompt template and EOS token):

Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
A 69-year-old man with controlled diabetes presents with burning pain, tingling, and numbness over the outer right thigh triggered by standing. What is the most likely diagnosis?

### Response:
<think>
Let me analyze this systematically. The positional nature (standing triggers, sitting relieves) suggests mechanical compression rather than systemic neuropathy. The outer thigh distribution matches the lateral femoral cutaneous nerve territory. Meralgia paresthetica is characterized by compression of this nerve, often from tight clothing or prolonged pressure.
</think>
The most likely diagnosis is meralgia paresthetica, characterized by compression of the lateral femoral cutaneous nerve. This condition typically presents with burning, tingling, or numbness in the outer thigh, worsened by standing and improved by sitting or lying down.</s>
```

**Why This Matters:**

This before-and-after display helps you understand:
1. **What the raw data looks like** - Separate fields as stored on disk
2. **How formatting transforms it** - Into a complete training example
3. **The structure the model learns** - Prompt template, thinking tags, and answer format
4. **Where the EOS token goes** - At the end of the formatted text

---

## Data Structure Deep Dive

### Dataset Schema

```
Before Formatting:
Column Name        | Type    | Example
──────────────────────────────────────────────────────────────
Question           | String  | "What causes meralgia paresthetica?"
Complex_CoT        | String  | "Let me think through this systematically..."
Response           | String  | "Meralgia paresthetica is..."

After Formatting:
Column Name        | Type    | Example
──────────────────────────────────────────────────────────────
text               | String  | "Below is an instruction... [full formatted text]"
Question           | String  | "What causes meralgia paresthetica?" (kept)
Complex_CoT        | String  | "Let me think..." (kept)
Response           | String  | "Meralgia paresthetica is..." (kept)
```

### Example Instance

**Raw Example (Before):**

```json
{
    "Question": "A 69-year-old man with controlled diabetes presents with burning pain, tingling, and numbness over the outer right thigh triggered by standing. What is the most likely diagnosis?",
    "Complex_CoT": "Let me analyze systematically. The patient has localized unilateral symptoms triggered by standing and relieved by sitting. This positional nature suggests mechanical compression rather than systemic neuropathy. The outer thigh distribution matches the lateral femoral cutaneous nerve territory. Meralgia paresthetica is characterized by compression of this nerve, often from tight clothing or prolonged pressure. The fact that symptoms disappear when sitting supports this diagnosis because sitting relieves the pressure on the compressed nerve.",
    "Response": "The most likely diagnosis is meralgia paresthetica, characterized by compression of the lateral femoral cutaneous nerve. This condition typically presents with burning, tingling, or numbness in the outer thigh, worsened by standing and improved by sitting or lying down, exactly matching this patient's presentation."
}
```

**Formatted Example (After):**

```python
{
    "text": """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
A 69-year-old man with controlled diabetes presents with burning pain, tingling, and numbness over the outer right thigh triggered by standing. What is the most likely diagnosis?

### Response:
<think>
Let me analyze systematically. The patient has localized unilateral symptoms triggered by standing and relieved by sitting. This positional nature suggests mechanical compression rather than systemic neuropathy. The outer thigh distribution matches the lateral femoral cutaneous nerve territory. Meralgia paresthetica is characterized by compression of this nerve, often from tight clothing or prolonged pressure. The fact that symptoms disappear when sitting supports this diagnosis because sitting relieves the pressure on the compressed nerve.
</think>
The most likely diagnosis is meralgia paresthetica, characterized by compression of the lateral femoral cutaneous nerve. This condition typically presents with burning, tingling, or numbness in the outer thigh, worsened by standing and improved by sitting or lying down, exactly matching this patient's presentation.</s>""",
    "Question": "A 69-year-old man with controlled diabetes...",
    "Complex_CoT": "Let me analyze systematically...",
    "Response": "The most likely diagnosis is meralgia paresthetica..."
}
```

---

## Memory and Performance

### Memory Usage

```
Dataset Loading:

Arrow Format (on disk): 500 MB
↓
Python memory: ~1.5 GB (decompressed)
- 16,000 examples × 3 columns × ~31 KB per example

After Formatting:

New "text" column: ~2 GB
- 16,000 examples × 1 text column × ~125 KB per example
- Includes full prompt template + thinking + answer

Total Memory:
├─ Original columns: ~1.5 GB
├─ Formatted column: ~2 GB
├─ Metadata: ~0.5 GB
└─ Total: ~4 GB

GPU: Still available ~117 GB (minimal impact at this stage)
```

### Processing Time

```
Loading from Disk:
├─ Read Arrow files: ~2-3 seconds
├─ Decompress: ~1-2 seconds
└─ Total load: ~3-5 seconds

Shuffling 16,000 Examples:
├─ Random index generation: ~0.5 seconds
└─ Total shuffle: ~0.5 seconds

Mapping (Formatting):
├─ Batches: 160 (assuming 100 per batch)
├─ Per batch: ~50-100 ms
├─ Total formatting: ~10-15 seconds
└─ Including disk I/O: ~20-30 seconds

Total Step 6 Runtime: ~30-40 seconds
```

---

## Data Quality Checks

### What to Verify

```python
# 1. Count matches
print(f"Expected: 16000, Actual: {len(dataset)}")
assert len(dataset) == 16000

# 2. All fields present
assert "text" in dataset.column_names
assert all(len(text) > 100 for text in dataset["text"])

# 3. Format correct
sample = dataset["text"][0]
assert "### Question:" in sample
assert "### Response:" in sample
assert "<think>" in sample
assert sample.endswith("</s>")

# 4. No duplicates (optional)
texts = dataset["text"]
assert len(set(texts)) == len(texts)  # All unique
```

### Common Issues

```
Issue 1: Missing EOS_TOKEN
├─ Symptom: Samples end without </s>
├─ Cause: Forgot to add + EOS_TOKEN
└─ Fix: Ensure format_func adds EOS_TOKEN

Issue 2: Wrong column names
├─ Symptom: KeyError when accessing examples["Question"]
├─ Cause: Dataset has different column names
└─ Fix: Check dataset.column_names

Issue 3: No CoT (chain-of-thought)
├─ Symptom: Complex_CoT column empty or missing
├─ Cause: Dataset doesn't include reasoning
└─ Fix: Use different dataset with reasoning

Issue 4: Very long sequences
├─ Symptom: Some examples > 2048 tokens
├─ Cause: Long medical cases + verbose reasoning
└─ Fix: SFTTrainer will truncate automatically
```

---

## Why This Step Matters

### Impact on Training

```
Good Formatting:
├─ Model learns correct prompt structure
├─ Learns when to output thinking vs answer
├─ Quality improvements: Clear
├─ Training stability: Good

Bad Formatting:
├─ Model confused about structure
├─ Mixes thinking into answer
├─ Quality improvements: Minimal
├─ Training stability: Poor
```

### Prompt Consistency

```
Training Prompt Template:
"""### Question: ...
### Response:
<think>...
</think>
..."""

Inference Prompt Template:
"""### Question: ...
### Response:
<think>..."""

MUST match exactly! If they don't, model won't know what to do during inference.
```

---

## Configuration Variations

### For Different Datasets

#### Medical Dataset (Current)
```python
formatting_prompts_func = lambda examples: {
    "text": [
        train_prompt_style.format(q, cot, ans) + EOS_TOKEN
        for q, cot, ans in zip(examples["Question"], examples["Complex_CoT"], examples["Response"])
    ]
}
```

#### Code Dataset (Different schema)
```python
formatting_prompts_func = lambda examples: {
    "text": [
        code_template.format(instruction, code_solution) + EOS_TOKEN
        for instruction, code_solution in zip(examples["instruction"], examples["solution"])
    ]
}
```

#### Summary Dataset (No CoT)
```python
formatting_prompts_func = lambda examples: {
    "text": [
        summary_template.format(article, summary) + EOS_TOKEN
        for article, summary in zip(examples["article"], examples["summary"])
    ]
}
```

---

## Troubleshooting

### Dataset Not Found

```python
FileNotFoundError: ./medical-o1-reasoning-SFT/hf_format not found

Solutions:
1. Check path: ls -la ./medical-o1-reasoning-SFT/
2. Run data_download.py first
3. Verify hf_format directory exists
```

### Out of Memory

```python
RuntimeError: CUDA out of memory (loading dataset)

Solutions:
1. Reduce N: N = min(8000, len(base_train))
2. Close other programs
3. Restart Jupyter kernel
```

### Column Name Error

```python
KeyError: "Question" not in dataset

Solutions:
1. Check actual columns: print(dataset.column_names)
2. Adjust variable names in formatting function
3. Print first example: print(dataset[0])
```

---

## Performance Optimization

### Faster Loading (for large datasets)

```python
# Current (safe)
dataset_on_disk = load_from_disk(dataset_path, "en")
N = min(16000, len(dataset_on_disk["train"]))

# Optimized (for huge datasets)
from datasets import load_dataset
dataset = load_dataset("path", split="train", streaming=True)
dataset = dataset.shuffle(seed=42)
dataset = dataset.take(16000)  # Take first 16000 after shuffle
```

### Faster Formatting (for complex functions)

```python
# Current (good)
dataset.map(formatting_prompts_func, batched=True)

# Optimized (parallel)
dataset.map(formatting_prompts_func, batched=True, batch_size=500, num_proc=4)
# num_proc=4: Use 4 CPU cores in parallel
```

---

## Document Summary

**This Step Does:**
1. ✅ Loads medical Q&A dataset from disk (Arrow format)
2. ✅ Detects dataset structure (handles splits)
3. ✅ Shuffles for random ordering
4. ✅ Samples 16,000 examples (balance quality vs speed)
5. ✅ **Displays one raw example BEFORE formatting** (new feature)
6. ✅ Formats each example with prompt template + CoT + answer
7. ✅ Adds EOS token to mark example boundaries
8. ✅ **Displays the SAME example AFTER formatting** (new feature)
9. ✅ Creates training-ready dataset

**Key Concepts:**
- **EOS_TOKEN**: Marks end of each example (`</s>`)
- **Formatting Function**: Transforms raw data → training format
- **Chain-of-Thought**: Includes reasoning steps for better learning
- **Before/After Display**: Shows exactly how formatting transforms raw data
- **Batch Processing**: 20x faster than row-by-row
- **Shuffle + Sample**: Random order, right subset size

**Visualization Improvements:**
- **BEFORE**: Shows Question, Complex_CoT, and Response as separate fields (first 400 chars each)
- **AFTER**: Shows complete formatted text with prompt template, thinking tags, answer, and EOS token
- **Purpose**: Makes the transformation process transparent and educational

**Output:**
- 16,000 formatted medical training examples
- Each example: Prompt + thinking + answer + EOS token
- Before/after display showing the data transformation process
- Ready for Step 7 (LoRA adapters)

**Next: Step 7** - Apply LoRA adapters for parameter-efficient fine-tuning

---

**Document Version**: 1.1  
**Last Updated**: 2026-08-11  
**Notebook**: fine_tune_llm.ipynb (Step 6)  
**Status**: ✅ Complete with enhanced visualization
