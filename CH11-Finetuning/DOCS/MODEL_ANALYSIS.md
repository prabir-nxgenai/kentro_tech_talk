# Model Architecture Analysis Guide

This guide explains how to extract and compare model architectures in the dgx-book project using two complementary tools.

## Quick Start

### 1. Analyze Fine-Tuned Model Architecture

```bash
python analyze_model_architecture.py
```

This script:
- Loads both base and fine-tuned models
- Extracts detailed architecture information
- Compares parameter counts and structure
- Shows LoRA adapter configuration
- Saves results to `model_architecture_comparison.json`

**Output includes:**
- Model type, hidden size, number of layers
- Total/trainable/frozen parameter counts
- LoRA configuration (rank, alpha, dropout)
- Side-by-side comparison of base vs fine-tuned
- Key insights about training efficiency

### 2. Multi-Source Model Comparison

```bash
python model_comparison_tool.py
```

This script:
- Analyzes models from multiple sources:
  - **Local models**: Fine-tuned models on disk
  - **Ollama models**: Models running on Ollama server
  - **Hugging Face**: Remote models from Hub
- Compares models side-by-side
- Exports to JSON and CSV formats

**Output formats:**
- `models_comparison.json`: Complete architecture details
- `models_comparison.csv`: Tabular comparison for spreadsheets

---

## Detailed Usage

### Analyze Fine-Tuned Model

The `analyze_model_architecture.py` script is specialized for comparing base and fine-tuned models:

```python
# Script automatically compares:
# 1. Base model: ./local-DeepSeek-R1-Distill-Llama-8B
# 2. Fine-tuned model: ./DeepSeek-R1-Medical-FT-8b-16bts
```

**What it extracts:**

```
Model Type:                LlamaForCausalLM
Hidden Size:               4096
Number of Layers:          32
Attention Heads:           32
Vocabulary Size:           128,256

Parameters:
  Total:                   8.03B
  Trainable:               1.68M (LoRA adapters)
  Frozen:                  8.03B

LoRA Adapter Configuration:
  Rank (r):                16
  Alpha:                   16
  Dropout:                 0
  Target Modules:          q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

### Multi-Source Model Comparison

Customize the `model_comparison_tool.py` to analyze different models:

```python
# Add Ollama models
analyzer.add_model("LLaMA 3.1", "ollama", "llama3.1")
analyzer.add_model("Embedding Model", "ollama", "nomic-embed-text")

# Add Hugging Face models
analyzer.add_model("Base Model", "huggingface", "meta-llama/Llama-2-7b-hf")

# Add local models
analyzer.add_model("My Fine-Tune", "local", "./my-finetuned-model")

# Compare any two models
analyzer.compare_models("LLaMA 3.1", "My Fine-Tune")
```

---

## Understanding Model Architecture Terms

### Core Parameters

| Term | Meaning | Example |
|------|---------|---------|
| **Hidden Size** | Dimension of internal representations | 4096 (larger = more capacity) |
| **Layers** | Number of transformer blocks | 32 (more = deeper, better reasoning) |
| **Attention Heads** | Number of parallel attention mechanisms | 32 (more = richer attention patterns) |
| **Intermediate Size** | Dimension of feed-forward layers | 11008 (usually 2.67x hidden size) |
| **Vocab Size** | Number of unique tokens | 128,256 |
| **Max Positions** | Maximum context length | 2048, 4096, 32768 |

### Parameter Counts

```
Total Parameters = all model weights
├── Frozen Parameters: base model weights (not trained)
└── Trainable Parameters: LoRA adapters only
```

### LoRA (Low-Rank Adaptation)

**Configuration fields:**
- **Rank (r)**: Size of adapter matrices (8-32 typical)
  - Higher = more capacity, more memory
  - 16 is a good balance
- **Alpha**: Scaling factor (usually equals r)
- **Dropout**: Regularization for adapter layers (0.0-0.1 typical)
- **Target Modules**: Which transformer components get adapters
  - q_proj, k_proj, v_proj, o_proj: attention projections
  - gate_proj, up_proj, down_proj: MLP layers

---

## Interpreting Output

### Comparison Results

#### Same Architecture ✓
```
✓ Base architecture is IDENTICAL (no structural changes)
  • LoRA adapters add ~1.68M trainable parameters
  • Base model weights (8.03B) remain frozen
  • Training efficiency: 0.02% learnable
```

**Meaning**: Only LoRA adapters were trained, base model unchanged.

#### Different Architecture ✗
```
✗ Architecture DIFFERS between base and fine-tuned
  - Hidden size: 4096 → 2048
  - Layers: 32 → 16
```

**Meaning**: Structural modifications were made (merging, pruning, etc.).

### Parameter Analysis

```
Training efficiency = Trainable Parameters / Total Parameters
```

**Examples:**
- **0.02% (LoRA)**: Very efficient, 500x reduction in training memory
- **100% (Full)**: All parameters trainable, requires most memory
- **10% (Selective)**: Only certain layers trained

---

## Common Use Cases

### 1. Verify Fine-Tuning Didn't Change Structure
```bash
python analyze_model_architecture.py | grep "✓ Base architecture"
```

### 2. Compare Parameter Efficiency Across Models
```bash
python model_comparison_tool.py
# Check "Training efficiency" row in comparison table
```

### 3. Extract Config for Deployment
```bash
python analyze_model_architecture.py
cat model_architecture_comparison.json | grep "hidden_size"
```

### 4. Track Training Impact
Run before and after training:
```bash
# Before training
cp model_architecture_comparison.json model_comparison_before.json

# After training
python analyze_model_architecture.py

# Compare
diff model_comparison_before.json model_architecture_comparison.json
```

---

## Output Files

### `model_architecture_comparison.json`
```json
{
  "base_model": {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "total_parameters": 8031000000
  },
  "finetuned_model": {
    "model_type": "llama",
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "total_parameters": 8031000000,
    "trainable_parameters": 1680000
  },
  "lora_adapter": {
    "lora_rank": 16,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", ...]
  }
}
```

### `models_comparison.csv`
```
Model,architecture,hidden_size,num_layers,vocab_size,...
LLaMA 3.1,llama,4096,32,128256,...
DeepSeek Medical FT,llama,4096,32,128256,...
```

---

## Troubleshooting

### Error: "Cannot connect to Ollama"
```
Solution: Ensure Ollama server is running
$ ollama serve
```

### Error: "No config.json found"
```
Solution: Model path must be a directory containing config.json
Check: ls ./your-model-path/config.json
```

### Error: "Cannot import transformers"
```
Solution: Install required packages
$ pip install transformers huggingface_hub
```

### Missing LoRA Information
```
Meaning: Model is not using LoRA (full fine-tuning or base model)
Check: ls ./model-path/adapter_config.json
```

---

## Advanced: Custom Analysis

Modify the scripts to extract specific metrics:

```python
# In analyze_model_architecture.py
def get_model_info(model_name, model_path=None):
    # Add custom extraction here
    info["custom_metric"] = custom_calculation(model)
    return info
```

```python
# In model_comparison_tool.py
def compare_models(self, *model_names):
    # Add custom comparison logic
    efficiency = trainable / total * 100
    print(f"Training efficiency: {efficiency:.2f}%")
```

---

## References

- **Hugging Face Config Docs**: https://huggingface.co/docs/transformers/model_doc/
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **DeepSeek Model Card**: https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B
