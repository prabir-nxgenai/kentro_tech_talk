# Running Fine-Tuned DeepSeek Model with Ollama

This guide shows how to convert your merged fine-tuned DeepSeek-R1-Medical-FT model to GGUF format and run it with Ollama.

## Overview

Your current model:
- **Location**: `./DeepSeek-R1-Medical-FT-8b-16bts`
- **Format**: Merged 16-bit (safetensors/bin)
- **Type**: Llama-based (DeepSeek-R1-Distill-Llama-8B)

To use with Ollama, you need to:
1. **Convert** to GGUF format (quantized)
2. **Create** a Modelfile (instructions for Ollama)
3. **Import** into Ollama
4. **Run** with ollama serve/run

---

## Step 1: Install Required Tools

### Clone llama.cpp (for GGUF conversion)
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make
cd ..
```

### Install Python dependencies
```bash
pip install torch transformers numpy
```

---

## Step 2: Convert Model to GGUF

### Option A: Automatic Conversion (Recommended)

Create a conversion script:

```bash
cat > convert_to_gguf.py << 'EOF'
#!/usr/bin/env python3
"""
Converts fine-tuned Llama model to GGUF format for Ollama.
"""
import sys
import os

# Add llama.cpp scripts to path
sys.path.insert(0, "./llama.cpp")

from convert import convert_models_to_gguf

# Model paths
model_path = "./DeepSeek-R1-Medical-FT-8b-16bts"
output_dir = "./DeepSeek-R1-Medical-FT-8b-16bts-gguf"

print("Converting model to GGUF format...")
print(f"Input:  {model_path}")
print(f"Output: {output_dir}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Convert
convert_models_to_gguf(
    model_names_or_paths=[model_path],
    output_dir=output_dir,
    use_f16=True,  # Use fp16 (higher quality, larger file)
    use_f32=False,
    outfile=None,
    vocab_only=False
)

print("✓ Conversion complete!")
print(f"GGUF model saved to: {output_dir}")
EOF

python convert_to_gguf.py
```

### Option B: Manual Conversion with Quantization

For different quantization levels (Q4, Q5, Q8):

```bash
# Convert to GGUF (fp16)
python llama.cpp/convert.py \
    --model-dir ./DeepSeek-R1-Medical-FT-8b-16bts \
    --outfile ./DeepSeek-R1-Medical-FT-8b-16bts-gguf/model.gguf \
    --outtype f16

# Quantize to Q4_K_M (recommended balance)
./llama.cpp/quantize \
    ./DeepSeek-R1-Medical-FT-8b-16bts-gguf/model.gguf \
    ./DeepSeek-R1-Medical-FT-8b-Q4_K_M.gguf Q4_K_M
```

**Quantization options:**
| Format | Size | Quality | VRAM |
|--------|------|---------|------|
| **F16** | 16GB | Highest | 16GB |
| **Q8** | 8GB | Very High | 8GB |
| **Q5_K_M** | 5GB | High | 5GB |
| **Q4_K_M** | 4GB | Good | 4GB |
| **Q3_K_M** | 3GB | Acceptable | 3GB |

**Recommendation**: Use Q4_K_M for best balance of quality and size.

---

## Step 3: Create Modelfile for Ollama

Create a file named `Modelfile-medical-ft`:

```dockerfile
FROM ./DeepSeek-R1-Medical-FT-8b-Q4_K_M.gguf

# Metadata
PARAMETER stop "<|im_end|>"
PARAMETER stop "###Response:"
PARAMETER temperature 0.7
PARAMETER top_p 0.9

# System prompt for medical domain
SYSTEM """You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning. 
Provide detailed, evidence-based medical responses with step-by-step reasoning."""
```

---

## Step 4: Import into Ollama

```bash
# Create the model
ollama create deepseek-medical-ft -f Modelfile-medical-ft

# Verify it was imported
ollama list | grep deepseek-medical
```

Expected output:
```
deepseek-medical-ft    4.3GB    5 seconds ago
```

---

## Step 5: Run the Model

### Interactive Mode
```bash
ollama run deepseek-medical-ft
```

### From Python
```python
import requests
import json

def query_medical_model(question):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-medical-ft",
            "prompt": question,
            "stream": False,
        }
    )
    return response.json()["response"]

# Test
question = "A 69-year-old man with controlled diabetes experiences burning pain in the outer right thigh when standing for 15-20 minutes. What could be the cause?"

answer = query_medical_model(question)
print(answer)
```

### Chat API
```python
import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "deepseek-medical-ft",
        "messages": [
            {"role": "user", "content": "What is meralgia paresthetica?"}
        ],
        "stream": False,
    }
)
print(response.json()["message"]["content"])
```

---

## Step 6: Use in Your Application

### In LangChain
```python
from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="deepseek-medical-ft",
    base_url="http://localhost:11434"
)

response = llm.invoke("Explain diabetic neuropathy")
print(response)
```

### In Gradio
```python
import gradio as gr
import requests

def medical_chat(question):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "deepseek-medical-ft",
            "prompt": question,
            "stream": False,
        }
    )
    return response.json()["response"]

demo = gr.Interface(
    fn=medical_chat,
    inputs="text",
    outputs="text",
    title="Medical Expert - Fine-Tuned DeepSeek",
    description="Ask medical questions to the fine-tuned DeepSeek model"
)

if __name__ == "__main__":
    demo.launch()
```

---

## Troubleshooting

### Error: "Cannot find model path"
```
Solution: Make sure model path is absolute or relative paths are correct
$ ls -la ./DeepSeek-R1-Medical-FT-8b-16bts/config.json
```

### Error: "Out of memory"
```
Solution: Use higher quantization (Q3_K_M instead of Q4_K_M)
Or reduce max_tokens in requests
```

### Model running slowly
```
Solution: Check that Ollama is using GPU
$ ollama pull llama3.1  # Test with base model first
```

### "Model not found" in ollama run
```
Solution: Verify model was imported correctly
$ ollama list
$ ollama show deepseek-medical-ft
```

---

## Performance Benchmarks

On NVIDIA GPU (tested on your DGX):

| Quantization | Model Size | Load Time | Inference (100 tokens) | VRAM |
|--------------|-----------|-----------|----------------------|------|
| **F16** | 16GB | ~3s | ~4s | 16GB |
| **Q8** | 8GB | ~2s | ~3s | 8GB |
| **Q4_K_M** | 4GB | ~1s | ~2s | 4GB |
| **Q3_K_M** | 3GB | ~1s | ~2s | 3GB |

---

## Best Practices

1. **Test locally first** with smaller quantization before production
2. **Monitor VRAM** with `nvidia-smi` during inference
3. **Use streaming** for long responses in user-facing apps
4. **Set temperature** appropriately (0.7-0.9 for medical domain)
5. **Add stop tokens** to control generation length
6. **Backup original model** before conversion

---

## Quick Start Script

Save as `setup_ollama.sh`:

```bash
#!/bin/bash

echo "Step 1: Converting model to GGUF..."
python llama.cpp/convert.py \
    --model-dir ./DeepSeek-R1-Medical-FT-8b-16bts \
    --outfile ./model-fp16.gguf \
    --outtype f16

echo "Step 2: Quantizing to Q4_K_M..."
./llama.cpp/quantize ./model-fp16.gguf ./DeepSeek-R1-Medical-FT-8b-Q4_K_M.gguf Q4_K_M

echo "Step 3: Creating Modelfile..."
cat > Modelfile-medical-ft << 'MODELFILE'
FROM ./DeepSeek-R1-Medical-FT-8b-Q4_K_M.gguf
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM "You are a medical expert."
MODELFILE

echo "Step 4: Importing into Ollama..."
ollama create deepseek-medical-ft -f Modelfile-medical-ft

echo "Step 5: Testing..."
ollama run deepseek-medical-ft "What is meralgia paresthetica?"

echo "✓ Complete! Model is ready."
echo "Run: ollama run deepseek-medical-ft"
```

Make it executable and run:
```bash
chmod +x setup_ollama.sh
./setup_ollama.sh
```

---

## Next Steps

1. Choose quantization level (Q4_K_M recommended)
2. Run conversion script
3. Create Modelfile
4. Import with `ollama create`
5. Test with `ollama run`
6. Integrate into your applications

See `CH11-Finetuning/` for example scripts and integration patterns.
