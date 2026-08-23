#!/usr/bin/env python3
"""
Converts fine-tuned DeepSeek model to GGUF format for Ollama deployment.

Usage:
    python convert_to_ollama.py [--quantize Q4_K_M]

Outputs:
    - DeepSeek-R1-Medical-FT-8b-fp16.gguf (full precision)
    - DeepSeek-R1-Medical-FT-8b-Q4_K_M.gguf (quantized, recommended)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Configuration
MODEL_PATH = "./DeepSeek-R1-Medical-FT-8b-16bts"
OUTPUT_DIR = "./DeepSeek-R1-Medical-FT-8b-gguf"
LLAMA_CPP_DIR = "./llama.cpp"


def check_dependencies():
    """Check if required tools are available."""
    print("\n" + "="*70)
    print("  CHECKING DEPENDENCIES")
    print("="*70)

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Model not found: {MODEL_PATH}")
        print("  Make sure you've run fine_tune_llm.py first")
        sys.exit(1)
    print(f"✓ Model found: {MODEL_PATH}")

    # Check if config.json exists
    if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        print(f"✗ config.json not found in {MODEL_PATH}")
        sys.exit(1)
    print(f"✓ Model config found")

    # Check if llama.cpp exists
    if not os.path.exists(LLAMA_CPP_DIR):
        print(f"\n✗ llama.cpp not found: {LLAMA_CPP_DIR}")
        print("\nTo clone llama.cpp:")
        print("  git clone https://github.com/ggerganov/llama.cpp.git")
        print("  cd llama.cpp && make && cd ..")
        sys.exit(1)
    print(f"✓ llama.cpp found: {LLAMA_CPP_DIR}")

    # Check if convert.py exists
    convert_script = os.path.join(LLAMA_CPP_DIR, "convert.py")
    if not os.path.exists(convert_script):
        print(f"✗ convert.py not found in {LLAMA_CPP_DIR}")
        sys.exit(1)
    print(f"✓ Convert script found")

    # Check if quantize binary exists
    quantize_bin = os.path.join(LLAMA_CPP_DIR, "quantize")
    if not os.path.exists(quantize_bin):
        print(f"⚠ quantize binary not found: {quantize_bin}")
        print("  Skipping quantization (will only create fp16 GGUF)")
    else:
        print(f"✓ Quantize binary found")

    # Check Python packages
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    try:
        import transformers
        print(f"✓ Transformers installed: {transformers.__version__}")
    except ImportError:
        print("✗ Transformers not installed. Run: pip install transformers")
        sys.exit(1)

    print("\n✓ All dependencies satisfied!\n")


def convert_to_gguf():
    """Convert model to GGUF format."""
    print("="*70)
    print("  STEP 1: CONVERTING TO GGUF (fp16)")
    print("="*70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    convert_script = os.path.join(LLAMA_CPP_DIR, "convert.py")
    output_file = os.path.join(OUTPUT_DIR, "model-fp16.gguf")

    cmd = [
        "python",
        convert_script,
        "--model-dir", MODEL_PATH,
        "--outfile", output_file,
        "--outtype", "f16",
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        print(f"\n✓ Conversion complete!")
        print(f"  Output: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Conversion failed: {e}")
        sys.exit(1)


def quantize_model(input_file, quantization_type="Q4_K_M"):
    """Quantize model to specified format."""
    print("\n" + "="*70)
    print(f"  STEP 2: QUANTIZING TO {quantization_type}")
    print("="*70)

    quantize_bin = os.path.join(LLAMA_CPP_DIR, "quantize")

    if not os.path.exists(quantize_bin):
        print(f"\n⚠ Quantize binary not found: {quantize_bin}")
        print("  Skipping quantization")
        return input_file

    # Generate output filename
    base_name = "model-" + quantization_type.lower()
    output_file = os.path.join(OUTPUT_DIR, f"{base_name}.gguf")

    cmd = [
        quantize_bin,
        input_file,
        output_file,
        quantization_type,
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        print(f"\n✓ Quantization complete!")
        print(f"  Output: {output_file}")

        # Show file sizes
        fp16_size = os.path.getsize(input_file) / (1024**3)
        quant_size = os.path.getsize(output_file) / (1024**3)
        print(f"\n  Size comparison:")
        print(f"    FP16: {fp16_size:.2f} GB")
        print(f"    {quantization_type}: {quant_size:.2f} GB ({quant_size/fp16_size*100:.1f}%)")

        return output_file
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Quantization failed: {e}")
        print("  Continuing with fp16 model instead...")
        return input_file


def create_modelfile(gguf_path):
    """Create Modelfile for Ollama."""
    print("\n" + "="*70)
    print("  STEP 3: CREATING MODELFILE")
    print("="*70)

    # Get absolute path for the GGUF file
    abs_gguf_path = os.path.abspath(gguf_path)

    modelfile_content = f"""FROM {abs_gguf_path}

# Model parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40

# Stop tokens
PARAMETER stop "<|im_end|>"
PARAMETER stop "###Response:"
PARAMETER stop "</think>"

# System prompt for medical reasoning
SYSTEM \"\"\"You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Provide detailed, evidence-based medical responses with step-by-step reasoning.
Think carefully through the clinical presentation and differential diagnoses.\"\"\"
"""

    modelfile_path = "Modelfile-medical-ft"

    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"\n✓ Modelfile created: {modelfile_path}")
    print(f"\nContent:\n{'-'*70}")
    print(modelfile_content)
    print(f"{'-'*70}\n")

    return modelfile_path


def create_ollama_model(modelfile_path):
    """Import model into Ollama."""
    print("="*70)
    print("  STEP 4: IMPORTING INTO OLLAMA")
    print("="*70)

    # Check if ollama is installed
    try:
        subprocess.run(["ollama", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n⚠ Ollama not found or not running")
        print("\nTo complete setup manually:")
        print(f"  1. Make sure Ollama is installed: https://ollama.ai")
        print(f"  2. Run: ollama create deepseek-medical-ft -f {modelfile_path}")
        print(f"  3. Run: ollama run deepseek-medical-ft")
        return False

    cmd = ["ollama", "create", "deepseek-medical-ft", "-f", modelfile_path]

    print(f"\nRunning: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        print(f"\n✓ Model imported successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Import failed: {e}")
        print(f"\nTry manually:")
        print(f"  ollama create deepseek-medical-ft -f {modelfile_path}")
        return False


def print_summary(gguf_path, modelfile_path, ollama_imported):
    """Print summary of completed steps."""
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)

    print(f"\n✓ Model conversion complete!")
    print(f"\nOutput files:")
    print(f"  GGUF model: {os.path.abspath(gguf_path)}")
    print(f"  Modelfile:  {os.path.abspath(modelfile_path)}")

    print(f"\nNext steps:")
    if ollama_imported:
        print(f"  1. Test the model: ollama run deepseek-medical-ft")
        print(f"  2. Query from Python (examples in OLLAMA_DEPLOYMENT.md)")
        print(f"  3. Use in your applications (LangChain, Gradio, etc.)")
    else:
        print(f"  1. Import the model:")
        print(f"     ollama create deepseek-medical-ft -f {modelfile_path}")
        print(f"  2. Test: ollama run deepseek-medical-ft")
        print(f"  3. Use in your applications")

    print(f"\nDocumentation: CH11-Finetuning/OLLAMA_DEPLOYMENT.md\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert fine-tuned DeepSeek model to GGUF for Ollama"
    )
    parser.add_argument(
        "--quantize",
        choices=["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q8_0"],
        default="Q4_K_M",
        help="Quantization level (default: Q4_K_M)"
    )
    parser.add_argument(
        "--skip-quantize",
        action="store_true",
        help="Skip quantization (keep fp16 only)"
    )
    parser.add_argument(
        "--skip-ollama",
        action="store_true",
        help="Skip Ollama import (just create GGUF)"
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("  OLLAMA DEPLOYMENT CONVERTER")
    print("  DeepSeek-R1-Medical-FT-8b")
    print("="*70)

    # Check dependencies
    check_dependencies()

    # Step 1: Convert to GGUF
    fp16_gguf = convert_to_gguf()

    # Step 2: Quantize
    if args.skip_quantize:
        final_gguf = fp16_gguf
        print(f"\n⊘ Quantization skipped (using fp16)")
    else:
        final_gguf = quantize_model(fp16_gguf, args.quantize)

    # Step 3: Create Modelfile
    modelfile = create_modelfile(final_gguf)

    # Step 4: Import to Ollama
    ollama_imported = False
    if not args.skip_ollama:
        ollama_imported = create_ollama_model(modelfile)

    # Summary
    print_summary(final_gguf, modelfile, ollama_imported)


if __name__ == "__main__":
    main()
