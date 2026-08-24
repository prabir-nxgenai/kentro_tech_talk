#!/usr/bin/env python3
"""
Quantize a HuggingFace format model locally and save quantized version.

Supports 8-bit and 4-bit quantization using bitsandbytes.
Saves quantized model in HuggingFace format with quantization level in the name.

Usage:
    python quantize_hf_model.py <model_dir> --bits 4 [--output_dir <dir>]
    python quantize_hf_model.py <model_dir> --bits 8

    
Example:
    python quantize_hf_model.py ./local-Mistral-7B-Instruct-v0.2 --bits 4
    Output: ./local-Mistral-7B-Instruct-v0.2-Q4

Requirements:
    pip install transformers bitsandbytes accelerate
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def quantize_model(model_dir: str, bits: int = 4, output_dir: str = None) -> bool:
    """
    Quantize a HuggingFace model and save locally.

    Args:
        model_dir: Path to local HF model directory
        bits: Quantization bits (4 or 8)
        output_dir: Output directory (default: model_dir-Q{bits})

    Returns:
        True if successful
    """
    model_path = Path(model_dir)

    if not model_path.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return False

    if bits not in [4, 8]:
        print(f"Error: bits must be 4 or 8, got {bits}")
        return False

    if output_dir is None:
        output_dir = str(model_path.parent / f"{model_path.name}-Q{bits}")
    else:
        output_dir = str(Path(output_dir))

    print(f"Loading model from: {model_dir}")
    print(f"Quantization: {bits}-bit")
    print(f"Output directory: {output_dir}")

    try:
        # Try loading with regex fix first (preferred for newer transformers versions)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
                fix_mistral_regex=True,  # Automatically fix Mistral regex pattern if needed
            )
            print("✅ Tokenizer loaded successfully")
        except (AttributeError, TypeError) as e:
            # Fall back to loading without the flag for older transformers versions
            # or if backend_tokenizer is not available
            print(f"  Note: fix_mistral_regex not available, using standard tokenizer loading")

            # Suppress the Mistral regex warning with a filter
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    trust_remote_code=True,
                )
            print("✅ Tokenizer loaded successfully (warning suppressed)")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=(bits == 4),
            load_in_8bit=(bits == 8),
            bnb_4bit_compute_dtype=torch.bfloat16 if bits == 4 else None,
            bnb_4bit_quant_type="nf4" if bits == 4 else None,
            bnb_4bit_use_double_quant=True if bits == 4 else None,
        )

        print(f"Loading model with {bits}-bit quantization...")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        print(f"Model loaded and quantized to {bits}-bit")
        print(f"\nSaving quantized model to: {output_dir}")

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        config_file = Path(model_dir) / "config.json"
        if config_file.exists():
            import shutil

            shutil.copy(config_file, Path(output_dir) / "config.json")
            print("Config copied")

        print(f"Successfully saved {bits}-bit quantized model!")
        print(f"\nTo use this model:")
        print(f"  from transformers import AutoModelForCausalLM")
        print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
        print(f"\nTo load with Ollama:")
        print(f"  python load_gguf_to_ollama.py {output_dir} model-q{bits}")

        return True

    except Exception as e:
        print(f"Error during quantization: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Quantize a HuggingFace format model locally"
    )
    parser.add_argument("model_dir", help="Path to local HuggingFace model directory")
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="Quantization bits (default: 4)",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: model_dir-Q{bits})",
    )

    args = parser.parse_args()

    print("Quantize HuggingFace Model Locally")
    print("=" * 50)

    success = quantize_model(args.model_dir, args.bits, args.output_dir)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
