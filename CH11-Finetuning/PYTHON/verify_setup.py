#!/usr/bin/env python3
"""
Verify that all required files for fine-tuning are in place.
Run this before starting the notebook to ensure everything is ready.
"""

import os
import sys
from pathlib import Path

def check_file(path, description):
    """Check if a file/directory exists."""
    if os.path.exists(path):
        if os.path.isdir(path):
            size = sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file()) / (1024**3)
            print(f"  ✅ {description}: {path} ({size:.2f} GB)")
        else:
            size = os.path.getsize(path) / (1024**2)
            print(f"  ✅ {description}: {path} ({size:.2f} MB)")
        return True
    else:
        print(f"  ❌ {description}: NOT FOUND at {path}")
        return False

def main():
    print("\n" + "="*70)
    print("FINE-TUNING SETUP VERIFICATION")
    print("="*70 + "\n")

    all_good = True

    # Check model
    print("1️⃣  Model Files:")
    model_path = "./Meta-Llama-3.1-8B-bnb-4bit"
    if not check_file(model_path, "Llama 3.1 8B (4-bit)"):
        all_good = False
        print(f"      → Run Step 0 in finetune-v2-fixed.ipynb to auto-download")

    # Check dataset
    print("\n2️⃣  Dataset Files:")
    dataset_path = "./python_code_instructions_18k_alpaca/hf_format"
    if not check_file(dataset_path, "Python Code Instructions (18K)"):
        all_good = False
        print(f"      → Run Step 0 in finetune-v2-fixed.ipynb to auto-download")
    else:
        # Check for required files
        required_files = [
            "dataset_info.json",
            "data-00000-of-00001.arrow",
        ]
        for req_file in required_files:
            full_path = os.path.join(dataset_path, req_file)
            if os.path.exists(full_path):
                print(f"     ✓ {req_file}")
            else:
                print(f"     ✗ MISSING: {req_file}")
                all_good = False

    # Check Python packages
    print("\n3️⃣  Python Packages:")
    required_packages = [
        ("unsloth", "Unsloth"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("datasets", "Hugging Face Datasets"),
        ("trl", "TRL (Transformer RL)"),
        ("peft", "PEFT (LoRA)"),
    ]

    for module, name in required_packages:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - NOT INSTALLED")
            all_good = False

    # Check GPU
    print("\n4️⃣  GPU Availability:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print(f"  ❌ CUDA not available - GPU acceleration disabled")
            all_good = False
    except Exception as e:
        print(f"  ❌ Error checking CUDA: {e}")
        all_good = False

    # Final status
    print("\n" + "="*70)
    if all_good:
        print("✅ ALL CHECKS PASSED - Ready to fine-tune!")
        print("\nNext steps:")
        print("  1. Open finetune-v2-fixed.ipynb")
        print("  2. Run all cells from top to bottom")
        print("  3. Monitor training progress")
    else:
        print("⚠️  SOME CHECKS FAILED - See above for details")
        print("\nTo fix:")
        print("  1. Run Step 0 in finetune-v2-fixed.ipynb to auto-download missing files")
        print("  2. Install missing packages: pip install -r requirements.txt")
        print("  3. Ensure CUDA/GPU is available: nvidia-smi")
    print("="*70 + "\n")

    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
