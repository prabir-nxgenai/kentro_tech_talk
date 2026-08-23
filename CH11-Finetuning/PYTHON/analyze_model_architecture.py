#!/usr/bin/env python3
"""
Extracts and compares model architecture information.

This script:
1. Loads base and fine-tuned models
2. Extracts architecture details (layers, parameters, config)
3. Compares them side-by-side
4. Shows LoRA adapter information
"""

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import json
import os
from peft import get_peft_model_state_dict


def get_model_info(model_name, model_path=None):
    """Extract detailed architecture information from a model."""
    try:
        if model_path:
            config = AutoConfig.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
        else:
            config = AutoConfig.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        "model_type": config.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": config.max_position_embeddings,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": total_params - trainable_params,
    }

    # Add model-specific info
    if hasattr(config, "rope_scaling"):
        info["rope_scaling"] = config.rope_scaling
    if hasattr(config, "attention_dropout"):
        info["attention_dropout"] = config.attention_dropout

    return info, model, config


def get_lora_info(model_path, finetuned_info=None, base_info=None):
    """Extract LoRA adapter information."""
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    config_path = os.path.join(model_path, "config.json")

    # Check if model is merged by examining parameter counts
    # Merged models: total_params increased but trainable_params = 0
    if finetuned_info and base_info:
        is_merged = (
            finetuned_info["trainable_parameters"] == 0 and
            finetuned_info["total_parameters"] > base_info["total_parameters"]
        )
        if is_merged:
            return {"type": "merged", "status": "LoRA adapters have been merged into base model weights"}

    # If no trainable params but has adapter config, it's merged
    if finetuned_info and finetuned_info["trainable_parameters"] == 0 and os.path.exists(adapter_config_path):
        return {"type": "merged", "status": "LoRA adapters have been merged into base model weights"}

    # Check if it's a PEFT model (has adapter config AND trainable params)
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)

        lora_info = {
            "type": "peft",
            "lora_rank": adapter_config.get("r", "N/A"),
            "lora_alpha": adapter_config.get("lora_alpha", "N/A"),
            "lora_dropout": adapter_config.get("lora_dropout", "N/A"),
            "target_modules": adapter_config.get("target_modules", []),
            "bias": adapter_config.get("bias", "N/A"),
        }
        return lora_info

    return None


def format_number(num):
    """Format large numbers with commas and K/M/B suffixes."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    return str(num)


def print_model_info(name, info, model_path=None):
    """Pretty-print model architecture information."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    if model_path:
        print(f"  Path: {model_path}")
    print(f"{'='*60}")
    if info is None:
        print("  Could not load model information")
        return

    print(f"  Model Type:                {info['model_type']}")
    print(f"  Hidden Size:               {info['hidden_size']}")
    print(f"  Number of Layers:          {info['num_hidden_layers']}")
    print(f"  Attention Heads:           {info['num_attention_heads']}")
    print(f"  Intermediate Size (FFN):   {info['intermediate_size']}")
    print(f"  Vocabulary Size:           {format_number(info['vocab_size'])}")
    print(f"  Max Position Embeddings:   {format_number(info['max_position_embeddings'])}")
    print(f"\n  Parameters:")
    print(f"    Total:                   {format_number(info['total_parameters'])}")
    print(f"    Trainable:               {format_number(info['trainable_parameters'])}")
    print(f"    Frozen:                  {format_number(info['frozen_parameters'])}")

    if "rope_scaling" in info and info["rope_scaling"]:
        print(f"\n  RoPE Scaling:              {info['rope_scaling']}")
    if "attention_dropout" in info:
        print(f"  Attention Dropout:         {info['attention_dropout']}")


def print_lora_info(name, lora_info):
    """Pretty-print LoRA adapter information."""
    if lora_info is None:
        print(f"\n  No LoRA or adapter configuration found for {name}")
        return

    if lora_info.get("type") == "merged":
        print(f"\n  Fine-Tuning Method: MERGED LoRA")
        print(f"    Status:                  {lora_info['status']}")
        print(f"    Effect:                  All parameters frozen (no longer trainable)")
        return

    print(f"\n  LoRA Adapter Configuration (PEFT - Not Merged):")
    print(f"    Rank (r):                {lora_info['lora_rank']}")
    print(f"    Alpha:                   {lora_info['lora_alpha']}")
    print(f"    Dropout:                 {lora_info['lora_dropout']}")
    print(f"    Target Modules:          {', '.join(lora_info['target_modules'])}")
    print(f"    Bias:                    {lora_info['bias']}")


def compare_architectures(base_info, finetuned_info, base_name="Base Model", finetuned_name="Fine-Tuned Model"):
    """Compare two model architectures."""
    print(f"\n{'='*60}")
    print(f"  Architecture Comparison")
    print(f"  {base_name} vs {finetuned_name}")
    print(f"{'='*60}")

    if base_info is None or finetuned_info is None:
        print("  Cannot compare: one or both models could not be loaded")
        return

    # Check if architectures are the same
    same_architecture = (
        base_info["model_type"] == finetuned_info["model_type"]
        and base_info["hidden_size"] == finetuned_info["hidden_size"]
        and base_info["num_hidden_layers"] == finetuned_info["num_hidden_layers"]
    )

    if same_architecture:
        print("\n  ✓ Base architecture is IDENTICAL (no structural changes)")
    else:
        print("\n  ✗ Architecture DIFFERS between base and fine-tuned")
        if base_info["hidden_size"] != finetuned_info["hidden_size"]:
            print(f"    - Hidden size: {base_info['hidden_size']} → {finetuned_info['hidden_size']}")
        if base_info["num_hidden_layers"] != finetuned_info["num_hidden_layers"]:
            print(f"    - Layers: {base_info['num_hidden_layers']} → {finetuned_info['num_hidden_layers']}")

    # Compare parameter counts
    base_total = base_info["total_parameters"]
    finetuned_total = finetuned_info["total_parameters"]
    diff = finetuned_total - base_total

    print(f"\n  Parameter Changes:")
    print(f"    Base model:              {format_number(base_total)}")
    print(f"    Fine-tuned model:        {format_number(finetuned_total)}")
    if diff != 0:
        print(f"    Difference:              {format_number(abs(diff))} {'added' if diff > 0 else 'removed'}")

    # Compare trainable parameters (indicates LoRA)
    if finetuned_info["trainable_parameters"] > 0:
        lora_params = finetuned_info["trainable_parameters"]
        trainable_pct = (lora_params / base_total) * 100
        print(f"\n  Fine-tuning Method: LoRA (PEFT - Parameter-Efficient)")
        print(f"    Trainable LoRA params:   {format_number(lora_params)} ({trainable_pct:.2f}% of base)")
        print(f"    Frozen base params:      {format_number(finetuned_info['frozen_parameters'])}")
    elif diff > 0:
        print(f"\n  Fine-tuning Method: MERGED LoRA")
        print(f"    LoRA merged into base:   {format_number(diff)}")
        print(f"    All parameters frozen:   {format_number(finetuned_total)}")
        print(f"    NOTE: Model is for inference. Adapters already integrated.")


def main():
    print("\n" + "="*60)
    print("  MODEL ARCHITECTURE ANALYZER")
    print("="*60)

    # Paths
    base_model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B"
    base_model_path = "./local-DeepSeek-R1-Distill-Llama-8B"
    finetuned_model_path = "./DeepSeek-R1-Medical-FT-8b-16bts"

    # Load models
    print("\nLoading base model...")
    base_info, base_model, base_config = get_model_info(base_model_name, base_model_path)

    print("Loading fine-tuned model...")
    finetuned_info, finetuned_model, finetuned_config = get_model_info(
        "local", finetuned_model_path
    )

    # Extract LoRA info (pass model info to detect merged models)
    lora_info = get_lora_info(finetuned_model_path, finetuned_info, base_info)

    # Print results
    print_model_info("BASE MODEL: DeepSeek-R1-Distill-Llama-8B", base_info, base_model_path)
    print_model_info("FINE-TUNED MODEL: DeepSeek-R1-Medical-FT", finetuned_info, finetuned_model_path)
    print_lora_info("Fine-tuned Model (DeepSeek-R1-Medical-FT)", lora_info)
    compare_architectures(base_info, finetuned_info, "DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Medical-FT")

    # Summary
    print(f"\n{'='*60}")
    print("  KEY INSIGHTS")
    print(f"  Model: DeepSeek-R1-Medical-FT")
    print(f"{'='*60}")

    if lora_info:
        if lora_info.get("type") == "merged":
            param_diff = finetuned_info['total_parameters'] - base_info['total_parameters']
            print(f"  • Fine-tuning Method: LoRA → MERGED into base model")
            print(f"  • Extra parameters: {format_number(param_diff)} (merged LoRA weights)")
            print(f"  • All parameters frozen: {format_number(finetuned_info['total_parameters'])} (not trainable)")
            print(f"  • Use this model for: inference only, or further fine-tuning with new LoRA")
            print(f"  • To continue training: wrap with FastLanguageModel.get_peft_model() again")
        else:
            print(f"  • LoRA adapters add ~{format_number(finetuned_info['trainable_parameters'])} trainable parameters")
            print(f"  • Base model weights ({format_number(finetuned_info['frozen_parameters'])}) remain frozen")
            print(f"  • Total model parameters: {format_number(finetuned_info['total_parameters'])}")
            print(f"  • Training efficiency: {(finetuned_info['trainable_parameters'] / finetuned_info['total_parameters'] * 100):.2f}% learnable")
    print(f"\n")

    # Save to JSON
    output_file = "model_architecture_comparison.json"
    comparison = {
        "base_model": base_info,
        "finetuned_model": finetuned_info,
        "lora_adapter": lora_info,
    }

    with open(output_file, "w") as f:
        # Convert to serializable format
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, int):
                return int(obj)
            else:
                return str(obj)

        json.dump(convert_to_serializable(comparison), f, indent=2)
        print(f"  Comparison saved to: {output_file}")


if __name__ == "__main__":
    main()
