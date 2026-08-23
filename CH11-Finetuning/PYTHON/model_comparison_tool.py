#!/usr/bin/env python3
"""
Comprehensive model comparison tool supporting:
- Local models (via transformers)
- Ollama models (via HTTP API)
- Hugging Face models (via hub API)
"""

import requests
import json
from typing import Dict, Optional
from pathlib import Path

try:
    from transformers import AutoConfig
    from huggingface_hub import model_info
except ImportError:
    print("Warning: transformers or huggingface_hub not installed")


class ModelAnalyzer:
    """Analyzes and compares model architectures from various sources."""

    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.models = {}

    def get_ollama_model_info(self, model_name: str) -> Optional[Dict]:
        """Fetch model info from Ollama server."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/show",
                json={"name": model_name},
                timeout=10
            )
            if response.status_code != 200:
                print(f"  ✗ Ollama model not found: {model_name}")
                return None

            data = response.json()
            model_info_dict = {
                "source": "ollama",
                "name": model_name,
                "details": data.get("details", {}),
                "model_file": data.get("modelfile", ""),
            }

            # Extract architecture details from model file
            details = data.get("details", {})
            model_info_dict.update({
                "parameter_size": details.get("parameter_size", "Unknown"),
                "quantization_level": details.get("quantization_level", "Unknown"),
                "family": details.get("family", "Unknown"),
            })

            return model_info_dict
        except requests.exceptions.ConnectionError:
            print(f"  ✗ Cannot connect to Ollama at {self.ollama_url}")
            return None
        except Exception as e:
            print(f"  ✗ Error fetching Ollama model: {e}")
            return None

    def get_huggingface_model_info(self, model_id: str) -> Optional[Dict]:
        """Fetch model info from Hugging Face Hub."""
        try:
            info = model_info(model_id)
            model_info_dict = {
                "source": "huggingface",
                "model_id": model_id,
                "downloads": info.downloads,
                "tags": info.tags,
                "pipeline_tag": info.pipeline_tag,
                "library_name": info.library_name,
            }

            # Try to load config for architecture details
            try:
                config = AutoConfig.from_pretrained(model_id)
                model_info_dict.update({
                    "architecture": config.model_type,
                    "hidden_size": getattr(config, "hidden_size", None),
                    "num_layers": getattr(config, "num_hidden_layers", None),
                    "num_heads": getattr(config, "num_attention_heads", None),
                    "vocab_size": getattr(config, "vocab_size", None),
                })
            except Exception as e:
                print(f"  Warning: Could not load config details: {e}")

            return model_info_dict
        except Exception as e:
            print(f"  ✗ Error fetching Hugging Face model: {e}")
            return None

    def get_local_model_info(self, model_path: str) -> Optional[Dict]:
        """Extract info from a local model directory."""
        try:
            config_path = Path(model_path) / "config.json"
            if not config_path.exists():
                print(f"  ✗ No config.json found at {model_path}")
                return None

            with open(config_path) as f:
                config = json.load(f)

            model_info_dict = {
                "source": "local",
                "path": model_path,
                "architecture": config.get("architectures", ["Unknown"])[0],
                "model_type": config.get("model_type", "Unknown"),
                "hidden_size": config.get("hidden_size"),
                "num_layers": config.get("num_hidden_layers"),
                "num_heads": config.get("num_attention_heads"),
                "vocab_size": config.get("vocab_size"),
                "intermediate_size": config.get("intermediate_size"),
                "max_position_embeddings": config.get("max_position_embeddings"),
            }

            # Check for LoRA adapter
            adapter_config = Path(model_path) / "adapter_config.json"
            if adapter_config.exists():
                with open(adapter_config) as f:
                    adapter_cfg = json.load(f)
                model_info_dict["lora"] = {
                    "rank": adapter_cfg.get("r"),
                    "alpha": adapter_cfg.get("lora_alpha"),
                    "dropout": adapter_cfg.get("lora_dropout"),
                    "target_modules": adapter_cfg.get("target_modules"),
                }

            return model_info_dict
        except Exception as e:
            print(f"  ✗ Error reading local model: {e}")
            return None

    def add_model(self, name: str, model_type: str, identifier: str):
        """Add a model to analyze."""
        print(f"\nAnalyzing: {name} ({model_type})")

        if model_type == "ollama":
            info = self.get_ollama_model_info(identifier)
        elif model_type == "huggingface":
            info = self.get_huggingface_model_info(identifier)
        elif model_type == "local":
            info = self.get_local_model_info(identifier)
        else:
            print(f"  ✗ Unknown model type: {model_type}")
            return

        if info:
            self.models[name] = info
            print(f"  ✓ Successfully loaded")

    def print_model_summary(self, name: str):
        """Print a summary of a model."""
        if name not in self.models:
            print(f"Model {name} not found")
            return

        info = self.models[name]
        print(f"\n{'='*70}")
        print(f"  {name.upper()}")
        print(f"{'='*70}")

        if info["source"] == "local":
            self._print_local_summary(info)
        elif info["source"] == "ollama":
            self._print_ollama_summary(info)
        elif info["source"] == "huggingface":
            self._print_huggingface_summary(info)

    def _print_local_summary(self, info: Dict):
        """Print summary for local model."""
        print(f"\n  Location: {info['path']}")
        print(f"  Type: {info.get('model_type', 'Unknown')}")
        print(f"  Architecture: {info.get('architecture', 'Unknown')}")
        print(f"\n  Model Dimensions:")
        print(f"    Hidden Size: {info.get('hidden_size', 'N/A')}")
        print(f"    Layers: {info.get('num_layers', 'N/A')}")
        print(f"    Attention Heads: {info.get('num_heads', 'N/A')}")
        print(f"    Vocabulary Size: {info.get('vocab_size', 'N/A')}")
        print(f"    Max Position Embeddings: {info.get('max_position_embeddings', 'N/A')}")

        if "lora" in info:
            print(f"\n  LoRA Adapter:")
            print(f"    Rank: {info['lora'].get('rank')}")
            print(f"    Alpha: {info['lora'].get('alpha')}")
            print(f"    Dropout: {info['lora'].get('dropout')}")
            print(f"    Target Modules: {', '.join(info['lora'].get('target_modules', []))}")

    def _print_ollama_summary(self, info: Dict):
        """Print summary for Ollama model."""
        print(f"\n  Model: {info['name']}")
        print(f"\n  Details:")
        print(f"    Parameter Size: {info.get('parameter_size', 'Unknown')}")
        print(f"    Quantization: {info.get('quantization_level', 'Unknown')}")
        print(f"    Family: {info.get('family', 'Unknown')}")

    def _print_huggingface_summary(self, info: Dict):
        """Print summary for Hugging Face model."""
        print(f"\n  Model ID: {info['model_id']}")
        print(f"  Pipeline: {info.get('pipeline_tag', 'Unknown')}")
        print(f"  Library: {info.get('library_name', 'Unknown')}")
        print(f"  Downloads: {info.get('downloads', 'Unknown')}")
        print(f"\n  Architecture:")
        print(f"    Type: {info.get('architecture', 'Unknown')}")
        print(f"    Hidden Size: {info.get('hidden_size', 'N/A')}")
        print(f"    Layers: {info.get('num_layers', 'N/A')}")
        print(f"    Attention Heads: {info.get('num_heads', 'N/A')}")
        print(f"    Vocab Size: {info.get('vocab_size', 'N/A')}")

    def compare_models(self, *model_names):
        """Compare multiple models side-by-side."""
        if not model_names or len(model_names) < 2:
            print("Need at least 2 models to compare")
            return

        print(f"\n{'='*70}")
        print(f"  ARCHITECTURE COMPARISON")
        print(f"  Models: {' vs '.join(model_names)}")
        print(f"{'='*70}\n")

        # Collect comparable fields
        models_to_compare = {name: self.models[name] for name in model_names if name in self.models}

        if len(models_to_compare) < 2:
            print("Not enough models loaded for comparison")
            return

        # Build header row with model names
        header_row = f"  {'Field':<20}"
        for model_name in model_names:
            header_row += f" | {model_name:<15}"
        print(header_row)
        print("  " + "-" * (len(header_row) - 2))

        # Extract common fields
        fields_to_compare = [
            ("Architecture", "architecture", "model_type"),
            ("Model Type", "model_type"),
            ("Hidden Size", "hidden_size"),
            ("Number of Layers", "num_layers", "num_hidden_layers"),
            ("Attention Heads", "num_heads", "num_attention_heads"),
            ("Vocabulary Size", "vocab_size"),
        ]

        for field_display, *field_keys in fields_to_compare:
            row = f"  {field_display:<20}"
            for model_name in model_names:
                if model_name not in models_to_compare:
                    row += " | N/A"
                    continue

                model = models_to_compare[model_name]
                value = None
                for key in field_keys:
                    value = model.get(key)
                    if value is not None:
                        break
                row += f" | {str(value):<15}"
            print(row)

        print()

    def export_to_json(self, filename: str = "models_comparison.json"):
        """Export all model info to JSON."""
        with open(filename, "w") as f:
            json.dump(self.models, f, indent=2)
        print(f"\nExported to: {filename}")

    def export_to_csv(self, filename: str = "models_comparison.csv"):
        """Export comparison to CSV."""
        import csv

        if not self.models:
            print("No models to export")
            return

        # Get all unique keys
        all_keys = set()
        for model_info in self.models.values():
            all_keys.update(model_info.keys())

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Model"] + sorted(all_keys))

            for model_name, model_info in self.models.items():
                row = [model_name]
                for key in sorted(all_keys):
                    value = model_info.get(key, "")
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    row.append(str(value))
                writer.writerow(row)

        print(f"\nExported to: {filename}")


def main():
    """Example usage of ModelAnalyzer."""
    analyzer = ModelAnalyzer()

    print("\n" + "="*70)
    print("  MULTI-SOURCE MODEL ANALYZER")
    print("="*70)

    # Add models from different sources
    print("\nLoading models...")

    # Ollama models
    analyzer.add_model("LLaMA 3.1 (Ollama)", "ollama", "llama3.1")
    analyzer.add_model("Nomic Embed (Ollama)", "ollama", "nomic-embed-text")

    # Hugging Face models
    analyzer.add_model("DeepSeek-R1 Base", "huggingface", "unsloth/DeepSeek-R1-Distill-Llama-8B")

    # Local fine-tuned models
    analyzer.add_model("DeepSeek Medical FT", "local", "./DeepSeek-R1-Medical-FT-8b-16bts")

    # Print individual summaries
    for model_name in analyzer.models.keys():
        analyzer.print_model_summary(model_name)

    # Compare selected models
    analyzer.compare_models(
        "DeepSeek-R1 Base",
        "DeepSeek Medical FT"
    )

    # Export results
    analyzer.export_to_json()
    analyzer.export_to_csv()

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
