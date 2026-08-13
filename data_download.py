"""
Downloads the medical reasoning SFT dataset from Hugging Face Hub
and saves it locally in multiple formats.

Author: Prabir Guha
"""

from datasets import load_dataset
import os
dataset = load_dataset("FreedomIntelligence/medical-o1-reasoning-SFT","en")

# Create a folder to save the data
save_folder = "./medical-o1-reasoning-SFT"
os.makedirs(save_folder, exist_ok=True)

# Save the dataset in multiple formats (choose what you need)
# Option 1: Save as Hugging Face dataset format (recommended)
dataset.save_to_disk(os.path.join(save_folder, "hf_format"))
# Option 2: Save as JSON
#dataset.to_json(os.path.join(save_folder, "dataset.json"))
# Option 3: Save as CSV
#dataset.to_csv(os.path.join(save_folder, "dataset.csv"))
print(f"Dataset saved to {save_folder}/")

