"""
This script downloads a full model repository from the Hugging Face Hub
and stores it locally on disk using snapshot_download.

Author: Prabir Guha

This is useful when you want to:
  - Work offline
  - Avoid re-downloading large models repeatedly
  - Fine-tune or run inference from a local path
  - Pin a specific model version or revision
  - Deploy models in air-gapped or restricted environments
"""

# Import the snapshot_download utility from huggingface_hub.
# This function downloads an entire repository snapshot (all files)
# including:
#   - model weights (e.g., *.safetensors, *.bin)
#   - config files (config.json, tokenizer.json, etc.)
#   - tokenizer files
#   - README, generation config, etc.
#
# It also uses Hugging Face's caching system and supports resumable downloads.
#
from huggingface_hub import snapshot_download


# -----------------------------------------------------------------------------
# 1) Specify the model repository on Hugging Face
# -----------------------------------------------------------------------------

# This is the exact repository ID on Hugging Face Hub.
# Format:
#   "<organization_or_user>/<repository_name>"
#
# In this case, we are downloading an Unsloth-optimized version of:
#   DeepSeek-R1-Distill-Llama-8B
#
# You can replace this with any public or private model repo, for example:
#   "meta-llama/Llama-2-7b-hf"
#   "mistralai/Mistral-7B-Instruct-v0.2"
#   "openai/whisper-large-v3"
#
model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B"


# -----------------------------------------------------------------------------
# 2) Choose the local directory where the model will be stored
# -----------------------------------------------------------------------------

# This is the path on your local filesystem where all model files will be saved.
# If the directory does not exist, it will be created automatically.
#
# After download, you will be able to load the model like this:
#   FastLanguageModel.from_pretrained("./local-DeepSeek-R1-Distill-Llama-8B")
#   or
#   AutoModel.from_pretrained("./local-DeepSeek-R1-Distill-Llama-8B")
#
local_dir = "./local-DeepSeek-R1-Distill-Llama-8B"


# -----------------------------------------------------------------------------
# 3) Download the model snapshot
# -----------------------------------------------------------------------------

# snapshot_download does the following:
#   - Downloads all files in the repository
#   - Verifies file integrity with hashes
#   - Supports resume if the download is interrupted
#   - Uses Hugging Face cache under the hood
#   - Optionally allows pinning a specific revision (commit / tag)
#
# Minimal usage:
#   snapshot_download(repo_id=..., local_dir=...)
#
# Advanced options you could add:
#   revision="main"          # or a specific commit hash / tag
#   token="hf_..."           # if the repo is private
#   local_dir_use_symlinks=False  # copy files instead of symlinks
#   ignore_patterns=["*.md"] # skip files you don't need
#
snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
)


# -----------------------------------------------------------------------------
# 4) Done! Print confirmation (optional)
# -----------------------------------------------------------------------------

print(f"Model '{model_name}' has been downloaded to: {local_dir}")
