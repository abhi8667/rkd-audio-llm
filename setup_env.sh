#!/bin/bash
# Environment setup for RKD Audio-LLM project

# Create a virtual environment if you prefer (commented out by default)
# python -m venv venv
# source venv/bin/activate

# Install PyTorch with CUDA support (adjust as necessary for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install required deep learning libraries
pip install transformers accelerate datasets evaluate
pip install peft trl unsloth bitsandbytes

# Install utility libraries
pip install wandb rich librosa soundfile pyyaml
