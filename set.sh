#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=skila
PY_VER=3.10

# 1) conda env
conda create -n ${ENV_NAME} python=${PY_VER} -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# 2) upgrade basic tooling
python -m pip install --upgrade pip setuptools wheel packaging

# 3) install PyTorch stack
# If your machine uses CUDA 12.x and the official torch 2.6 wheels work:
pip install \
  torch==2.6.0 \
  torchvision==0.21.0 \
  torchaudio==2.6.0

# 4) core training libs
pip install \
  transformers==4.51.0 \
  tokenizers==0.21.4 \
  accelerate==1.6.0 \
  datasets==4.0.0 \
  peft==0.15.2 \
  trl==0.17.0 \
  deepspeed==0.16.7 \
  bitsandbytes==0.45.5 \
  sentencepiece==0.2.1 \
  protobuf==6.30.2 \
  safetensors==0.5.3 \
  huggingface-hub==0.35.3 \
  evaluate==0.4.6 \
  scikit-learn==1.7.2 \
  scipy==1.16.3 \
  numpy==2.1.2 \
  pillow==11.0.0 \
  tqdm==4.67.1 \
  pyyaml==6.0.2 \
  einops==0.8.1 \
  timm==1.0.22 \
  opencv-python-headless==4.12.0.88

# 5) optional but commonly needed
pip install \
  wandb==0.19.10 \
  tensorboard==2.20.0 \
  ipdb==0.13.13 \
  sentence-transformers==5.1.2

# 6) flash-attn
# install only if nvcc + CUDA headers are available
# if this fails, comment it out first and make sure training can run without it
pip install flash-attn==2.7.1.post4 --no-build-isolation

# 7) verify
python - <<'PY'
import torch
import transformers
import accelerate
import datasets
import peft
import deepspeed

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
print("datasets:", datasets.__version__)
print("peft:", peft.__version__)
print("deepspeed:", deepspeed.__version__)
PY