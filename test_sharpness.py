# -*- coding: utf-8 -*-
from sharpness.sharpness import compute_epsilon_hessian_sharpness, power_iteration_hessian
import torch.nn.functional as F
from data.preprocessing import load_and_preprocess_data
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader

import os
os.environ["PYTORCH_USE_FLASH_ATTENTION"] = "0"

class ConfigObject:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

# Load YAML config
with open("config/base.yaml", "r") as f:
    config_dict = yaml.safe_load(f)
config = ConfigObject(config_dict)

# Load texts
train_texts, val_texts, test_texts = load_and_preprocess_data(config)
print(f"Number of training texts: {len(train_texts)}")
print(f"Number of validation texts: {len(val_texts)}")
print(f"Number of test texts: {len(test_texts)}")


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilgpt2", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize val_texts
val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding="max_length",
    max_length=32,
    return_tensors="pt"  
)

# Create TensorDataset
val_dataset = TensorDataset(val_encodings["input_ids"], val_encodings["attention_mask"])

# Create DataLoader
val_dataloader = DataLoader(val_dataset, batch_size=2)

model = AutoModelForCausalLM.from_pretrained("distilgpt2", use_auth_token=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.gradient_checkpointing_disable()
model.config.use_flash_attention = False

EPS = 1e-2
NUM_SAMPLES = 25
NUM_ITERS = 30
NUM_BATCHES = len(val_dataloader)

print(f"Computing Hessian sharpness with {NUM_ITERS} iterations and {NUM_BATCHES} batches...")
hessian_sharpness, v_max = power_iteration_hessian(
    model=model,
    dataloader=val_dataloader,
    device=device, 
    num_iters = 50,
    num_batches=NUM_BATCHES
)

print(f"Computing ε-sharpness with ε={EPS} and {NUM_SAMPLES} samples...")
sharpness, base_loss = compute_epsilon_hessian_sharpness(
    model=model,
    dataloader=val_dataloader,
    loss_fn=F.cross_entropy,
    v=v_max,
    epsilon = EPS,
    num_samples = NUM_SAMPLES,
    device=device
)

print(f"\nBase loss: {base_loss:.4f}")
print(f"Estimated ε-sharpness: {sharpness:.2f}%")
print(f"Estimated Hessian sharpness: {hessian_sharpness:.3f}")