# PATCHED test_sharpness.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from data.preprocessing import load_and_preprocess_data
from sharpness import compute_hessian_sharpness

# Create a simple config
class Config:
    dataset_name = "wikitext-2-raw-v1"  # or "wikitext-103-raw-v1"
    min_text_length = 10

config = Config()

# Load and preprocess Wikitext data
train_texts, val_texts, test_texts = load_and_preprocess_data(config)

print(f"Number of training texts: {len(train_texts)}")
print(f"Number of validation texts: {len(val_texts)}")
print(f"Number of test texts: {len(test_texts)}")

# Since Wikitext has no labels, we create dummy labels
test_labels = torch.zeros(len(test_texts), dtype=torch.long)


# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model = model.to(device)
model.eval()

# ====== PATCH: Disable FlashAttention if exists ======
if hasattr(model.config, 'use_flash_attention'):
    print("Disabling FlashAttention...")
    model.config.use_flash_attention = False
# ======================================================

# Tokenize the test data
encodings = tokenizer(list(test_texts), truncation=True, padding=True, return_tensors="pt").to(device)
inputs = {"input_ids": encodings["input_ids"], "attention_mask": encodings["attention_mask"]}

# Compute Hessian sharpness
print("Computing Hessian sharpness with 20 samples...")
hessian_sharpness = compute_hessian_sharpness(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),  # <-- tuple!
    test_labels.to(device)
)

print(f"Hessian Sharpness: {hessian_sharpness:.6f}")
