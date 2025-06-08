from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

def get_model(model_name, tokenizer_len, device):
    """Initialize and configure the GPT-2 model"""
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.resize_token_embeddings(tokenizer_len)
    model = model.to(device)
    return model

def get_tokenizer(model_name):
    """Initialize and configure the tokenizer"""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Add padding token if not already present
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    return tokenizer, len(tokenizer)