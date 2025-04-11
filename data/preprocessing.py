from datasets import load_dataset

def load_and_preprocess_data(config):
    dataset = load_dataset("wikitext", config.dataset_name)
    
    def preprocess(examples):
        texts = [text.strip() for text in examples["text"] 
                if len(text.strip()) > config.min_text_length 
                and not text.strip().startswith("=")]
        return {"text": texts}
    
    dataset = dataset.map(preprocess, batched=True)
    train_texts = dataset["train"]["text"][:1000]
    val_texts = dataset["validation"]["text"][:100]
    test_texts = dataset["test"]["text"][:100]
    
    return train_texts, val_texts, test_texts