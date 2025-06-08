from datasets import load_dataset

def load_and_preprocess_data(config):
    """
    Load and preprocess the dataset based on the provided configuration.
    Args:
        config (object): Configuration object containing dataset parameters.
    Returns:
        tuple: Preprocessed train, validation, and test texts.
    """
    dataset = load_dataset("wikitext", config.dataset_name)
    
    def preprocess(examples):
        """
        Preprocess the dataset examples by filtering and cleaning text.
        Args:
            examples (dict): Dictionary containing dataset examples.
        Returns:
            dict: Dictionary with cleaned text examples.
        """
        texts = [text.strip() for text in examples["text"] 
                if len(text.strip()) > config.min_text_length 
                and not text.strip().startswith("=")]
        return {"text": texts}
    
    dataset = dataset.map(preprocess, batched=True)

    # Select a subset of the dataset for training, validation, and testing
    train_texts = dataset["train"]["text"][:1000]
    val_texts = dataset["validation"]["text"][:100]
    test_texts = dataset["test"]["text"][:100]
    
    return train_texts, val_texts, test_texts