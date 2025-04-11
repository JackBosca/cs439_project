import numpy as np
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def shuffle(self):
        np.random.shuffle(self.texts)
        
    def sort_by_length(self):
        self.texts.sort(key=lambda x: len(x))
        
    def __getitem__(self, idx):
        text = self.texts[idx]

        # Create encoding
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return encoding.input_ids.squeeze()