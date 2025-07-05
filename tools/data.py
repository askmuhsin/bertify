from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import random


class WikiTextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]['text']
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def load_wikitext(split='train'):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    return dataset[split]


def create_masked_lm_dataset(texts, tokenizer, mask_prob=0.15):
    masked_texts = []
    labels = []
    
    for text in texts:
        if not text['text'].strip():
            continue
            
        tokens = tokenizer.tokenize(text['text'])
        if len(tokens) < 3:
            continue
            
        # Random masking
        masked_tokens = tokens.copy()
        token_labels = [-100] * len(tokens)  # -100 is ignored in loss
        
        for i, token in enumerate(tokens):
            if random.random() < mask_prob:
                token_labels[i] = tokenizer.convert_tokens_to_ids(token)
                masked_tokens[i] = tokenizer.mask_token
        
        if any(label != -100 for label in token_labels):
            masked_text = tokenizer.convert_tokens_to_string(masked_tokens)
            masked_texts.append(masked_text)
            labels.append(token_labels)
    
    return masked_texts, labels


def get_dataloader(dataset, batch_size=8, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)