from datasets import load_dataset
import random
import torch


def load_wikitext():
    return load_dataset("wikitext", "wikitext-2-raw-v1")


def evaluate_masked_lm(model, tokenizer, dataset, num_samples=100):
    correct = 0
    total = 0
    
    for example in dataset:
        tokens = tokenizer.tokenize(example['text'])
        if len(tokens) < 3:
            continue
            
        mask_idx = random.randint(1, len(tokens)-2)
        masked_token = tokens[mask_idx]
        tokens[mask_idx] = tokenizer.mask_token
        masked_sentence = tokenizer.convert_tokens_to_string(tokens)
        
        inputs = tokenizer(masked_sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()
        mask_logits = logits[0, mask_token_index, :]
        predicted_token = tokenizer.decode([torch.argmax(mask_logits).item()])
        
        if predicted_token.strip() == masked_token.strip():
            correct += 1
        total += 1
        
        if total >= num_samples:
            break
    
    return correct / total


def quick_mask_test(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0].item()
    mask_logits = logits[0, mask_token_index, :]
    top_token_id = torch.argmax(mask_logits).item()
    return tokenizer.decode([top_token_id])