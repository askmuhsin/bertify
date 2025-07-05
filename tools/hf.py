from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
import torch


def load_model(model_name):
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def save_model(model, tokenizer, save_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def load_local_model(model_path):
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def predict_mask(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    mask_logits = logits[0, mask_token_index, :]
    top_token = torch.topk(mask_logits, k=1, dim=1).indices[0].tolist()
    return tokenizer.decode(top_token)


def get_config(model_name):
    return AutoConfig.from_pretrained(model_name)


def model_info(model):
    config = model.config
    return {
        'parameters': model.num_parameters(),
        'hidden_size': config.hidden_size,
        'num_layers': config.num_hidden_layers,
        'num_heads': config.num_attention_heads,
        'vocab_size': config.vocab_size
    }