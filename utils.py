# utils.py
import torch, random, numpy as np
from transformers import AutoTokenizer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_load_tokenizer(backbone):
    try:
        return AutoTokenizer.from_pretrained(backbone, use_fast=True)
    except Exception as e:
        print(f"[warning] Falling back to slow tokenizer for {backbone} — {e}")
        return AutoTokenizer.from_pretrained(backbone, use_fast=False)

def chunk_token_ids(tokenizer, text, max_len, overlap):
    enc = tokenizer.encode(text, add_special_tokens=False)
    if not enc:
        return [tokenizer.encode("[UNK]", add_special_tokens=True)]
    stride = max_len - overlap
    chunks = []
    for start in range(0, len(enc), stride):
        sub = enc[start:start+max_len-2]
        chunks.append(tokenizer.build_inputs_with_special_tokens(sub))
        if start + max_len >= len(enc):
            break
    return chunks

def pad_batch_input_ids(batch_input_ids, pad_token_id):
    max_len = max(len(x) for x in batch_input_ids)
    input_ids, masks = [], []
    for ids in batch_input_ids:
        attn = [1]*len(ids) + [0]*(max_len - len(ids))
        ids_padded = ids + [pad_token_id]*(max_len - len(ids))
        input_ids.append(ids_padded)
        masks.append(attn)
    return torch.tensor(input_ids), torch.tensor(masks)
