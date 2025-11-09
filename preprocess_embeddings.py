# preprocess_embeddings.py
import os, torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel
from config import *
from utils import safe_load_tokenizer, chunk_token_ids, pad_batch_input_ids, set_seed

set_seed(SEED)

def extract_backbone_embedding(model, tokenizer, text):
    chunks = chunk_token_ids(tokenizer, text, MAX_LEN, OVERLAP)
    chunk_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(chunks), 8):
            batch = chunks[i:i+8]
            input_ids, attn = pad_batch_input_ids(batch, tokenizer.pad_token_id)
            input_ids, attn = input_ids.to(DEVICE), attn.to(DEVICE)
            out = model(input_ids=input_ids, attention_mask=attn, return_dict=True)
            hidden = out.last_hidden_state
            pooled = (hidden * attn.unsqueeze(-1)).sum(1) / attn.sum(1, keepdim=True)
            chunk_embs.append(pooled.cpu())
    return torch.cat(chunk_embs).mean(0)

def precompute_embeddings(texts, labels, split_name):
    df_idx = pd.DataFrame({"label": labels})
    df_idx.to_csv(os.path.join(EMB_DIR, f"{split_name}_index.csv"), index=False)
    for backbone in BACKBONES:
        print(f"[{split_name}] Encoding with {backbone}")
        tokenizer = safe_load_tokenizer(backbone)
        model = AutoModel.from_pretrained(backbone).to(DEVICE)
        all_embs = []
        for text in tqdm(texts):
            emb = extract_backbone_embedding(model, tokenizer, text)
            all_embs.append(emb.unsqueeze(0))
        torch.save(torch.cat(all_embs), os.path.join(EMB_DIR, f"{split_name}_{backbone.replace('/','_')}.pt"))
        del model, tokenizer
        torch.cuda.empty_cache()
