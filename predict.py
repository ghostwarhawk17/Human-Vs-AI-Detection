import torch, joblib, numpy as np
from transformers import AutoModel
from config import *
from utils import safe_load_tokenizer, chunk_token_ids, pad_batch_input_ids, set_seed
from fusion_model import FusionClassifier

set_seed(SEED)

# ---------------------------
# Load models
# ---------------------------
def load_models():
    print("Loading fusion model...")
    emb_dim = torch.load(os.path.join(EMB_DIR, f"train_{BACKBONES[0].replace('/','_')}.pt")).shape[1]
    n_models = len(BACKBONES)
    fusion_model = FusionClassifier(emb_dim, n_models)
    fusion_model.load_state_dict(torch.load(os.path.join(EMB_DIR, "fusion_model.pt"), map_location=DEVICE))
    fusion_model.to(DEVICE)
    fusion_model.eval()

    print("Loading adaptive ensemble...")
    rf_path = os.path.join(EMB_DIR, "adaptive_rf.pkl")
    rf_meta = joblib.load(rf_path)

    backbone_models = []
    for b in BACKBONES:
        tokenizer = safe_load_tokenizer(b)
        model = AutoModel.from_pretrained(b).to(DEVICE)
        model.eval()
        backbone_models.append((model, tokenizer))
    return fusion_model, rf_meta, backbone_models

# ---------------------------
# Embedding extraction
# ---------------------------
def get_document_embedding(text, model, tokenizer):
    chunks = chunk_token_ids(tokenizer, text, MAX_LEN, OVERLAP)
    if len(chunks) == 0:
        chunks = [tokenizer.encode("[UNK]", add_special_tokens=True)]
    chunk_embs = []
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

def get_all_embeddings(text, backbone_models):
    return [get_document_embedding(text, m, t) for m, t in backbone_models]

# ---------------------------
# Prediction
# ---------------------------
def predict_text(text, fusion_model, rf_meta, backbone_models):
    features = [emb.to(DEVICE).unsqueeze(0) for emb in get_all_embeddings(text, backbone_models)]
    with torch.no_grad():
        logits, _ = fusion_model(features)
        fusion_prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    meta_pred = rf_meta.predict(fusion_prob.reshape(-1, 1))[0]
    meta_prob = rf_meta.predict_proba(fusion_prob.reshape(-1, 1))[0, 1]

    label = "AI-generated" if meta_pred == 1 else "Human-written"
    return label, float(meta_prob)

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    print("\nInitializing models...")
    fusion_model, rf_meta, backbone_models = load_models()
    print("Ready ✅")

    while True:
        text = input("\nEnter text (or 'exit'): ").strip()
        if text.lower() == "exit":
            break
        label, prob = predict_text(text, fusion_model, rf_meta, backbone_models)
        print(f"\nPrediction: {label} (confidence: {prob:.3f})")
