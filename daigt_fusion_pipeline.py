import os, random, math
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer, AutoModel

# ---------------------------
# Config
# ---------------------------
DATA_CSV = "train_v2_drcat_02.csv"   # path to DAIGT CSV
TEXT_COL = "text"
LABEL_COL = "label"
BACKBONES = [
    "microsoft/deberta-v3-base",
    "albert-base-v2",
    "distilbert-base-uncased"
]
MAX_LEN = 256
OVERLAP = 64
EMB_DIR = "./embeddings_cache"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
EPOCHS = 6
FUSION_HIDDEN = 512
LEARNING_RATE = 1e-4
os.makedirs(EMB_DIR, exist_ok=True)

# ---------------------------
# Utils
# ---------------------------
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed()

# ---------------------------
# Tokenizer-safe loader
# ---------------------------
def safe_load_tokenizer(backbone):
    try:
        return AutoTokenizer.from_pretrained(backbone, use_fast=True)
    except Exception as e:
        print(f"[warning] Using slow tokenizer for {backbone} due to: {e}")
        return AutoTokenizer.from_pretrained(backbone, use_fast=False)

# ---------------------------
# Chunking + Embedding
# ---------------------------
def chunk_token_ids(tokenizer, text, max_len=MAX_LEN, overlap=OVERLAP):
    enc = tokenizer.encode(text, add_special_tokens=False)
    if len(enc) == 0:
        return [tokenizer.encode("[UNK]", add_special_tokens=True)]
    stride = max_len - overlap
    chunks = []
    for start in range(0, len(enc), stride):
        chunk_ids = enc[start:start + max_len - 2]
        chunk_with_special = tokenizer.build_inputs_with_special_tokens(chunk_ids)
        chunks.append(chunk_with_special)
        if start + max_len >= len(enc):
            break
    return chunks

def pad_batch_input_ids(batch_input_ids, pad_token_id):
    max_len = max(len(x) for x in batch_input_ids)
    input_ids, attn_masks = [], []
    for ids in batch_input_ids:
        attn = [1]*len(ids) + [0]*(max_len - len(ids))
        ids_padded = ids + [pad_token_id]*(max_len - len(ids))
        input_ids.append(ids_padded)
        attn_masks.append(attn)
    return (torch.tensor(input_ids), torch.tensor(attn_masks))

def extract_backbone_embedding(model, tokenizer, text, device=DEVICE):
    chunks = chunk_token_ids(tokenizer, text)
    if len(chunks) == 0:
        chunks = [tokenizer.encode("[UNK]", add_special_tokens=True)]
    chunk_embs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(chunks), 8):
            batch = chunks[i:i+8]
            input_ids, attn = pad_batch_input_ids(batch, tokenizer.pad_token_id)
            input_ids, attn = input_ids.to(device), attn.to(device)
            out = model(input_ids=input_ids, attention_mask=attn, return_dict=True)
            hidden = out.last_hidden_state
            attn_mask = attn.unsqueeze(-1)
            pooled = (hidden * attn_mask).sum(1) / attn_mask.sum(1).clamp(min=1e-9)
            chunk_embs.append(pooled.cpu())
    return torch.cat(chunk_embs).mean(0)

# ---------------------------
# Precompute embeddings
# ---------------------------
def precompute_embeddings(texts, labels, split_name, backbones=BACKBONES):
    idx_file = os.path.join(EMB_DIR, f"{split_name}_index.csv")
    pd.DataFrame({"label": labels}).to_csv(idx_file, index=False)
    for backbone in backbones:
        print(f"\n[embedding] {backbone}")
        tokenizer = safe_load_tokenizer(backbone)
        model = AutoModel.from_pretrained(backbone).to(DEVICE)
        model.eval()
        all_embs = []
        for text in tqdm(texts, desc=f"{split_name} - {backbone}"):
            emb = extract_backbone_embedding(model, tokenizer, text)
            all_embs.append(emb.unsqueeze(0))
        all_embs = torch.cat(all_embs, dim=0)
        torch.save(all_embs, os.path.join(EMB_DIR, f"{split_name}_{backbone.replace('/','_')}.pt"))
        del model, tokenizer, all_embs
        torch.cuda.empty_cache()

# ---------------------------
# Dataset
# ---------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, split, backbones=BACKBONES):
        df = pd.read_csv(os.path.join(EMB_DIR, f"{split}_index.csv"))
        self.labels = df['label'].tolist()
        self.embs = []
        for b in backbones:
            fname = os.path.join(EMB_DIR, f"{split}_{b.replace('/','_')}.pt")
            self.embs.append(torch.load(fname))
        self.n = len(self.labels)
    def __len__(self): return self.n
    def __getitem__(self, idx):
        return [e[idx] for e in self.embs], int(self.labels[idx])

def collate_fn(batch):
    feats = list(zip(*[item[0] for item in batch]))
    feats = [torch.stack(f) for f in feats]
    labels = torch.tensor([item[1] for item in batch])
    return feats, labels

# ---------------------------
# Fusion Model
# ---------------------------
class AttentionFusion(nn.Module):
    def __init__(self, emb_dim, n_models):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, 1)
        )
    def forward(self, features):
        stacked = torch.stack(features, dim=1)
        b, m, h = stacked.shape
        flat = stacked.view(b*m, h)
        scores = self.score(flat).view(b, m)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        fused = (stacked * weights).sum(1)
        return fused, weights.squeeze(-1)

class FusionClassifier(nn.Module):
    def __init__(self, emb_dim, n_models):
        super().__init__()
        self.fuse = AttentionFusion(emb_dim, n_models)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, FUSION_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(FUSION_HIDDEN, 2)
        )
    def forward(self, features):
        fused, attn = self.fuse(features)
        logits = self.classifier(fused)
        return logits, attn

# ---------------------------
# Training + Eval
# ---------------------------
def train_fusion(train_loader, val_loader, emb_dim, n_models):
    model = FusionClassifier(emb_dim, n_models).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = 0, None
    for ep in range(EPOCHS):
        model.train()
        losses=[]
        for feats, y in tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS}"):
            feats=[f.to(DEVICE) for f in feats]; y=y.to(DEVICE)
            out,_=model(feats)
            loss=loss_fn(out,y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        val_acc,_=evaluate(model,val_loader)
        print(f"Epoch {ep+1}: loss={np.mean(losses):.4f} val_acc={val_acc:.4f}")
        if val_acc>best_acc: best_acc,best_state=val_acc,model.state_dict()
    if best_state: model.load_state_dict(best_state)
    return model

def evaluate(model, loader):
    model.eval(); preds=[]; trues=[]
    with torch.no_grad():
        for feats,y in loader:
            feats=[f.to(DEVICE) for f in feats]
            logits,_=model(feats)
            p=torch.argmax(logits,1).cpu().numpy()
            preds+=p.tolist(); trues+=y.numpy().tolist()
    acc=accuracy_score(trues,preds)
    pr,rc,f1,_=precision_recall_fscore_support(trues,preds,average='binary',zero_division=0)
    return acc,(pr,rc,f1)

# ---------------------------
# Collect probs
# ---------------------------
def collect_probs(model, loader):
    model.eval(); probs=[]; ys=[]
    with torch.no_grad():
        for feats,y in loader:
            feats=[f.to(DEVICE) for f in feats]
            logits,_=model(feats)
            p=torch.softmax(logits,1)[:,1].cpu().numpy()
            probs+=p.tolist(); ys+=y.numpy().tolist()
    return np.array(probs).reshape(-1,1), np.array(ys)

# ---------------------------
# Main
# ---------------------------
def main():
    print("Loading dataset...")
    df=pd.read_csv(DATA_CSV)
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(f"Columns {TEXT_COL} / {LABEL_COL} not found.")
    texts=df[TEXT_COL].astype(str).tolist()
    labels=df[LABEL_COL].astype(int).tolist()

    X_temp,X_test,y_temp,y_test=train_test_split(texts,labels,test_size=0.1,random_state=SEED,stratify=labels)
    X_train,X_val,y_train,y_val=train_test_split(X_temp,y_temp,test_size=0.1111,random_state=SEED,stratify=y_temp)
    print(f"Train:{len(X_train)}  Val:{len(X_val)}  Test:{len(X_test)}")

    # Precompute embeddings
    precompute_embeddings(X_train,y_train,"train")
    precompute_embeddings(X_val,y_val,"val")
    precompute_embeddings(X_test,y_test,"test")

    train_ds=EmbeddingDataset("train")
    val_ds=EmbeddingDataset("val")
    test_ds=EmbeddingDataset("test")
    emb_dim=train_ds.embs[0].shape[1]; n_models=len(BACKBONES)
    train_dl=DataLoader(train_ds,batch_size=32,shuffle=True,collate_fn=collate_fn)
    val_dl=DataLoader(val_ds,batch_size=64,collate_fn=collate_fn)
    test_dl=DataLoader(test_ds,batch_size=64,collate_fn=collate_fn)

    # Fusion model training
    fusion_model=train_fusion(train_dl,val_dl,emb_dim,n_models)
    torch.save(fusion_model.state_dict(), os.path.join(EMB_DIR,"fusion_model.pt"))
    print("Fusion model saved.")

    # ---------------------------
    # Adaptive Ensemble (Option B)
    # ---------------------------
    print("\n=== Training Adaptive Ensemble ===")
    train_probs,y_train_meta=collect_probs(fusion_model,train_dl)
    val_probs,y_val_meta=collect_probs(fusion_model,val_dl)
    X_meta=np.vstack([train_probs,val_probs])
    y_meta=np.concatenate([y_train_meta,y_val_meta])

    rf_meta=RandomForestClassifier(n_estimators=200,random_state=SEED)
    rf_meta.fit(X_meta,y_meta)
    print("Adaptive ensemble trained.")

    test_probs,y_test_meta=collect_probs(fusion_model,test_dl)
    meta_preds=rf_meta.predict(test_probs)

    acc=accuracy_score(y_test_meta,meta_preds)
    pr,rc,f1,_=precision_recall_fscore_support(y_test_meta,meta_preds,average='binary',zero_division=0)
    cm=confusion_matrix(y_test_meta,meta_preds)
    print("\n=== Results (Fusion + RF) ===")
    print(f"Accuracy: {acc:.4f}  Precision: {pr:.4f}  Recall: {rc:.4f}  F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()
