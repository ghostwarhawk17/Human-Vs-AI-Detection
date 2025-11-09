# fusion_model.py
import torch, numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from config import *
from utils import set_seed

set_seed(SEED)

# Dataset for embeddings
class EmbeddingDataset(Dataset):
    def __init__(self, split, backbones=BACKBONES):
        import pandas as pd, os
        df = pd.read_csv(os.path.join(EMB_DIR, f"{split}_index.csv"))
        self.labels = df['label'].tolist()
        self.embs = [torch.load(os.path.join(EMB_DIR, f"{split}_{b.replace('/','_')}.pt")) for b in backbones]
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return [e[i] for e in self.embs], int(self.labels[i])

def collate_fn(batch):
    feats = list(zip(*[item[0] for item in batch]))
    feats = [torch.stack(f) for f in feats]
    y = torch.tensor([item[1] for item in batch])
    return feats, y

# Fusion model
class AttentionFusion(nn.Module):
    def __init__(self, emb_dim, n_models):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Linear(emb_dim//2, 1)
        )
    def forward(self, features):
        stacked = torch.stack(features, dim=1)
        b,m,h = stacked.shape
        scores = self.scorer(stacked.view(b*m,h)).view(b,m)
        w = torch.softmax(scores,1).unsqueeze(-1)
        fused = (stacked*w).sum(1)
        return fused,w.squeeze(-1)

class FusionClassifier(nn.Module):
    def __init__(self, emb_dim, n_models):
        super().__init__()
        self.fuse = AttentionFusion(emb_dim, n_models)
        self.clf = nn.Sequential(
            nn.Linear(emb_dim, FUSION_HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(FUSION_HIDDEN, 2)
        )
    def forward(self, features):
        fused,attn = self.fuse(features)
        logits = self.clf(fused)
        return logits,attn

def train_fusion(train_loader, val_loader, emb_dim, n_models):
    model = FusionClassifier(emb_dim, n_models).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    best, state = 0, None
    for ep in range(EPOCHS):
        model.train(); losses=[]
        for feats,y in tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS}"):
            feats=[f.to(DEVICE) for f in feats]; y=y.to(DEVICE)
            out,_=model(feats)
            loss=loss_fn(out,y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        val_acc,_=evaluate(model,val_loader)
        print(f"Ep{ep+1} loss={np.mean(losses):.4f} val_acc={val_acc:.4f}")
        if val_acc>best: best,val_acc=val_acc,val_acc; state=model.state_dict()
    if state: model.load_state_dict(state)
    torch.save(model.state_dict(), os.path.join(EMB_DIR,"fusion_model.pt"))
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
    pr,rc,f1,_=precision_recall_fscore_support(trues,preds,average='binary')
    return acc,(pr,rc,f1)
