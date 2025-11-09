# adaptive_ensemble.py
import numpy as np, torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from config import *
from fusion_model import FusionClassifier, EmbeddingDataset, collate_fn
from torch.utils.data import DataLoader

def collect_probs(model, loader):
    model.eval(); probs=[]; ys=[]
    with torch.no_grad():
        for feats,y in loader:
            feats=[f.to(DEVICE) for f in feats]
            logits,_=model(feats)
            p=torch.softmax(logits,1)[:,1].cpu().numpy()
            probs+=p.tolist(); ys+=y.numpy().tolist()
    return np.array(probs).reshape(-1,1), np.array(ys)

def train_adaptive_ensemble():
    train_ds=EmbeddingDataset("train")
    val_ds=EmbeddingDataset("val")
    test_ds=EmbeddingDataset("test")
    emb_dim=train_ds.embs[0].shape[1]; n_models=len(BACKBONES)
    model=FusionClassifier(emb_dim,n_models)
    model.load_state_dict(torch.load(os.path.join(EMB_DIR,"fusion_model.pt"), map_location=DEVICE))
    model.to(DEVICE)

    train_dl=DataLoader(train_ds,batch_size=32,collate_fn=collate_fn)
    val_dl=DataLoader(val_ds,batch_size=64,collate_fn=collate_fn)
    test_dl=DataLoader(test_ds,batch_size=64,collate_fn=collate_fn)

    # Collect probabilities
    tr_p,y_tr=collect_probs(model,train_dl)
    v_p,y_v=collect_probs(model,val_dl)
    t_p,y_t=collect_probs(model,test_dl)
    X_meta=np.vstack([tr_p,v_p]); y_meta=np.concatenate([y_tr,y_v])

    rf=RandomForestClassifier(n_estimators=200,random_state=SEED)
    rf.fit(X_meta,y_meta)

    preds=rf.predict(t_p)
    acc=accuracy_score(y_t,preds)
    pr,rc,f1,_=precision_recall_fscore_support(y_t,preds,average='binary')
    cm=confusion_matrix(y_t,preds)
    print(f"Accuracy:{acc:.4f}  Precision:{pr:.4f}  Recall:{rc:.4f}  F1:{f1:.4f}")
    print("Confusion Matrix:\n",cm)

if __name__=="__main__":
    train_adaptive_ensemble()
