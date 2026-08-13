"""
Fine-tune distilbert-base-uncased on blind-bio gender leakage
==============================================================
The last untested configuration from bio_arm.md §2.3: does a TUNED
transformer push the blind-bio gender-leak bound above the word-level
TF-IDF baseline of AUC 0.7091?

Setup (same protocol as bio_leak_strong.py / bio_leak_bert.py):
  - blind bios, train 19,200 / test 4,800, seed 42
  - cached distilbert-base-uncased, fully fine-tuned (all weights),
    CLS + dropout(0.1) + Linear(768 -> 1), BCEWithLogitsLoss
  - AdamW lr 2e-5, wd 0.01, batch 32, max_len 160
  - after each epoch: checkpoint, test AUC (full 4,800) + train AUC
    (fixed 3,000-row subsample), append a row to
    results/bio_leak_finetune.csv (partial results survive a crash)

Intended to run detached on CPU:  python bio_ft_distilbert.py > log 2>&1 &
"""

import os
import sys
import re
import time
import argparse
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "12")

warnings.filterwarnings("ignore")
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--maxlen", type=int, default=160)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--train-sub", type=int, default=3000, help="train-AUC eval subsample")
args = ap.parse_args()

RNG_SEED = 42
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "bio_leak_finetune.csv")

import torch
torch.set_num_threads(12)
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, roc_auc_score

# ── Load & clean (identical to bio_arm.py) ───────────────────────────────────
db = np.load("FairCVdb.npy", allow_pickle=True).item()
P_tr, P_te = db["Profiles Train"], db["Profiles Test"]
B_tr, B_te = db["Bios Train"], db["Bios Test"]
g_tr, g_te = P_tr[:, 1].astype(int), P_te[:, 1].astype(int)


def clean_text(txt):
    t = txt.replace("_", " ")
    t = re.sub(r"\s+'s\b", "", t)
    return re.sub(r"\s+", " ", t).strip()


bio_tr = [clean_text(t) for t in B_tr[:, 1]]
bio_te = [clean_text(t) for t in B_te[:, 1]]
print("=" * 74, flush=True)
print(f"Fine-tune distilbert-base-uncased on blind-bio gender leak "
      f"(epochs={args.epochs}, maxlen={args.maxlen}, batch={args.batch}, lr={args.lr})", flush=True)
print("=" * 74, flush=True)
print(f"  train {len(bio_tr)} | test {len(bio_te)}", flush=True)

# ── Tokenize once ─────────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc_tr = tok(bio_tr, truncation=True, max_length=args.maxlen, padding=True)
enc_te = tok(bio_te, truncation=True, max_length=args.maxlen, padding=True)
Xtr = torch.tensor(enc_tr["input_ids"]), torch.tensor(enc_tr["attention_mask"])
Xte = torch.tensor(enc_te["input_ids"]), torch.tensor(enc_te["attention_mask"])
ytr = torch.tensor(g_tr.astype(np.float32))
yte = g_te.astype(np.float32)
print(f"  tokenized train {Xtr[0].shape} test {Xte[0].shape}", flush=True)

tr_ds = TensorDataset(Xtr[0], Xtr[1], ytr)
tr_dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                   num_workers=0)   # num_workers=0 avoids fork issues when detached

# ── Model ────────────────────────────────────────────────────────────────────
import torch.nn as nn

bert = AutoModel.from_pretrained("distilbert-base-uncased")
head = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 1))


class FT(nn.Module):
    def __init__(self, bert, head):
        super().__init__()
        self.bert = bert
        self.head = head

    def forward(self, ids, mask):
        h = self.bert(ids, attention_mask=mask).last_hidden_state
        return self.head(h[:, 0]).squeeze(1)      # CLS logit


model = FT(bert, head)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
lossf = nn.BCEWithLogitsLoss()
print(f"  params: {sum(p.numel() for p in model.parameters()):,}", flush=True)


@torch.no_grad()
def eval_auc(ids, mask, labels, bs=96):
    model.eval()
    logits = []
    for s in range(0, len(ids), bs):
        logits.append(model(ids[s:s + bs], mask[s:s + bs]))
    z = torch.cat(logits)
    p = torch.sigmoid(z).numpy()
    return roc_auc_score(labels, p), accuracy_score(labels, p > 0.5)


# warm-start reference (frozen, pre-fine-tuning)
t0 = time.time()
te_auc0, _ = eval_auc(*Xte, yte)
row0 = {"epoch": 0, "train_auc": np.nan, "test_auc": te_auc0,
        "test_acc": float(accuracy_score(yte, np.ones_like(yte) * (g_te.mean() >= 0.5))),
        "secs": round(time.time() - t0)}
print(f"  [warm-start] frozen test AUC={te_auc0:.4f} ({time.time() - t0:.0f}s)", flush=True)

best = {"epoch": 0, "test_auc": te_auc0}
for ep in range(1, args.epochs + 1):
    model.train()
    ep_t0 = time.time()
    running, cnt = 0.0, 0
    for bi, (ids, mask, yb) in enumerate(tr_dl):
        opt.zero_grad()
        loss = lossf(model(ids, mask), yb)
        loss.backward()
        opt.step()
        running += float(loss) * len(yb)
        cnt += len(yb)
        if (bi + 1) % 100 == 0:
            print(f"    ep{ep} batch {bi + 1}/{len(tr_dl)} loss={running / cnt:.4f} "
                  f"({time.time() - ep_t0:.0f}s)", flush=True)
    # evaluate
    te_auc, te_acc = eval_auc(*Xte, yte)
    sub_idx = torch.randperm(len(Xtr[0]))[:args.train_sub]
    tr_auc, _ = eval_auc(Xtr[0][sub_idx], Xtr[1][sub_idx], ytr[sub_idx].numpy())
    secs = round(time.time() - ep_t0)
    ckpt = os.path.join(RESULTS_DIR, "models", f"distilbert_ft_ep{ep}.pt")
    torch.save({"bert": bert.state_dict(), "head": head.state_dict()}, ckpt)
    pd.DataFrame([{"epoch": ep, "train_auc": tr_auc, "test_auc": te_auc,
                   "test_acc": te_acc, "secs": secs, "maxlen": args.maxlen,
                   "lr": args.lr, "batch": args.batch}]).to_csv(
        CSV_PATH, mode="a", header=not os.path.exists(CSV_PATH), index=False)
    if te_auc > best["test_auc"]:
        best = {"epoch": ep, "test_auc": te_auc}
    print(f"  [epoch {ep}/{args.epochs}] loss={running / cnt:.4f}  "
          f"train AUC={tr_auc:.4f}  TEST AUC={te_auc:.4f}  acc={te_acc:.4f}  "
          f"({secs}s)  checkpoint {ckpt}", flush=True)

print(f"\n  BEST test AUC = {best['test_auc']:.4f} (epoch {best['epoch']})", flush=True)
print("  vs word TF-IDF baseline = 0.7091", flush=True)
print("  => fine-tuned transformer EXCEEDS baseline" if best["test_auc"] > 0.7091
      else "  => fine-tuned transformer does NOT exceed baseline", flush=True)
print("Done.", flush=True)
