"""
Stronger-model bound on blind-bio gender leakage
=================================================
Companion to bio_arm.py: how much MORE gender leaks from the REDACTED
bios when the model is stronger than word-level TF-IDF + logistic
regression (which scored AUC 0.7091)?

Tiers (same train/test split as bio_arm.py, seed 42, blind bios -> gender):
  T0  word TF-IDF (1-2 grams) + LR          -> must reproduce ~0.7091
  T1  char TF-IDF (char_wb 2-5 grams) + LR  -> char-level lexical cues
  T2  char-level CNN (PyTorch, CPU)         -> learned char morphology
  T3  distilbert-base-uncased (cached, FROZEN embeddings) + LR
      -> transformer semantic bound (feature extraction, not fine-tuned)

Leakage is evaluated on the held-out TEST set (fit on train only), so
the AUC ladder is a lower-bound estimate of how much gender a deployed
text model could recover from the dataset's "blind" bios.

Outputs: results/bio_leak_strong.csv (tiers 0-2); the transformer tier and the
         combined results/fig8_leak_ladder.png come from bio_leak_bert.py.
"""

import os
import sys
import re
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

RNG_SEED = 42
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(RNG_SEED)

# ── 0. Load & clean (identical to bio_arm.py) ───────────────────────────────
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
print("=" * 74)
print("Stronger-model bound on blind-bio gender leakage")
print("=" * 74)
print(f"  train {len(bio_tr)} | test {len(bio_te)} | female rate test {g_te.mean():.3f}")

rows = []
LADDER = []


def record(tier, feats, acc, auc, train_auc=None):
    rows.append({"tier": tier, "features": feats, "accuracy": acc,
                 "auc": auc, "train_auc": train_auc})
    LADDER.append((tier, feats, acc, auc))
    extra = f"  (train AUC {train_auc:.4f})" if train_auc else ""
    print(f"  {tier:<24s} {feats:<34s} acc={acc:.4f}  AUC={auc:.4f}{extra}")


# ── T0: word TF-IDF + LR (reproduce bio_arm.py's 0.7091) ────────────────────
print("\n[1/4] T0 word TF-IDF + LR …")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

vec_w = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.9,
                        sublinear_tf=True, lowercase=True)
Xw_tr = vec_w.fit_transform(bio_tr)
Xw_te = vec_w.transform(bio_te)
clf_w = LogisticRegression(max_iter=1000, C=1.0, random_state=RNG_SEED)
clf_w.fit(Xw_tr, g_tr)
p0 = clf_w.predict_proba(Xw_te)[:, 1]
record("T0", "word TF-IDF (1-2g) + LR", accuracy_score(g_te, clf_w.predict(Xw_te)),
       roc_auc_score(g_te, p0))
assert abs(roc_auc_score(g_te, p0) - 0.7091) < 0.02, "T0 failed to reproduce baseline"

# ── T1: char TF-IDF + LR ─────────────────────────────────────────────────────
print("\n[2/4] T1 char TF-IDF + LR …")
vec_c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5,
                        max_df=0.9, sublinear_tf=True, lowercase=True)
Xc_tr = vec_c.fit_transform(bio_tr)
Xc_te = vec_c.transform(bio_te)
clf_c = LogisticRegression(max_iter=1000, C=1.0, random_state=RNG_SEED)
clf_c.fit(Xc_tr, g_tr)
p1 = clf_c.predict_proba(Xc_te)[:, 1]
record("T1", "char TF-IDF (2-5g) + LR", accuracy_score(g_te, clf_c.predict(Xc_te)),
       roc_auc_score(g_te, p1))

# top char-ngrams for the female class (interpretability of what T1 sees)
coef_c = clf_c.coef_[0]
feats_c = np.array(vec_c.get_feature_names_out())
top_c = feats_c[np.argsort(-coef_c)[:10]]
print("    top female-side char-ngrams:", ", ".join(top_c))

# ── T2: char-level CNN (PyTorch, CPU) ────────────────────────────────────────
print("\n[3/4] T2 char-level CNN (PyTorch, CPU) …")
import torch
import torch.nn as nn

torch.set_num_threads(12)
torch.manual_seed(RNG_SEED)

MAXLEN = 700
BATCH = 128
EPOCHS = 6
EMB_DIM = 32
N_FILTERS = 64


def build_char_vocab(texts, max_vocab=300):
    from collections import Counter
    cnt = Counter()
    for t in texts:
        cnt.update(t)
    chars = [c for c, _ in cnt.most_common(max_vocab)]
    return {c: i + 2 for i, c in enumerate(chars)}   # 0=PAD, 1=UNK


def encode_chars(texts, vocab, maxlen):
    out = np.zeros((len(texts), maxlen), dtype=np.int64)
    for i, t in enumerate(texts):
        for j, ch in enumerate(t[:maxlen]):
            out[i, j] = vocab.get(ch, 1)
    return out


vocab = build_char_vocab(bio_tr)
print(f"    char vocab size {len(vocab)} | maxlen {MAXLEN}")
Xt2_tr = torch.from_numpy(encode_chars(bio_tr, vocab, MAXLEN))
Xt2_te = torch.from_numpy(encode_chars(bio_te, vocab, MAXLEN))
yt2_tr = torch.from_numpy(g_tr.astype(np.float32))
yt2_te = g_te.astype(np.float32)


class CharCNN(nn.Module):
    def __init__(self, n_chars, emb_dim, n_filters):
        super().__init__()
        self.emb = nn.Embedding(n_chars, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, n_filters, k, padding=k // 2) for k in (3, 4, 5)
        ])
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(n_filters * 3, 1)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)          # B x C x L
        h = [torch.relu(c(e)).max(dim=2).values for c in self.convs]  # B x F each
        h = torch.cat(h, dim=1)
        return self.fc(self.drop(h)).squeeze(1)


model = CharCNN(len(vocab) + 2, EMB_DIM, N_FILTERS)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
lossf = nn.BCEWithLogitsLoss()
n = len(Xt2_tr)


def train_epoch():
    model.train()
    perm = torch.randperm(n)
    tot, cnt = 0.0, 0
    for s in range(0, n, BATCH):
        idx = perm[s:s + BATCH]
        xb, yb = Xt2_tr[idx], yt2_tr[idx]
        opt.zero_grad()
        loss = lossf(model(xb), yb)
        loss.backward()
        opt.step()
        tot += float(loss) * len(idx)
        cnt += len(idx)
    return tot / cnt


def eval_auc(te_x, te_y):
    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(te_x[s:s + BATCH]) for s in range(0, len(te_x), BATCH)])
    return roc_auc_score(te_y, torch.sigmoid(logits).numpy()), \
           (torch.sigmoid(logits).numpy() > 0.5)


t0 = time.time()
tr_auc_last = 0.0
for ep in range(EPOCHS):
    loss = train_epoch()
    ta, _ = eval_auc(Xt2_tr, yt2_tr.numpy())
    tr_auc_last = ta
    print(f"    epoch {ep + 1}/{EPOCHS}  loss={loss:.4f}  train AUC={ta:.4f}  "
          f"({time.time() - t0:.0f}s)")
te_auc2, pred2 = eval_auc(Xt2_te, yt2_te)
record("T2", "char CNN (torch, CPU)", accuracy_score(yt2_te, pred2), te_auc2,
       train_auc=tr_auc_last)

# ── Exports (tiers 0-2; the transformer tier runs separately: bio_leak_bert.py) ──
pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "bio_leak_strong.csv"), index=False)
print(f"\n  wrote results/bio_leak_strong.csv ({len(rows)} rows)")
print("  Transformer tier is a separate step: python bio_leak_bert.py")
print("\n  Leakage bound ladder so far (blind bios -> gender, test AUC):")
for t, f, _, a in LADDER:
    print(f"    {t:<10s} {f:<34s} {a:.4f}")
print("\nTiers 0-2 complete.")
